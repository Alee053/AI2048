"""Master process orchestration: spawn workers, drain queues, write outputs."""
from __future__ import annotations

import multiprocessing as mp
import math
import os
import queue
import signal
import sys
import time
import uuid
from pathlib import Path

from scripts.benchmark import _tqdm_iter, build_config
from scripts.benchmark_io import (
    CSVWriter,
    episode_to_row, move_to_row, outcome_fingerprint,
)
from scripts.benchmark_worker import run_worker


ESTIMATED_STEPS_PER_EPISODE = 500
AVG_BYTES_PER_MOVE_ROW = 350
LARGE_MOVE_LOG_THRESHOLD = 5_000_000
WORKER_POLL_INTERVAL_S = 0.05
WORKER_CLEANUP_JOIN_TIMEOUT_S = 10.0
RESULT_QUEUE_GRACE_S = 0.2
STATUS_QUEUE_GRACE_S = 0.2
WORKER_EXIT_GRACE_S = 10.0
INTERRUPTED_DRAIN_WINDOW_S = 1.0
INTERRUPTED_DRAIN_MAX_MESSAGES = 100_000
CLEANUP_DRAIN_MAX_MESSAGES = 1_024


def _install_sigterm_handler():
    """Funnel SIGTERM through KeyboardInterrupt so the existing handler runs."""
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        # signal only works in main thread; ignore if not available
        pass


def _check_log_moves_guard(n_runs, log_moves, yes_large):
    if not log_moves:
        return
    est = n_runs * ESTIMATED_STEPS_PER_EPISODE
    est_bytes = est * AVG_BYTES_PER_MOVE_ROW
    print(f"Estimated moves.csv rows: {est:,} (~{est_bytes / 1e6:.1f} MB)")
    if est > LARGE_MOVE_LOG_THRESHOLD and not yes_large:
        print(f"Error: --log-moves would produce >{LARGE_MOVE_LOG_THRESHOLD:,} rows.")
        print("Re-run with --yes-large-move-log to acknowledge.")
        sys.exit(1)


def _assign_seeds(env_seed_base, n_runs, n_workers):
    seeds = [env_seed_base + i for i in range(n_runs)]
    # Contiguous chunks: each worker gets a slice [w*per, (w+1)*per).
    # This guarantees that eval_seed for episode_idx i is invariant in n_workers.
    per_worker = max(1, (n_runs + n_workers - 1) // n_workers)  # ceil
    chunks = []
    for w in range(n_workers):
        start = w * per_worker
        end = min(start + per_worker, n_runs)
        chunks.append(seeds[start:end])
    return chunks


def _drain_status_queue(status_queue):
    """Drain the status queue non-blocking. Returns list of messages."""
    return _drain_queue(status_queue)


def _drain_queue(message_queue):
    """Drain a multiprocessing queue without blocking the supervisor."""
    return _drain_queue_limited(message_queue)


def _drain_queue_limited(message_queue, max_messages=None):
    """Drain at most max_messages without blocking the supervisor."""
    msgs = []
    while max_messages is None or len(msgs) < max_messages:
        try:
            msg = message_queue.get_nowait()
        except queue.Empty:
            break
        except (EOFError, OSError, ValueError):
            break
        msgs.append(msg)
    return msgs


def _drain_queue_bounded(
    message_queue, window_s: float, max_messages: int = INTERRUPTED_DRAIN_MAX_MESSAGES,
):
    """Drain queue items for a bounded feeder-settle window."""
    deadline = time.monotonic() + window_s
    messages = []
    while len(messages) < max_messages and time.monotonic() < deadline:
        remaining = max_messages - len(messages)
        drained_messages = _drain_queue_limited(message_queue, remaining)
        drained = bool(drained_messages)
        messages.extend(drained_messages)
        if drained:
            continue
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            break
        time.sleep(min(WORKER_POLL_INTERVAL_S, remaining_time))
    return messages


def _validate_runner_args(args) -> tuple[float | None, str | None]:
    n_runs = getattr(args, "n_runs", None)
    if type(n_runs) is not int or n_runs <= 0:
        return None, f"--n-runs must be > 0, got {n_runs!r}"

    workers = getattr(args, "workers", None)
    if type(workers) is not int or workers < 1:
        return None, f"--workers must be >= 1, got {workers!r}"

    worker_timeout = getattr(args, "worker_timeout", 300.0)
    try:
        worker_timeout = float(worker_timeout)
    except (TypeError, ValueError):
        return None, f"--worker-timeout must be a finite number > 0, got {worker_timeout!r}"
    if not math.isfinite(worker_timeout) or worker_timeout <= 0:
        return None, f"--worker-timeout must be a finite number > 0, got {worker_timeout!r}"
    return worker_timeout, None


def _identity_values(value):
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    if any(type(item) is not int for item in value):
        return None
    return list(value)


def _matching_identity_values(actual, expected) -> bool:
    actual = _identity_values(actual)
    expected = list(expected)
    return (
        actual is not None
        and len(actual) == len(expected)
        and len(set(actual)) == len(actual)
        and set(actual) == set(expected)
    )


def _handle_worker_status(message, worker_states, progress_at=None) -> str | None:
    if not isinstance(message, dict):
        return "invalid worker status message"

    worker_id = message.get("worker_id")
    if type(worker_id) is not int or worker_id not in worker_states:
        return f"invalid worker_id in status message: {worker_id!r}"

    status = message.get("status")
    state = worker_states[worker_id]
    expected_run_id = state.get("run_id")
    if expected_run_id is not None and message.get("run_id") != expected_run_id:
        return f"worker {worker_id} sent unexpected run_id: {message.get('run_id')!r}"
    progress_at = time.monotonic() if progress_at is None else progress_at
    if status == "started":
        if state["started"]:
            return f"worker {worker_id} sent duplicate started ACK"
        state["started"] = True
        state["last_progress"] = progress_at
        return None

    if status == "episode_started":
        if not state["started"]:
            return f"worker {worker_id} episode started before started ACK"
        if state["completed"]:
            return f"worker {worker_id} sent episode progress after completion"
        episode_idx = message.get("episode_idx")
        eval_seed = message.get("eval_seed")
        if type(episode_idx) is not int:
            return f"worker {worker_id} sent invalid episode_idx: {episode_idx!r}"
        if type(eval_seed) is not int:
            return f"worker {worker_id} sent invalid eval_seed: {eval_seed!r}"
        expected_eval_seed = state["expected_eval_seed_by_index"].get(episode_idx)
        if expected_eval_seed is None:
            return (
                f"worker {worker_id} sent unexpected episode_idx: "
                f"{episode_idx}"
            )
        if eval_seed != expected_eval_seed:
            return (
                f"worker {worker_id} episode_idx={episode_idx} expected "
                f"eval_seed={expected_eval_seed}, got {eval_seed}"
            )
        state["last_progress"] = progress_at
        return None

    if status == "completed":
        if not state["started"]:
            return f"worker {worker_id} completed before started ACK"
        if state["completed"]:
            return f"worker {worker_id} sent duplicate completed ACK"

        expected_fields = {
            "expected_episode_indices": state["expected_episode_indices"],
            "expected_eval_seeds": state["expected_eval_seeds"],
            "actual_episode_indices": state["expected_episode_indices"],
            "actual_eval_seeds": state["expected_eval_seeds"],
        }
        for field, expected in expected_fields.items():
            if field not in message:
                return f"worker {worker_id} completed ACK missing {field}"
            if not _matching_identity_values(message[field], expected):
                return f"worker {worker_id} completed ACK has invalid {field}"
        state["completed"] = True
        state["completed_at"] = progress_at
        state["last_progress"] = progress_at
        return None

    if status == "failed":
        error = message.get("error") or message.get("exception") or "unknown error"
        return f"worker {worker_id} failed: {error}"

    if status == "stopped":
        return f"worker {worker_id} stopped before completing"

    return f"worker {worker_id} sent unknown status: {status!r}"


def _drain_interrupted_statuses(status_queue, worker_states) -> str | None:
    """Drain lifecycle ACKs after interruption, preserving worker failures."""
    failure = None
    for message in _drain_status_queue(status_queue):
        if isinstance(message, dict) and message.get("status") == "stopped":
            continue
        status_error = _handle_worker_status(
            message, worker_states, time.monotonic(),
        )
        if status_error and (
            failure is None
            or (isinstance(message, dict) and message.get("status") == "failed")
        ):
            failure = status_error
    return failure


def _result_to_row(
    result,
    expected_seed_by_index,
    seen_indices,
    seen_seeds,
    expected_run_id=None,
    expected_worker_by_index=None,
):
    try:
        episode_idx = result.episode_idx
        eval_seed = result.eval_seed
        run_id = result.run_id
        worker_id = result.worker_id
    except Exception as exc:
        return None, f"invalid episode result missing identity: {exc}"

    if type(run_id) is not str:
        return None, f"invalid run_id: {run_id!r}"
    if expected_run_id is not None and run_id != expected_run_id:
        return None, f"unexpected run_id: {run_id!r}"
    if type(worker_id) is not int:
        return None, f"invalid worker_id: {worker_id!r}"
    if type(episode_idx) is not int:
        return None, f"invalid episode_idx: {episode_idx!r}"
    if type(eval_seed) is not int:
        return None, f"invalid eval_seed: {eval_seed!r}"
    if episode_idx not in expected_seed_by_index:
        return None, f"unexpected episode_idx: {episode_idx}"
    if episode_idx in seen_indices:
        return None, f"duplicate episode_idx: {episode_idx}"
    if eval_seed in seen_seeds:
        return None, f"duplicate eval_seed: {eval_seed}"
    if eval_seed not in expected_seed_by_index.values():
        return None, f"unexpected eval_seed: {eval_seed}"
    expected_seed = expected_seed_by_index[episode_idx]
    if eval_seed != expected_seed:
        return None, (
            f"episode_idx={episode_idx} expected eval_seed={expected_seed}, "
            f"got {eval_seed}"
        )
    if expected_worker_by_index is not None:
        expected_worker = expected_worker_by_index[episode_idx]
        if worker_id != expected_worker:
            return None, (
                f"episode_idx={episode_idx} expected worker_id={expected_worker}, "
                f"got {worker_id}"
            )

    try:
        row = episode_to_row(result)
    except Exception as exc:
        return None, f"invalid episode result for episode_idx={episode_idx}: {exc}"
    seen_indices.add(episode_idx)
    seen_seeds.add(eval_seed)
    return row, None


def _validate_move_records(result, expected_run_id):
    try:
        move_records = result.move_records
        expected_worker_id = result.worker_id
        expected_episode_idx = result.episode_idx
        expected_steps = result.steps
    except Exception as exc:
        return None, f"invalid move records: {exc}"

    if not isinstance(move_records, (list, tuple)):
        return None, "invalid move records: expected a list or tuple"
    if type(expected_steps) is not int or expected_steps < 0:
        return None, f"invalid result steps: {expected_steps!r}"
    if len(move_records) != expected_steps:
        return None, (
            f"move record count {len(move_records)} does not match "
            f"result steps {expected_steps}"
        )

    seen_move_indices = set()
    for move in move_records:
        try:
            move_idx = move.move_idx
            move_run_id = move.run_id
            move_worker_id = move.worker_id
            move_episode_idx = move.episode_idx
        except Exception as exc:
            return None, f"invalid move record: {exc}"
        if type(move_run_id) is not str or move_run_id != expected_run_id:
            return None, f"unexpected move run_id: {move_run_id!r}"
        if type(move_worker_id) is not int or move_worker_id != expected_worker_id:
            return None, (
                f"unexpected move worker_id: expected {expected_worker_id}, "
                f"got {move_worker_id}"
            )
        if type(move_episode_idx) is not int or move_episode_idx != expected_episode_idx:
            return None, (
                f"unexpected move episode_idx: expected {expected_episode_idx}, "
                f"got {move_episode_idx}"
            )
        if type(move_idx) is not int:
            return None, f"invalid move_idx: {move_idx!r}"
        if move_idx < 0 or move_idx >= expected_steps:
            return None, (
                f"unexpected move_idx: {move_idx}; expected range "
                f"0..{expected_steps - 1}"
            )
        if move_idx in seen_move_indices:
            return None, f"duplicate move_idx: {move_idx}"
        seen_move_indices.add(move_idx)

    expected_move_indices = set(range(expected_steps))
    if seen_move_indices != expected_move_indices:
        return None, (
            f"move_idx set {sorted(seen_move_indices)!r} does not match "
            f"expected {sorted(expected_move_indices)!r}"
        )

    try:
        move_rows = [move_to_row(move) for move in move_records]
    except Exception as exc:
        return None, f"invalid move record serialization: {exc}"
    return move_rows, None


def _prepare_result(
    result,
    expected_seed_by_index,
    seen_indices,
    seen_seeds,
    expected_run_id,
    expected_worker_by_index,
    log_moves,
):
    row, result_error = _result_to_row(
        result,
        expected_seed_by_index,
        seen_indices,
        seen_seeds,
        expected_run_id,
        expected_worker_by_index,
    )
    if result_error:
        return None, None, result_error

    search_failure = paper_search_failure_reason(row)
    if search_failure:
        seen_indices.discard(result.episode_idx)
        seen_seeds.discard(result.eval_seed)
        return None, None, search_failure

    move_rows = []
    if log_moves:
        move_rows, move_error = _validate_move_records(result, expected_run_id)
        if move_error:
            seen_indices.discard(result.episode_idx)
            seen_seeds.discard(result.eval_seed)
            return None, None, move_error
    return row, move_rows, None


def _write_result(csv_writer, row, move_rows, train_seed):
    row["train_seed"] = train_seed
    csv_writer.writerow_episode(row)
    if move_rows:
        csv_writer.writerow_moves(move_rows)


def _drain_interrupted_results(
    result_queue,
    csv_writer,
    rows,
    seen_indices,
    seen_seeds,
    expected_seed_by_index,
    expected_run_id,
    expected_worker_by_index,
    log_moves,
    train_seed,
):
    """Persist only validated results already queued before interruption."""
    results = _drain_queue_bounded(
        result_queue,
        INTERRUPTED_DRAIN_WINDOW_S,
        INTERRUPTED_DRAIN_MAX_MESSAGES,
    )
    _consume_interrupted_results(
        results,
        csv_writer,
        rows,
        seen_indices,
        seen_seeds,
        expected_seed_by_index,
        expected_run_id,
        expected_worker_by_index,
        log_moves,
        train_seed,
    )


def _consume_interrupted_results(
    results,
    csv_writer,
    rows,
    seen_indices,
    seen_seeds,
    expected_seed_by_index,
    expected_run_id,
    expected_worker_by_index,
    log_moves,
    train_seed,
):
    for result in results:
        row, move_rows, result_error = _prepare_result(
            result,
            expected_seed_by_index,
            seen_indices,
            seen_seeds,
            expected_run_id,
            expected_worker_by_index,
            log_moves,
        )
        if result_error:
            continue
        _write_result(csv_writer, row, move_rows, train_seed)
        rows.append(row)


def _snapshot_worker_exitcodes(workers):
    exitcodes = {}
    errors = []
    for worker_id, process in workers:
        try:
            exitcodes[worker_id] = process.exitcode
        except BaseException as exc:
            errors.append(f"worker {worker_id} exitcode unavailable: {exc}")
    return exitcodes, "; ".join(errors) if errors else None


def _all_workers_completed(workers, worker_states, expected_indices, expected_seeds,
                           seen_indices, seen_seeds) -> bool:
    if seen_indices != expected_indices or seen_seeds != expected_seeds:
        return False
    if not all(state["started"] and state["completed"] for state in worker_states.values()):
        return False
    return all(not process.is_alive() and process.exitcode == 0 for _, process in workers)


def _stop_workers(workers, stop_event, drain_callback=None) -> str | None:
    errors = []
    cleanup_deadline = time.monotonic() + WORKER_CLEANUP_JOIN_TIMEOUT_S

    def drain_for_cleanup():
        if drain_callback is None:
            return
        try:
            drain_callback()
        except BaseException as exc:
            errors.append(f"cleanup drain failed: {exc}")

    def wait_for_exit(worker_id, process):
        join_failed = False
        while True:
            drain_for_cleanup()
            try:
                if not process.is_alive():
                    return False, join_failed
            except BaseException as exc:
                errors.append(f"worker {worker_id} liveness check failed: {exc}")
                return True, True
            remaining = cleanup_deadline - time.monotonic()
            if remaining <= 0:
                return True, join_failed
            try:
                process.join(timeout=min(WORKER_POLL_INTERVAL_S, remaining))
            except BaseException as exc:
                errors.append(f"worker {worker_id} join failed: {exc}")
                join_failed = True
                return True, join_failed

    try:
        stop_event.set()
    except BaseException as exc:
        errors.append(f"could not set stop event: {exc}")

    initial_join_failures = {}
    for worker_id, process in workers:
        _, join_failed = wait_for_exit(worker_id, process)
        initial_join_failures[worker_id] = join_failed

    for worker_id, process in workers:
        terminate_failed = False
        try:
            alive = process.is_alive()
        except BaseException as exc:
            errors.append(f"worker {worker_id} liveness check failed: {exc}")
            alive = True
        if alive or initial_join_failures.get(worker_id, False):
            if not alive and initial_join_failures.get(worker_id, False):
                terminate_failed = True
            try:
                if alive:
                    process.terminate()
            except BaseException as exc:
                errors.append(f"worker {worker_id} terminate failed: {exc}")
                terminate_failed = True
            if alive:
                alive_after_terminate, join_failed = wait_for_exit(worker_id, process)
            else:
                alive_after_terminate, join_failed = False, False
            needs_kill = alive_after_terminate or terminate_failed or join_failed
            if needs_kill:
                if hasattr(process, "kill"):
                    try:
                        process.kill()
                    except BaseException as exc:
                        errors.append(f"worker {worker_id} kill failed: {exc}")
                    wait_for_exit(worker_id, process)
                else:
                    errors.append(f"worker {worker_id} has no kill method")

        drain_for_cleanup()
        try:
            if process.is_alive():
                errors.append(f"worker {worker_id} remains alive after cleanup")
        except BaseException as exc:
            errors.append(
                f"worker {worker_id} liveness check failed after cleanup: {exc}"
            )

    return "; ".join(errors) if errors else None


def paper_search_failure_reason(row):
    """Return a failure reason for an incomplete or malformed search result."""
    cap_hits = row.get("total_cap_hits")
    unresolved = row.get("total_moves_unresolved")
    for field, value in (
        ("total_cap_hits", cap_hits),
        ("total_moves_unresolved", unresolved),
    ):
        if type(value) is not int or value < 0:
            return f"invalid {field}: expected a non-negative integer"
    if cap_hits > 0:
        return f"search reported {cap_hits} cap hits"
    if unresolved > 0:
        return f"search reported {unresolved} unresolved moves"
    return None


def run_benchmark(args):
    """Single-model benchmark entry point. Returns process exit code."""
    worker_timeout, validation_error = _validate_runner_args(args)
    if validation_error:
        print(f"Error: {validation_error}")
        return 1

    try:
        args.worker_timeout = worker_timeout
    except (AttributeError, TypeError):
        pass

    log_moves = bool(args.log_moves)
    yes_large = bool(args.yes_large_move_log)
    _check_log_moves_guard(args.n_runs, log_moves, yes_large)
    _install_sigterm_handler()

    run_name = args.output or f"run_{int(time.time())}"
    output_dir = Path("data/benchmarks") / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Error: output directory already exists and is non-empty: {output_dir}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.base_eval_seed is None:
        env_seed_base = int.from_bytes(os.urandom(4), "big")
        eval_seed_strategy = "random"
    else:
        env_seed_base = args.base_eval_seed
        eval_seed_strategy = "deterministic-offset"

    run_id = str(uuid.uuid4())
    t_run_start = time.perf_counter()
    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config = build_config(args, run_name, env_seed_base,
                          eval_seed_strategy, started_at_iso)
    config["run_id"] = run_id

    from twenty_forty_eight_ai.evaluation.benchmarker import md5_of_file
    config["model_md5"] = md5_of_file(args.model_path) or ""

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    status_queue: mp.Queue = ctx.Queue()
    stop_event = ctx.Event()

    worker_assignments = _assign_seeds(env_seed_base, args.n_runs, args.workers)
    worker_states = {}
    expected_worker_by_index = {}
    for worker_id, seeds in enumerate(worker_assignments):
        if seeds:
            expected_episode_indices = [
                seed - env_seed_base for seed in seeds
            ]
            for episode_idx in expected_episode_indices:
                expected_worker_by_index[episode_idx] = worker_id
            worker_states[worker_id] = {
                "run_id": run_id,
                "expected_episode_indices": expected_episode_indices,
                "expected_eval_seeds": list(seeds),
                "expected_eval_seed_by_index": dict(
                    zip(expected_episode_indices, seeds),
                ),
                "started": False,
                "completed": False,
                "last_progress": None,
                "completed_at": None,
                "exit_seen_at": None,
            }

    expected_indices = set(range(args.n_runs))
    expected_seeds = {env_seed_base + index for index in range(args.n_runs)}
    expected_seed_by_index = {
        index: env_seed_base + index for index in range(args.n_runs)
    }

    csv_writer = CSVWriter(output_dir, log_moves=log_moves)
    csv_writer.write_config(config)

    rows: list = []
    seen_indices: set[int] = set()
    seen_seeds: set[int] = set()
    workers = []
    interrupted = False
    failed = False
    completed = False
    cleanup_error = None
    failure_msg = ""
    t_loop_start = time.perf_counter()

    try:
        for wid, seeds in enumerate(worker_assignments):
            if not seeds:
                continue
            process = ctx.Process(
                target=run_worker,
                args=(
                    wid, args.model_path, args.device, args.depth,
                    seeds, log_moves, run_id, env_seed_base,
                    result_queue, status_queue, stop_event,
                ),
            )
            workers.append((wid, process))
            try:
                process.start()
            except Exception as exc:
                failed = True
                failure_msg = f"could not start worker {wid}: {exc}"
                break
            worker_states[wid]["last_progress"] = time.monotonic()

        if not failed:
            with _tqdm_iter(total=args.n_runs, desc="Benchmarking", unit="game") as pbar:
                while True:
                    for message in _drain_status_queue(status_queue):
                        status_error = _handle_worker_status(
                            message, worker_states, time.monotonic(),
                        )
                        if status_error:
                            failed = True
                            failure_msg = status_error
                            break
                    if failed:
                        break

                    for result in _drain_queue(result_queue):
                        row, move_rows, result_error = _prepare_result(
                            result,
                            expected_seed_by_index,
                            seen_indices,
                            seen_seeds,
                            run_id,
                            expected_worker_by_index,
                            log_moves,
                        )
                        if result_error:
                            failed = True
                            failure_msg = result_error
                            break
                        worker_states[result.worker_id]["last_progress"] = time.monotonic()
                        _write_result(
                            csv_writer,
                            row,
                            move_rows,
                            getattr(args, "train_seed", None),
                        )
                        rows.append(row)
                        pbar.update(1)
                        pbar.set_postfix({
                            "score": result.score,
                            "max": result.max_tile,
                            "nps": f"{result.mean_nps:.0f}",
                        })
                    if failed:
                        break

                    worker_runtime = {}
                    runtime_checked_at = time.monotonic()
                    for worker_id, process in workers:
                        try:
                            alive = process.is_alive()
                            exitcode = process.exitcode
                        except Exception as exc:
                            failed = True
                            failure_msg = f"could not inspect worker {worker_id}: {exc}"
                            break
                        worker_runtime[worker_id] = (alive, exitcode)
                        if (
                            not alive
                            and exitcode == 0
                            and worker_states[worker_id]["exit_seen_at"] is None
                        ):
                            worker_states[worker_id]["exit_seen_at"] = time.monotonic()
                        if exitcode is not None and exitcode != 0:
                            failed = True
                            failure_msg = (
                                f"worker {worker_id} exited with code {exitcode}"
                            )
                            break
                        if not alive and exitcode is not None:
                            if not worker_states[worker_id]["completed"]:
                                if (
                                    runtime_checked_at
                                    - worker_states[worker_id]["exit_seen_at"]
                                    >= STATUS_QUEUE_GRACE_S
                                ):
                                    failed = True
                                    failure_msg = (
                                        f"worker {worker_id} exited cleanly without a "
                                        "valid completed ACK"
                                    )
                                    break
                    if failed:
                        break

                    now = time.monotonic()
                    for worker_id, (alive, exitcode) in worker_runtime.items():
                        state = worker_states[worker_id]
                        if state["completed"] and not alive and exitcode == 0:
                            missing_indices = [
                                index
                                for index in state["expected_episode_indices"]
                                if index not in seen_indices
                            ]
                            missing_seeds = [
                                expected_seed_by_index[index]
                                for index in missing_indices
                            ]
                            if missing_indices:
                                if (
                                    now - state["exit_seen_at"]
                                    >= RESULT_QUEUE_GRACE_S
                                ):
                                    failed = True
                                    failure_msg = (
                                        f"worker {worker_id} completed ACK but missing "
                                        f"episode_idx={missing_indices} "
                                        f"eval_seed={missing_seeds}"
                                    )
                                    break
                            continue
                        if state["completed"]:
                            if (
                                now - state["completed_at"]
                                >= WORKER_EXIT_GRACE_S
                            ):
                                failed = True
                                failure_msg = (
                                    f"worker {worker_id} completed ACK but did not "
                                    f"exit cleanly within exit grace "
                                    f"{WORKER_EXIT_GRACE_S:g}s"
                                )
                                break
                            continue
                        if not state["completed"] and not alive and exitcode == 0:
                            continue

                        last_progress = state["last_progress"]
                        if (
                            last_progress is not None
                            and now - last_progress >= worker_timeout
                        ):
                            failed = True
                            failure_msg = (
                                f"worker {worker_id} inactive for "
                                f"{now - last_progress:.3f}s "
                                f"(timeout {worker_timeout:g}s)"
                            )
                            break
                    if failed:
                        break

                    if _all_workers_completed(
                        workers,
                        worker_states,
                        expected_indices,
                        expected_seeds,
                        seen_indices,
                        seen_seeds,
                    ):
                        # A result put just before a completed ACK can still be
                        # waiting on the queue feeder. Drain once more before
                        # accepting success so duplicate/late rows are rejected.
                        for message in _drain_status_queue(status_queue):
                            status_error = _handle_worker_status(
                                message, worker_states, time.monotonic(),
                            )
                            if status_error:
                                failed = True
                                failure_msg = status_error
                                break
                        if not failed:
                            for result in _drain_queue(result_queue):
                                _, _, result_error = _prepare_result(
                                    result,
                                    expected_seed_by_index,
                                    seen_indices,
                                    seen_seeds,
                                    run_id,
                                    expected_worker_by_index,
                                    log_moves,
                                )
                                if result_error:
                                    failed = True
                                    failure_msg = result_error
                                    break
                        if not failed and _all_workers_completed(
                            workers,
                            worker_states,
                            expected_indices,
                            expected_seeds,
                            seen_indices,
                            seen_seeds,
                        ):
                            completed = True
                            break
                        if failed:
                            break

                    time.sleep(WORKER_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        failed = True
        failure_msg = f"benchmark supervisor error: {exc}"
    finally:
        interruption_requested = interrupted
        pre_cleanup_status_error = None
        if interruption_requested:
            pre_cleanup_status_error = _drain_interrupted_statuses(
                status_queue, worker_states,
            )
        pre_cleanup_exitcodes = {}
        if interruption_requested:
            pre_cleanup_exitcodes, exitcode_error = _snapshot_worker_exitcodes(workers)
            if exitcode_error:
                failed = True
                interrupted = False
                failure_msg = exitcode_error
            else:
                failed_workers = [
                    (worker_id, exitcode)
                    for worker_id, exitcode in pre_cleanup_exitcodes.items()
                    if exitcode not in (None, 0)
                ]
                if failed_workers:
                    worker_id, exitcode = failed_workers[0]
                    failed = True
                    interrupted = False
                    failure_msg = (
                        f"worker {worker_id} exited with code {exitcode}"
                    )
        def cleanup_drain():
            results = _drain_queue_limited(
                result_queue, CLEANUP_DRAIN_MAX_MESSAGES,
            )
            if interrupted:
                _consume_interrupted_results(
                    results,
                    csv_writer,
                    rows,
                    seen_indices,
                    seen_seeds,
                    expected_seed_by_index,
                    run_id,
                    expected_worker_by_index,
                    log_moves,
                    getattr(args, "train_seed", None),
                )

        try:
            cleanup_error = _stop_workers(
                workers,
                stop_event,
                drain_callback=cleanup_drain,
            )
        except BaseException as exc:
            cleanup_error = f"cleanup raised {exc}"
        post_cleanup_status_error = None
        if interruption_requested:
            post_cleanup_status_error = _drain_interrupted_statuses(
                status_queue, worker_states,
            )
            status_error = post_cleanup_status_error or pre_cleanup_status_error
            if status_error:
                failed = True
                completed = False
                interrupted = False
                failure_msg = status_error
        if cleanup_error:
            failed = True
            completed = False
            failure_msg = f"cleanup failed: {cleanup_error}"
        elif interrupted:
            _drain_interrupted_results(
                result_queue,
                csv_writer,
                rows,
                seen_indices,
                seen_seeds,
                expected_seed_by_index,
                run_id,
                expected_worker_by_index,
                log_moves,
                getattr(args, "train_seed", None),
            )
        csv_writer.close()

    total_time_s = time.perf_counter() - t_loop_start

    finished_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_wall_time_s = time.perf_counter() - t_run_start
    config["finished_at_iso"] = finished_at_iso
    config["total_wall_time_s"] = total_wall_time_s
    config["n_completed"] = len(rows)

    if not failed and not interrupted and not completed:
        failed = True
        failure_msg = failure_msg or "benchmark ended without a valid completion"

    if failed:
        config["status"] = "failed"
        config["interrupted"] = False
    elif interrupted:
        config["status"] = "interrupted"
        config["interrupted"] = True
    else:
        config["status"] = "completed"
        config["interrupted"] = False

    if config["status"] == "completed":
        config["outcome_fingerprint"] = outcome_fingerprint(rows)

    from scripts.benchmark_summary import compute_summary_from_rows
    summary = compute_summary_from_rows(rows, config, total_time_s)
    summary["status"] = config["status"]
    summary["interrupted"] = config["interrupted"]
    summary["n_completed"] = len(rows)
    summary["n_runs_requested"] = args.n_runs
    if failed:
        summary["error"] = failure_msg

    csv_writer.write_summary(summary)
    csv_writer.write_config(config)

    print(f"\nWrote {len(rows)} episodes to {output_dir}")
    print(f"Status: {config['status']}, total_time_s: {total_time_s:.1f}")
    return 2 if failed or interrupted else 0
