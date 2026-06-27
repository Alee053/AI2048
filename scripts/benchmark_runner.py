"""Master process orchestration: spawn workers, drain queues, write outputs."""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
import uuid
from pathlib import Path

from scripts.benchmark import _tqdm_iter, build_config
from scripts.benchmark_io import (
    CSVWriter,
    episode_to_row, move_to_row,
)
from scripts.benchmark_worker import run_worker


ESTIMATED_STEPS_PER_EPISODE = 500
AVG_BYTES_PER_MOVE_ROW = 350
LARGE_MOVE_LOG_THRESHOLD = 5_000_000


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
    return [seeds[w::n_workers] for w in range(n_workers)]


def _drain_status_queue(status_queue):
    """Drain the status queue non-blocking. Returns list of messages."""
    msgs = []
    while True:
        try:
            msg = status_queue.get_nowait()
        except Exception:
            break
        msgs.append(msg)
    return msgs


def run_benchmark(args):
    """Single-model benchmark entry point. Returns process exit code."""
    log_moves = bool(args.log_moves)
    yes_large = bool(args.yes_large_move_log)
    _check_log_moves_guard(args.n_runs, log_moves, yes_large)
    _install_sigterm_handler()

    if args.workers < 1:
        print(f"Error: --workers must be >= 1, got {args.workers}")
        return 1

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

    workers = []
    for wid, seeds in enumerate(worker_assignments):
        if not seeds:
            continue
        p = ctx.Process(
            target=run_worker,
            args=(
                wid, args.model_path, args.device, args.depth,
                seeds, log_moves, run_id, env_seed_base,
                result_queue, status_queue, stop_event,
            ),
        )
        p.start()
        workers.append(p)

    csv_writer = CSVWriter(output_dir, log_moves=log_moves)
    csv_writer.write_config(config)

    rows: list = []
    interrupted = False
    failed = False
    failure_msg = ""
    t_loop_start = time.perf_counter()

    try:
        with _tqdm_iter(total=args.n_runs, desc="Benchmarking", unit="game") as pbar:
            while len(rows) < args.n_runs:
                msgs = _drain_status_queue(status_queue)
                for msg in msgs:
                    if msg.get("status") == "failed":
                        failed = True
                        failure_msg = msg.get("error", "unknown")
                        stop_event.set()
                        break
                if failed:
                    break

                try:
                    result = result_queue.get(timeout=1.0)
                except Exception:
                    if stop_event.is_set():
                        break
                    continue

                row = episode_to_row(result)
                rows.append(row)
                csv_writer.writerow_episode(row)
                if log_moves and result.move_records:
                    csv_writer.writerow_moves(
                        [move_to_row(m) for m in result.move_records]
                    )
                pbar.update(1)
                pbar.set_postfix({
                    "score": result.score,
                    "max": result.max_tile,
                    "nps": f"{result.mean_nps:.0f}",
                })
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()

    total_time_s = time.perf_counter() - t_loop_start

    stop_event.set()
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    while True:
        try:
            result = result_queue.get_nowait()
            row = episode_to_row(result)
            rows.append(row)
            csv_writer.writerow_episode(row)
            if log_moves and result.move_records:
                csv_writer.writerow_moves(
                    [move_to_row(m) for m in result.move_records]
                )
        except Exception:
            break

    csv_writer.close()

    finished_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_wall_time_s = time.perf_counter() - t_run_start
    config["finished_at_iso"] = finished_at_iso
    config["total_wall_time_s"] = total_wall_time_s
    config["n_completed"] = len(rows)

    if failed:
        config["status"] = "failed"
        config["interrupted"] = False
    elif interrupted:
        config["status"] = "interrupted"
        config["interrupted"] = True
    else:
        config["status"] = "completed"
        config["interrupted"] = False

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
    return 2 if failed else 0