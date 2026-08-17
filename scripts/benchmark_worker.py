"""Worker subprocess for the benchmark harness.

Executed via multiprocessing.Process with the spawn context. Each
worker owns its own model, env, and searcher. Episodes are posted to
the master's result_queue. Per-worker status messages go to the
master's status_queue.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import traceback

from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker
from scripts.benchmark_io import EpisodeResult


def _flush_queue(message_queue) -> None:
    """Wait for this worker's queue feeder after its final message."""
    if mp.current_process().name == "MainProcess":
        return
    close = getattr(message_queue, "close", None)
    join_thread = getattr(message_queue, "join_thread", None)
    if close is None or join_thread is None:
        return
    close()
    join_thread()


def _flush_result_queue(result_queue) -> None:
    """Wait for result messages before acknowledging completion."""
    _flush_queue(result_queue)


def _flush_status_queue(status_queue) -> None:
    """Wait for the final lifecycle ACK to reach the master."""
    _flush_queue(status_queue)


def run_worker(
    worker_id: int,
    model_path: str,
    device: str,
    depth: int,
    eval_seeds: list,
    log_moves: bool,
    run_id: str,
    env_seed_base: int,
    result_queue: mp.Queue,
    status_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Run a worker's share of episodes.

    The lifecycle ACKs make worker completion independently verifiable by the
    master. Before each episode, posts an "episode_started" progress message
    so the master can bound an active episode. On exception, posts "failed"
    with traceback and re-raises. On stop_event, posts "stopped" and exits
    cleanly.
    """
    expected_episode_indices = [eval_seed - env_seed_base for eval_seed in eval_seeds]
    expected_eval_seeds = list(eval_seeds)
    actual_episode_indices: list[int] = []
    actual_eval_seeds: list[int] = []

    status_queue.put({
        "worker_id": worker_id,
        "run_id": run_id,
        "status": "started",
    })

    try:
        # These hooks keep abrupt-exit and no-response supervision tests
        # bounded without loading a model in the child process first.
        if os.environ.get("BENCHMARK_FORCE_SIGKILL") == "1":
            os.kill(os.getpid(), signal.SIGKILL)
        if os.environ.get("BENCHMARK_FORCE_HANG") == "1":
            while not stop_event.is_set():
                time.sleep(0.05)
            _flush_result_queue(result_queue)
            status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "stopped",
            })
            _flush_status_queue(status_queue)
            return

        bench = Benchmarker(model_path, depth > 0, depth, device)
        for eval_seed in eval_seeds:
            if stop_event.is_set():
                _flush_result_queue(result_queue)
                status_queue.put({
                    "worker_id": worker_id,
                    "run_id": run_id,
                    "status": "stopped",
                })
                _flush_status_queue(status_queue)
                return
            episode_idx = eval_seed - env_seed_base
            status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "episode_started",
                "episode_idx": episode_idx,
                "eval_seed": eval_seed,
            })
            result = bench.run_episode(
                eval_seed=eval_seed,
                log_moves=log_moves,
                run_id=run_id,
                worker_id=worker_id,
                episode_idx=episode_idx,
            )
            actual_episode_indices.append(result.episode_idx)
            actual_eval_seeds.append(result.eval_seed)
            result_queue.put(result)
        _flush_result_queue(result_queue)
        status_queue.put({
            "worker_id": worker_id,
            "run_id": run_id,
            "status": "completed",
            "expected_episode_indices": expected_episode_indices,
            "actual_episode_indices": actual_episode_indices,
            "expected_eval_seeds": expected_eval_seeds,
            "actual_eval_seeds": actual_eval_seeds,
        })
        _flush_status_queue(status_queue)
    except BaseException as e:
        try:
            status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "failed",
                "error": traceback.format_exc(),
                "exception": repr(e),
                "expected_episode_indices": expected_episode_indices,
                "actual_episode_indices": actual_episode_indices,
                "expected_eval_seeds": expected_eval_seeds,
                "actual_eval_seeds": actual_eval_seeds,
            })
            _flush_status_queue(status_queue)
        except BaseException:
            pass
        raise
