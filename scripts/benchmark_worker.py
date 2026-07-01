"""Worker subprocess for the benchmark harness.

Executed via multiprocessing.Process with the spawn context. Each
worker owns its own model, env, and searcher. Episodes are posted to
the master's result_queue. Per-worker status messages go to the
master's status_queue.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np

from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker
from scripts.benchmark_io import EpisodeResult


def seed_worker_rng(env_seed_base: int, worker_id: int) -> None:
    """Seed the worker's Python numpy RNG.

    The C++ searcher's chance-node evaluation is deterministic (no
    RandomUtil calls during search), so only numpy is seeded here.
    The 10_000 offset keeps worker streams from accidentally
    aliasing with episode seeds.
    """
    np.random.seed(env_seed_base + worker_id * 10_000)


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

    On success, posts "completed" to status_queue.
    On exception, posts "failed" with traceback to status_queue and re-raises.
    On stop_event, posts "stopped" and exits cleanly.
    """
    seed_worker_rng(env_seed_base, worker_id)
    try:
        bench = Benchmarker(model_path, depth > 0, depth, device)
        for eval_seed in eval_seeds:
            if stop_event.is_set():
                status_queue.put({"worker_id": worker_id, "status": "stopped"})
                return
            episode_idx = eval_seed - env_seed_base
            result = bench.run_episode(
                eval_seed=eval_seed,
                log_moves=log_moves,
                run_id=run_id,
                worker_id=worker_id,
                episode_idx=episode_idx,
            )
            result_queue.put(result)
        status_queue.put({"worker_id": worker_id, "status": "completed"})
    except Exception as e:
        status_queue.put({
            "worker_id": worker_id,
            "status": "failed",
            "error": traceback.format_exc(),
            "exception": repr(e),
        })
        raise
