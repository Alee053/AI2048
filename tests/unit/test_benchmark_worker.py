"""Tests for the worker subprocess function."""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PRODUCTION_MODEL = os.path.join(
    _REPO_ROOT, "data", "models", "release",
    "Hybrid-PPO-Expectimax-v3.zip",
)


def test_seed_worker_rng_isolates_streams():
    from scripts.benchmark_worker import seed_worker_rng

    seed_worker_rng(env_seed_base=1000, worker_id=0)
    a = np.random.randint(0, 1_000_000)

    seed_worker_rng(env_seed_base=1000, worker_id=1)
    b = np.random.randint(0, 1_000_000)

    assert a != b, "Different worker_id with same env_seed_base must produce different streams"


def test_run_worker_posts_episode_result_to_queue():
    if not os.path.exists(_PRODUCTION_MODEL):
        pytest.skip("Production model not found")

    from scripts.benchmark_worker import run_worker

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    status_queue: mp.Queue = ctx.Queue()
    stop_event = ctx.Event()

    p = ctx.Process(
        target=run_worker,
        args=(
            0, _PRODUCTION_MODEL, "cpu", 0,
            [42], False, "test-run", 42,
            result_queue, status_queue, stop_event,
        ),
    )
    p.start()
    p.join(timeout=60)
    assert not p.is_alive(), "worker did not exit cleanly"

    assert result_queue.qsize() >= 1
    result = result_queue.get_nowait()
    assert result.eval_seed == 42
    assert result.episode_idx == 0
    assert result.steps > 0
