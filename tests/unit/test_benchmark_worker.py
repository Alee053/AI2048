"""Tests for the worker subprocess function."""
from __future__ import annotations

import multiprocessing as mp
import os

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PRODUCTION_MODEL = os.path.join(
    _REPO_ROOT, "data", "archive", "v3-100m", "models", "release",
    "Hybrid-PPO-Expectimax-v3.zip",
)


@pytest.mark.integration
@pytest.mark.slow
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


def test_run_worker_flushes_results_before_stopped_ack(monkeypatch):
    from scripts import benchmark_worker

    events = []

    class FakeQueue:
        def put(self, message):
            events.append(("status", message["status"]))

    class StoppedEvent:
        def is_set(self):
            return True

    monkeypatch.setattr(benchmark_worker, "Benchmarker", lambda *args: object())

    monkeypatch.setattr(
        benchmark_worker,
        "_flush_result_queue",
        lambda queue: events.append(("flush", "result")),
    )
    monkeypatch.setattr(
        benchmark_worker,
        "_flush_status_queue",
        lambda queue: events.append(("flush", "status")),
    )

    benchmark_worker.run_worker(
        0, "model.zip", "cpu", 0, [42], False, "run", 42,
        FakeQueue(), FakeQueue(), StoppedEvent(),
    )

    assert events == [
        ("status", "started"),
        ("flush", "result"),
        ("status", "stopped"),
        ("flush", "status"),
    ]
