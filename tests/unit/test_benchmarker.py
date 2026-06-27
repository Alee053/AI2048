"""Tests for the Benchmarker class (raw-policy and search modes)."""
from __future__ import annotations

import os

import numpy as np
import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PRODUCTION_MODEL = os.path.join(
    _REPO_ROOT, "data", "models", "release",
    "Hybrid-PPO-Expectimax-v3.zip",
)


@pytest.fixture(scope="module")
def production_model_path():
    if not os.path.exists(_PRODUCTION_MODEL):
        pytest.skip(f"Production model not found at {_PRODUCTION_MODEL}")
    return _PRODUCTION_MODEL


def test_benchmarker_raw_policy_returns_episode_result(production_model_path):
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker
    from scripts.benchmark_io import EpisodeResult

    bencher = Benchmarker(production_model_path, use_expectimax=False,
                          search_depth=0, device="cpu")
    result = bencher.run_episode(eval_seed=42, log_moves=False,
                                 run_id="test-run", worker_id=0)

    assert isinstance(result, EpisodeResult)
    assert result.use_expectimax is False
    assert result.requested_depth == 0
    assert result.effective_depth == 0
    assert result.termination_reason == "board_full"
    assert result.eval_seed == 42
    assert result.steps > 0
    assert result.score >= 0
    assert 0 < result.max_tile <= 32768
    assert result.episode_time_s > 0


def test_benchmarker_win_booleans_match_max_tile(production_model_path):
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker

    bencher = Benchmarker(production_model_path, use_expectimax=False,
                          search_depth=0, device="cpu")
    result = bencher.run_episode(eval_seed=42, log_moves=False,
                                 run_id="test-run", worker_id=0)
    assert result.win_1024 == (result.max_tile >= 1024)
    assert result.win_2048 == (result.max_tile >= 2048)
    assert result.win_4096 == (result.max_tile >= 4096)
    assert result.win_8192 == (result.max_tile >= 8192)
