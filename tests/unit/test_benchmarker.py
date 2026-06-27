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


def test_benchmarker_search_mode_returns_episode_result(production_model_path):
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker

    bencher = Benchmarker(production_model_path, use_expectimax=True,
                          search_depth=3, device="cpu")
    result = bencher.run_episode(eval_seed=42, log_moves=False,
                                 run_id="test-run", worker_id=0)

    assert result.use_expectimax is True
    assert result.requested_depth == 3
    assert result.effective_depth == 3
    assert result.termination_reason == "board_full"
    assert result.steps > 0
    assert result.total_think_ms >= 0.0
    assert result.total_nodes >= 0


def test_benchmarker_d4_augmentation_disabled_by_default(production_model_path):
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker

    bencher = Benchmarker(production_model_path, use_expectimax=False,
                          search_depth=0, device="cpu")
    assert bencher.env.d4_augment is False


def test_benchmarker_log_moves_populates_move_records(production_model_path):
    import math
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker
    from scripts.benchmark_io import MoveRecord

    bencher = Benchmarker(production_model_path, use_expectimax=True,
                          search_depth=2, device="cpu")
    result = bencher.run_episode(eval_seed=42, log_moves=True,
                                 run_id="test-run", worker_id=0)

    assert len(result.move_records) == result.steps
    assert all(isinstance(m, MoveRecord) for m in result.move_records)

    first = result.move_records[0]
    assert 0 <= first.empty_cells_before <= 16
    assert first.max_tile_before in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    assert first.max_log_tile_before == int(math.log2(first.max_tile_before)) if first.max_tile_before > 1 else 0
    assert 0 <= first.n_legal_actions <= 4
    parts = first.board_state.split(",")
    assert len(parts) == 16
    assert all(0 <= int(p) <= 16 for p in parts)
    assert not math.isnan(first.score_up)


def test_benchmarker_log_moves_raw_policy_has_nan_scores(production_model_path):
    import math
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker

    bencher = Benchmarker(production_model_path, use_expectimax=False,
                          search_depth=0, device="cpu")
    result = bencher.run_episode(eval_seed=42, log_moves=True,
                                 run_id="test-run", worker_id=0)

    assert len(result.move_records) == result.steps
    for m in result.move_records:
        assert math.isnan(m.score_up)
        assert math.isnan(m.score_right)
        assert math.isnan(m.score_down)
        assert math.isnan(m.score_left)


def test_benchmarker_result_roundtrips_through_episode_to_row(production_model_path):
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker
    from scripts.benchmark_io import episode_to_row, EPISODE_COLUMNS

    bencher = Benchmarker(production_model_path, use_expectimax=False,
                          search_depth=0, device="cpu")
    result = bencher.run_episode(eval_seed=42, log_moves=False,
                                 run_id="test-run", worker_id=0,
                                 episode_idx=7)
    row = episode_to_row(result)
    assert set(row.keys()) == set(EPISODE_COLUMNS)
    assert row["episode_idx"] == 7
    assert row["worker_id"] == 0
