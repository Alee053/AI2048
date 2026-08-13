"""Tests for the Benchmarker class (raw-policy and search modes)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from twenty_forty_eight_ai.constants import MAX_EXPONENT


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


class _FakePolicy:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.policy = _FakePolicy()
        self.predict_calls = 0

    def predict(self, _obs, *, action_masks, deterministic):
        assert action_masks.any()
        assert deterministic is True
        self.predict_calls += 1
        return 0, None


class _FakeGame:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.max_tile = 0


class _FakeEnv:
    def __init__(self, moves_until_done=1):
        self.game = _FakeGame()
        self._moves_until_done = moves_until_done
        self._steps = 0

    def reset(self, *, seed):
        self._steps = 0
        self.game.board.fill(0)
        return np.zeros((1, 4, 4), dtype=np.int64), {}

    def action_masks(self):
        return np.array([True, False, False, False])

    def step(self, action):
        assert action == 0
        self._steps += 1
        done = self._steps == self._moves_until_done
        return np.zeros((1, 4, 4), dtype=np.int64), 0.0, done, False, {
            "merge_score": 0,
        }


class _FakeSearcher:
    def __init__(self):
        self.clear_tt_calls = 0
        self.find_best_move_calls = 0

    def clear_tt(self):
        self.clear_tt_calls += 1

    def find_best_move(self, _board, _depth, _evaluate_batch):
        self.find_best_move_calls += 1
        return {"best_move": 0, "move_scores": [0.0] * 4}


def _make_benchmarker(monkeypatch, *, use_expectimax, search_depth, env, model=None,
                      searcher=None):
    from twenty_forty_eight_ai.evaluation import benchmarker as benchmarker_module

    model = model or _FakeModel()
    monkeypatch.delenv("BENCHMARK_FORCE_CRASH", raising=False)
    monkeypatch.setattr(
        benchmarker_module.MaskablePPO,
        "load",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(benchmarker_module, "Game2048Env", lambda: env)
    if searcher is not None:
        monkeypatch.setattr(
            benchmarker_module,
            "ExpectimaxSearcher",
            lambda: searcher,
        )

    return benchmarker_module.Benchmarker(
        "unused.zip",
        use_expectimax=use_expectimax,
        search_depth=search_depth,
        device="cpu",
    ), model


def test_benchmarker_switches_loaded_policy_to_eval_mode(monkeypatch):
    bencher, model = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=_FakeEnv(),
    )

    assert bencher.model is model
    assert model.policy.eval_calls == 1


def test_benchmarker_clears_search_tt_once_per_episode(monkeypatch):
    searcher = _FakeSearcher()
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=True,
        search_depth=2,
        env=_FakeEnv(moves_until_done=2),
        searcher=searcher,
    )

    bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")
    bencher.run_episode(eval_seed=43, log_moves=False, run_id="test-run")

    assert searcher.clear_tt_calls == 2
    assert searcher.find_best_move_calls == 4


def test_benchmarker_move_records_preserve_supplied_episode_index(monkeypatch):
    from twenty_forty_eight_ai.utils import searcher as searcher_module

    monkeypatch.setattr(
        searcher_module,
        "_load_searcher_impl",
        lambda: SimpleNamespace(
            BoardEncoder=SimpleNamespace(canonicalize_board=lambda _board: 123)
        ),
    )
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=_FakeEnv(),
    )

    result = bencher.run_episode(
        eval_seed=42,
        log_moves=True,
        run_id="test-run",
        episode_idx=37,
    )

    assert [record.episode_idx for record in result.move_records] == [37]


def test_benchmarker_raw_depth_zero_uses_policy_without_search(monkeypatch):
    bencher, model = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=_FakeEnv(),
    )

    result = bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert bencher.searcher is None
    assert model.predict_calls == 1
    assert result.use_expectimax is False
    assert result.effective_depth == 0


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
    assert all(0 <= int(p) <= MAX_EXPONENT for p in parts)
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
