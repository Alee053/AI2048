"""Tests for the Benchmarker class (raw-policy and search modes)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from twenty_forty_eight_ai.constants import MAX_EXPONENT


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PRODUCTION_MODEL = os.path.join(
    _REPO_ROOT, "data", "archive", "v3-100m", "models", "release",
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
    def __init__(self, action=0):
        self.device = "cpu"
        self.policy = _FakePolicy()
        self.predict_calls = 0
        self.action = action

    def predict(self, _obs, *, action_masks, deterministic):
        assert action_masks.any()
        assert deterministic is True
        self.predict_calls += 1
        return self.action, None


class _FakeGame:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.max_tile = 0


class _FakeEnv:
    def __init__(self, moves_until_done=1, action_mask=None, terminated=None,
                 truncated=False, action_space_n=4):
        self.game = _FakeGame()
        self.action_space = SimpleNamespace(n=action_space_n)
        self._moves_until_done = moves_until_done
        self._action_mask = np.asarray(
            action_mask if action_mask is not None else [True, False, False, False],
        )
        self._terminated = terminated
        self._truncated = truncated
        self._steps = 0
        self.reset_seeds = []
        self.step_calls = []

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        self._steps = 0
        self.game.board.fill(0)
        return np.zeros((1, 4, 4), dtype=np.int64), {}

    def action_masks(self):
        return self._action_mask

    def step(self, action):
        assert action == 0
        self.step_calls.append(action)
        self._steps += 1
        terminated = (
            self._steps == self._moves_until_done
            if self._terminated is None else self._terminated
        )
        return np.zeros((1, 4, 4), dtype=np.int64), 0.0, terminated, self._truncated, {
            "merge_score": 0,
        }


class _FakeSearcher:
    def __init__(self, result=None, include_flags=True, omit_fields=()):
        self.clear_tt_calls = 0
        self.find_best_move_calls = 0
        self.result = result or {}
        self.include_flags = include_flags
        self.omit_fields = set(omit_fields)

    def clear_tt(self):
        self.clear_tt_calls += 1

    def find_best_move(self, _board, _depth, _evaluate_batch):
        self.find_best_move_calls += 1
        result = {
            "best_move": 0,
            "move_scores": [0.0] * 4,
            "think_ms": 0.0,
            "nodes_visited": 0,
            "batches_eval": 0,
            "tt_lookups": 0,
            "tt_hits": 0,
            "tt_collisions": 0,
            "tt_same_key_overwrites": 0,
            "moves_resolved": 0,
            "cap_hits": 0,
            "moves_unresolved": 0,
            "chance_nodes_evaluated": 0,
            "max_nodes_evaluated": 0,
            "chance_value_sum": 0.0,
            "chance_value_count": 0,
        }
        if self.include_flags:
            result.update({
                "has_legal_move": True,
                "search_complete": True,
                "failure_reason": None,
            })
        result.update(self.result)
        for field in self.omit_fields:
            result.pop(field, None)
        return result


def _make_benchmarker(monkeypatch, *, use_expectimax, search_depth, env, model=None,
                      searcher=None, max_steps=None):
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

    kwargs = {
        "use_expectimax": use_expectimax,
        "search_depth": search_depth,
        "device": "cpu",
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return benchmarker_module.Benchmarker("unused.zip", **kwargs), model


def test_benchmark_episode_error_exposes_stable_reason():
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    error = BenchmarkEpisodeError("search_incomplete", "search did not finish")

    assert error.reason == "search_incomplete"
    assert str(error) == "search did not finish"


def test_benchmarker_rejects_nonpositive_max_steps(monkeypatch):
    with pytest.raises(ValueError, match="max_steps"):
        _make_benchmarker(
            monkeypatch,
            use_expectimax=False,
            search_depth=0,
            env=_FakeEnv(),
            max_steps=0,
        )


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (4, "invalid_action"),
        (-1, "invalid_action"),
        (1.5, "invalid_action"),
        (1, "illegal_action"),
    ],
)
def test_benchmarker_rejects_invalid_policy_action_before_step(
    monkeypatch, action, reason,
):
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=_FakeEnv(),
        model=_FakeModel(action=action),
    )

    with pytest.raises(BenchmarkEpisodeError) as exc_info:
        bencher.run_episode(eval_seed=42, log_moves=True, run_id="test-run")

    assert exc_info.value.reason == reason
    assert bencher.env.step_calls == []


@pytest.mark.parametrize(
    "action_mask",
    [
        [True, False, False, False, False],
        [1, 0, 0, 0],
    ],
)
def test_benchmarker_rejects_malformed_action_mask_before_step(
    monkeypatch, action_mask,
):
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    env = _FakeEnv(action_mask=action_mask)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=env,
    )

    with pytest.raises(BenchmarkEpisodeError) as exc_info:
        bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert exc_info.value.reason == "invalid_action_mask"
    assert env.step_calls == []


@pytest.mark.parametrize(
    "searcher",
    [
        _FakeSearcher(include_flags=False),
        _FakeSearcher({"search_complete": 1, "has_legal_move": True}),
        _FakeSearcher({"search_complete": True, "has_legal_move": "yes"}),
    ],
)
def test_benchmarker_rejects_missing_or_malformed_search_flags_before_step(
    monkeypatch, searcher,
):
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    env = _FakeEnv(moves_until_done=2)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=True,
        search_depth=2,
        env=env,
        searcher=searcher,
    )

    with pytest.raises(BenchmarkEpisodeError) as exc_info:
        bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert exc_info.value.reason == "invalid_search_result"
    assert env.step_calls == []


@pytest.mark.parametrize(
    "searcher",
    [
        _FakeSearcher(omit_fields=("nodes_visited",)),
        _FakeSearcher(omit_fields=("cap_hits",)),
        _FakeSearcher(omit_fields=("moves_unresolved",)),
        _FakeSearcher({"batches_eval": "0"}),
        _FakeSearcher({"cap_hits": True}),
        _FakeSearcher({"moves_unresolved": False}),
        _FakeSearcher({"cap_hits": 1.0}),
        _FakeSearcher({"moves_unresolved": "0"}),
        _FakeSearcher({"cap_hits": -1}),
        _FakeSearcher({"moves_unresolved": -1}),
    ],
)
def test_benchmarker_rejects_invalid_search_counters_before_step(
    monkeypatch, searcher,
):
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    env = _FakeEnv(moves_until_done=2)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=True,
        search_depth=2,
        env=env,
        searcher=searcher,
    )

    with pytest.raises(BenchmarkEpisodeError) as exc_info:
        bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert exc_info.value.reason == "invalid_search_result"
    assert env.step_calls == []


@pytest.mark.parametrize(
    "move_scores",
    [
        [0.0, 0.0, 0.0],
        [0.0, float("nan"), 0.0, 0.0],
        [0.0, float("inf"), 0.0, 0.0],
    ],
)
def test_benchmarker_rejects_malformed_move_scores_before_step(
    monkeypatch, move_scores,
):
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    env = _FakeEnv(moves_until_done=2)
    searcher = _FakeSearcher({"move_scores": move_scores})
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=True,
        search_depth=2,
        env=env,
        searcher=searcher,
    )

    with pytest.raises(BenchmarkEpisodeError) as exc_info:
        bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert exc_info.value.reason == "invalid_search_result"
    assert env.step_calls == []


def test_benchmarker_accepts_negative_infinity_for_illegal_move_scores(monkeypatch):
    env = _FakeEnv(moves_until_done=1)
    searcher = _FakeSearcher({
        "move_scores": [0.0, float("-inf"), float("-inf"), float("-inf")],
    })
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=True,
        search_depth=2,
        env=env,
        searcher=searcher,
    )

    result = bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert result.steps == 1


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [
        (True, True),
        (1, False),
        (False, 0),
    ],
)
def test_benchmarker_rejects_invalid_environment_flags(
    monkeypatch, terminated, truncated,
):
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    env = _FakeEnv(terminated=terminated, truncated=truncated)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=env,
    )

    with pytest.raises(BenchmarkEpisodeError) as exc_info:
        bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert exc_info.value.reason == "invalid_env_flags"
    assert len(env.step_calls) == 1


def test_benchmarker_preserves_environment_truncated_flag(monkeypatch):
    env = _FakeEnv(terminated=False, truncated=True)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=env,
    )

    result = bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert result.termination_reason == "env_truncated"
    assert result.terminated is False
    assert result.truncated is True


@pytest.mark.parametrize(
    ("search_result", "reason"),
    [
        ({"cap_hits": 1}, "search_cap_hit"),
        ({"moves_unresolved": 1}, "search_moves_unresolved"),
        ({"search_complete": False, "failure_reason": "search_incomplete"},
         "search_incomplete"),
        ({"has_legal_move": False, "failure_reason": "no_legal_move"},
         "no_legal_move"),
        ({"best_move": -1}, "invalid_search_move"),
    ],
)
def test_benchmarker_rejects_failed_search_before_step(
    monkeypatch, search_result, reason,
):
    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    env = _FakeEnv(moves_until_done=2)
    searcher = _FakeSearcher(search_result)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=True,
        search_depth=2,
        env=env,
        searcher=searcher,
    )

    with pytest.raises(BenchmarkEpisodeError) as exc_info:
        bencher.run_episode(eval_seed=42, log_moves=True, run_id="test-run")

    assert exc_info.value.reason == reason
    assert env.step_calls == []
    assert searcher.find_best_move_calls == 1


@pytest.mark.parametrize("use_expectimax", [False, True])
def test_terminal_board_returns_zero_action_episode_without_fallback(
    monkeypatch, use_expectimax,
):
    env = _FakeEnv(action_mask=[False, False, False, False])
    searcher = _FakeSearcher()
    bencher, model = _make_benchmarker(
        monkeypatch,
        use_expectimax=use_expectimax,
        search_depth=2 if use_expectimax else 0,
        env=env,
        searcher=searcher,
    )

    result = bencher.run_episode(eval_seed=42, log_moves=True, run_id="test-run")

    assert result.steps == 0
    assert result.move_records == []
    assert result.termination_reason == "terminal_board"
    assert result.terminated is True
    assert result.truncated is False
    assert env.step_calls == []
    assert model.predict_calls == 0
    assert searcher.find_best_move_calls == 0


def test_benchmarker_preserves_environment_terminated_and_truncated_flags(monkeypatch):
    env = _FakeEnv(terminated=True, truncated=False)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=env,
    )

    result = bencher.run_episode(eval_seed=42, log_moves=False, run_id="test-run")

    assert result.termination_reason == "board_full"
    assert result.terminated is True
    assert result.truncated is False


def test_benchmarker_truncates_nonterminal_episode_at_max_steps(monkeypatch):
    env = _FakeEnv(moves_until_done=100)
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=env,
        max_steps=2,
    )

    result = bencher.run_episode(eval_seed=42, log_moves=True, run_id="test-run")

    assert result.steps == 2
    assert result.termination_reason == "max_steps"
    assert result.terminated is False
    assert result.truncated is True
    assert len(result.move_records) == 2


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


def test_benchmarker_uses_episode_seed_without_global_numpy_seed(monkeypatch):
    env = _FakeEnv()
    bencher, _ = _make_benchmarker(
        monkeypatch,
        use_expectimax=False,
        search_depth=0,
        env=env,
    )
    monkeypatch.setattr(
        np.random,
        "seed",
        lambda *_args, **_kwargs: pytest.fail(
            "benchmark must not seed global numpy RNG"
        ),
    )

    bencher.run_episode(eval_seed=314, log_moves=False, run_id="test-run")

    assert env.reset_seeds == [314]


@pytest.mark.integration
@pytest.mark.slow
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


@pytest.mark.integration
@pytest.mark.slow
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


@pytest.mark.integration
@pytest.mark.slow
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


@pytest.mark.integration
@pytest.mark.slow
def test_benchmarker_d4_augmentation_disabled_by_default(production_model_path):
    from twenty_forty_eight_ai.evaluation.benchmarker import Benchmarker

    bencher = Benchmarker(production_model_path, use_expectimax=False,
                          search_depth=0, device="cpu")
    assert bencher.env.d4_augment is False


@pytest.mark.integration
@pytest.mark.slow
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


@pytest.mark.integration
@pytest.mark.slow
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


@pytest.mark.integration
@pytest.mark.slow
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
