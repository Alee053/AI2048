from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from twenty_forty_eight_ai.env.d4_transforms import ID
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.env.game import LEFT


def _environment_with_board(board):
    env = Game2048Env()
    env.reset(seed=0)
    env.game.board = np.asarray(board, dtype=np.int32)
    env.game.generate_random = lambda: None
    env._current_d4 = ID
    return env


def test_environment_reports_zero_merge_score_for_a_move_without_merge():
    env = _environment_with_board([
        [1, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    _, _, _, _, info = env.step(LEFT)

    assert info["merge_score"] == 0


def test_environment_reports_simple_merge_score():
    env = _environment_with_board([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    _, _, _, _, info = env.step(LEFT)

    assert info["merge_score"] == 4


def test_environment_reports_multiple_merges_in_one_move():
    env = _environment_with_board([
        [1, 1, 0, 0],
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    _, _, _, _, info = env.step(LEFT)

    assert info["merge_score"] == 12


class _TelemetryModel:
    device = "cpu"

    class _Policy:
        def eval(self):
            return None

    policy = _Policy()

    def predict(self, _obs, *, action_masks, deterministic):
        assert action_masks[0]
        assert deterministic is True
        return 0, None


class _TelemetryEnv:
    action_space = SimpleNamespace(n=4)

    def __init__(self, merge_scores, *, include_merge_score=True):
        self.game = SimpleNamespace(
            board=np.zeros((4, 4), dtype=np.int32), score=0, max_tile=0,
        )
        self._merge_scores = list(merge_scores)
        self._include_merge_score = include_merge_score
        self._step = 0

    def reset(self, *, seed):
        self._step = 0
        return np.zeros((1, 4, 4), dtype=np.int64), {}

    def action_masks(self):
        return np.array([True, False, False, False], dtype=bool)

    def step(self, action):
        assert action == 0
        merge_score = self._merge_scores[self._step]
        self._step += 1
        terminated = self._step == len(self._merge_scores)
        info = {"merge_score": merge_score} if self._include_merge_score else {}
        self.game.score += merge_score
        return (
            np.zeros((1, 4, 4), dtype=np.int64),
            0.0,
            terminated,
            False,
            info,
        )


def _make_telemetry_benchmarker(monkeypatch, env):
    from twenty_forty_eight_ai.evaluation import benchmarker as benchmarker_module

    monkeypatch.setattr(benchmarker_module, "load_ppo_model", lambda *args, **kwargs: _TelemetryModel())
    monkeypatch.setattr(benchmarker_module, "Game2048Env", lambda: env)
    monkeypatch.setattr(
        benchmarker_module,
        "D4ValueEvaluator",
        lambda *args, **kwargs: object(),
    )
    return benchmarker_module.Benchmarker(
        "unused.zip", use_expectimax=False, search_depth=0, device="cpu",
    )


def test_benchmarker_mean_merge_score_matches_logged_moves(monkeypatch):
    env = _TelemetryEnv([0, 4, 12])
    benchmarker = _make_telemetry_benchmarker(monkeypatch, env)

    result = benchmarker.run_episode(
        eval_seed=42, log_moves=True, run_id="telemetry-test",
    )

    merge_scores = [move.merge_score for move in result.move_records]
    assert merge_scores == [0, 4, 12]
    assert result.mean_merge_score == pytest.approx(sum(merge_scores) / len(merge_scores))


def test_benchmarker_fails_when_merge_score_telemetry_is_missing(monkeypatch):
    benchmarker = _make_telemetry_benchmarker(
        monkeypatch, _TelemetryEnv([0], include_merge_score=False),
    )

    from twenty_forty_eight_ai.evaluation.benchmarker import BenchmarkEpisodeError

    with pytest.raises(BenchmarkEpisodeError, match="merge_score"):
        benchmarker.run_episode(
            eval_seed=42, log_moves=False, run_id="telemetry-test",
        )


def test_search_summary_has_no_alpha_beta_metric():
    from scripts.benchmark_summary import compute_summary_from_rows

    row = {
        "score": 100,
        "max_tile": 16,
        "steps": 10,
        "total_think_ms": 1.0,
        "total_nodes": 20,
        "total_batches": 2,
        "total_tt_collisions": 0,
        "total_tt_same_key_overwrites": 0,
        "total_moves_resolved": 10,
        "total_moves_unresolved": 0,
        "total_cap_hits": 0,
        "total_chance_nodes": 5,
        "total_max_nodes": 5,
        "mean_nps": 20.0,
        "mean_tt_hit_rate": 0.5,
        "mean_chance_value": 1.0,
    }

    summary = compute_summary_from_rows(
        [row], {"run_name": "test", "use_expectimax": True}, 1.0,
    )

    assert summary["benchmark_schema_version"] == "2.0.0"
    assert "avg_alpha_beta_cuts" not in summary["metrics"]


def test_episode_csv_rejects_missing_metric_instead_of_coercing_to_zero(tmp_path):
    from scripts.aggregate import _load_episode_rows
    from scripts.benchmark_io import EPISODE_COLUMNS

    row = {column: "0" for column in EPISODE_COLUMNS}
    row.update({"use_expectimax": "false", "total_nodes": ""})
    path = tmp_path / "episodes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="total_nodes"):
        _load_episode_rows(path)


def test_csv_writer_rejects_missing_episode_column(tmp_path):
    from scripts.benchmark_io import CSVWriter

    with CSVWriter(tmp_path, log_moves=False) as writer:
        with pytest.raises(ValueError, match="required columns"):
            writer.writerow_episode({"score": 0})


def test_episode_csv_rejects_removed_alpha_beta_column(tmp_path):
    from scripts.aggregate import _load_episode_rows
    from scripts.benchmark_io import EPISODE_COLUMNS

    fieldnames = [*EPISODE_COLUMNS, "total_alpha_beta_cuts"]
    row = {column: "0" for column in fieldnames}
    row["use_expectimax"] = "false"
    path = tmp_path / "episodes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="unexpected.*total_alpha_beta_cuts"):
        _load_episode_rows(path)
