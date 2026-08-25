from __future__ import annotations

import math
import csv

import pytest

from scripts.benchmark_statistics import (
    EXPECTED_TRAINING_SEEDS,
    STATISTICAL_METRICS,
    compute_model_metrics,
    confidence_interval,
    cross_depth_effects,
    model_level_confidence_intervals,
    paired_d4_no_d4_effects,
    student_t_critical_95,
    validate_complete_design,
)


def _episodes(scores, max_tiles=None):
    max_tiles = max_tiles or [0] * len(scores)
    return [
        {
            "score": score,
            "steps": 10,
            "max_tile": max_tile,
            "total_think_ms": 2.0,
            "total_nodes": 3,
            "total_batches": 1,
            "mean_tt_hit_rate": 0.5,
            "mean_nps": 10.0,
        }
        for score, max_tile in zip(scores, max_tiles)
    ]


def _model(condition, seed, depth, scores, max_tiles=None):
    return compute_model_metrics(
        {"condition": condition, "training_seed": seed, "depth": depth},
        _episodes(scores, max_tiles),
    )


def _complete_models(depths=(0,)):
    models = []
    for depth in depths:
        for seed in EXPECTED_TRAINING_SEEDS:
            models.append(_model("d4", seed, depth, [10 + seed + depth]))
            models.append(_model("no_d4", seed, depth, [8 + seed + depth]))
    return models


def test_student_t_ci_uses_model_df_and_known_critical_value():
    interval = confidence_interval([2.0, 3.0, 3.0, 4.0])

    assert student_t_critical_95(3) == pytest.approx(3.1824463, rel=1e-6)
    assert interval["n"] == 4
    assert interval["df"] == 3
    assert interval["mean"] == pytest.approx(3.0)
    assert interval["sd"] == pytest.approx(math.sqrt(2 / 3))
    assert interval["ci95_low"] == pytest.approx(1.7005, abs=1e-3)
    assert interval["ci95_high"] == pytest.approx(4.2995, abs=1e-3)


def test_model_metrics_aggregate_win_rates_and_keep_episode_ci_descriptive():
    model = _model(
        "d4", 0, 0, [0, 100], max_tiles=[2048, 4096],
    )

    assert model["analysis_unit"] == "training_seed"
    assert model["mean_score"] == pytest.approx(50.0)
    assert model["win_rate_1024"] == pytest.approx(1.0)
    assert model["win_rate_2048"] == pytest.approx(1.0)
    assert model["win_rate_4096"] == pytest.approx(0.5)
    assert model["win_rate_8192"] == pytest.approx(0.0)
    assert model["episode_uncertainty_unit"] == "episode_descriptive_only"
    assert model["episode_score_ci95_low_descriptive"] is not None


def test_win_rate_uncertainty_is_across_training_models():
    models = []
    for seed in EXPECTED_TRAINING_SEEDS:
        max_tile = 2048 if seed else 1024
        models.append(_model("d4", seed, 0, [10], max_tiles=[max_tile]))
        models.append(_model("no_d4", seed, 0, [10], max_tiles=[1024]))

    rows = model_level_confidence_intervals(models)
    win_rate = next(
        row for row in rows
        if row["condition"] == "d4" and row["metric"] == "win_rate_2048"
    )

    assert win_rate["mean"] == pytest.approx(0.75)
    assert win_rate["n_models"] == 4
    assert win_rate["df"] == 3
    assert win_rate["unit"] == "training_seed"


def test_paired_d4_no_d4_effects_use_one_delta_per_training_seed():
    models = []
    for seed, delta in zip(EXPECTED_TRAINING_SEEDS, [2, 3, 3, 4]):
        models.append(_model("d4", seed, 0, [10 + delta]))
        models.append(_model("no_d4", seed, 0, [10]))

    effects, deltas = paired_d4_no_d4_effects(models)
    score_effect = next(row for row in effects if row["metric"] == "mean_score")
    score_deltas = [row["delta"] for row in deltas if row["metric"] == "mean_score"]

    assert score_deltas == [2.0, 3.0, 3.0, 4.0]
    assert score_effect["n_models"] == 4
    assert score_effect["df"] == 3
    assert score_effect["mean_delta"] == pytest.approx(3.0)
    assert score_effect["sd_delta"] == pytest.approx(math.sqrt(2 / 3))
    assert score_effect["unit"] == "training_seed"
    assert score_effect["ci95_low"] == pytest.approx(1.7005, abs=1e-3)


def test_episode_ci_is_not_model_effect_ci():
    models = []
    for seed in EXPECTED_TRAINING_SEEDS:
        score = 0 if seed == 0 else 100
        models.append(_model("d4", seed, 0, [score, score]))
        models.append(_model("no_d4", seed, 0, [0, 0]))

    effects, _ = paired_d4_no_d4_effects(models)
    effect = next(row for row in effects if row["metric"] == "mean_score")
    model_level_width = effect["ci95_high"] - effect["ci95_low"]
    episode_level = confidence_interval([0, 0, 100, 100, 100, 100, 100, 100])

    assert effect["n_models"] == 4
    assert episode_level["n"] == 8
    assert model_level_width > episode_level["ci95_high"] - episode_level["ci95_low"]


def test_cross_depth_effects_pair_by_condition_and_training_seed():
    models = []
    for condition in ("d4", "no_d4"):
        for seed in EXPECTED_TRAINING_SEEDS:
            models.append(_model(condition, seed, 0, [seed]))
            models.append(_model(condition, seed, 1, [seed + 5]))

    effects, deltas = cross_depth_effects(models)
    score_effect = next(
        row for row in effects
        if row["condition"] == "d4" and row["metric"] == "mean_score"
    )
    score_deltas = [
        row["delta"] for row in deltas
        if row["condition"] == "d4" and row["metric"] == "mean_score"
    ]

    assert score_deltas == [5.0] * 4
    assert score_effect["depth_a"] == 0
    assert score_effect["depth_b"] == 1
    assert score_effect["mean_delta"] == pytest.approx(5.0)
    assert score_effect["ci95_low"] == pytest.approx(5.0)
    assert score_effect["ci95_high"] == pytest.approx(5.0)
    assert score_effect["unit"] == "training_seed"


def test_seed_order_does_not_change_paired_effects():
    models = _complete_models()
    forward, _ = paired_d4_no_d4_effects(models)
    reverse, _ = paired_d4_no_d4_effects(list(reversed(models)))

    assert forward == reverse


def test_design_rejects_missing_seed_and_unpaired_condition():
    complete = _complete_models()
    with pytest.raises(ValueError, match="missing"):
        validate_complete_design(complete[:-1])

    no_d4_missing = [model for model in complete if model["condition"] == "d4"]
    with pytest.raises(ValueError, match="depth sets are not paired|pairing is missing"):
        validate_complete_design(no_d4_missing)


def test_design_rejects_duplicate_model_even_if_outcome_is_identical():
    models = _complete_models()
    models.append(models[0])

    with pytest.raises(ValueError, match="duplicate model/training seed"):
        validate_complete_design(models)


def test_manifest_statistics_write_machine_tables_and_exclude_twins(tmp_path):
    from scripts.aggregate import _aggregate_manifest_statistics

    runs = []
    for condition in ("d4", "no_d4"):
        for seed in EXPECTED_TRAINING_SEEDS:
            runs.append({
                "path": tmp_path / f"{condition}-{seed}",
                "config": {
                    "condition": condition,
                    "training_seed": seed,
                    "depth": 0,
                    "model_path": str(tmp_path / f"{condition}-{seed}.zip"),
                    "model_sha256": f"{seed:064x}",
                },
                "episodes": _episodes([seed + (1 if condition == "d4" else 0)]),
            })
    twin = {
        "path": tmp_path / "logged-copy",
        "twin_of": str(runs[0]["path"]),
        "config": {
            "condition": "d4",
            "training_seed": 0,
            "depth": 0,
            "outcome_fingerprint": "a" * 64,
        },
    }

    assert _aggregate_manifest_statistics(runs, [twin], tmp_path, "paper") == 0
    with (tmp_path / "per_model_metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 8
    with (tmp_path / "paired_d4_no_d4_effects.csv").open() as stream:
        effects = list(csv.DictReader(stream))
    assert len(effects) == len(STATISTICAL_METRICS)
    with (tmp_path / "model_level_confidence_intervals.csv").open() as stream:
        model_cis = list(csv.DictReader(stream))
    assert len(model_cis) == 2 * len(STATISTICAL_METRICS)
    assert {row["unit"] for row in model_cis} == {"training_seed"}
    assert (tmp_path / "cross_depth_paired_effects.csv").exists()
    with (tmp_path / "excluded_twins.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
