"""Seed-level statistics for manifest-bound benchmark runs.

Episodes describe the stochastic evaluation of one trained model. They are
therefore used to form one model-level estimate, never as independent
replicates of a training treatment.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


EXPECTED_TRAINING_SEEDS: tuple[int, ...] = (0, 1, 2, 3)
STATISTICAL_METRICS: tuple[str, ...] = (
    "mean_score",
    "win_rate_1024",
    "win_rate_2048",
    "win_rate_4096",
    "win_rate_8192",
    "mean_steps",
    "mean_max_tile",
    "mean_total_think_ms",
    "mean_total_nodes",
    "mean_total_batches",
    "mean_tt_hit_rate",
    "mean_nps",
)


def sample_sd(values: Sequence[float]) -> float:
    """Return sample SD, which is the estimator used for model deltas."""
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def student_t_critical_95(df: int) -> float:
    """Return the two-sided 95% Student-t critical value for ``df``."""
    if not isinstance(df, int) or isinstance(df, bool) or df < 1:
        raise ValueError("Student-t degrees of freedom must be a positive integer")

    target = 0.975
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, df) < target:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, df) < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def confidence_interval(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize a sample with a two-sided 95% Student-t CI."""
    if not values:
        raise ValueError("cannot calculate a confidence interval for no values")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("confidence-interval values must be finite")
    n = len(numeric)
    mean = math.fsum(numeric) / n
    sd = sample_sd(numeric)
    if n < 2:
        return {
            "n": n,
            "df": 0,
            "mean": mean,
            "sd": sd,
            "ci95_low": None,
            "ci95_high": None,
        }
    margin = student_t_critical_95(n - 1) * sd / math.sqrt(n)
    return {
        "n": n,
        "df": n - 1,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def compute_model_metrics(config: Mapping, episodes: Sequence[Mapping]) -> dict:
    """Compute one model/training-seed estimate from its episode rows."""
    if not episodes:
        raise ValueError("a model must contain at least one episode")
    scores = [float(row["score"]) for row in episodes]
    steps = [float(row["steps"]) for row in episodes]
    max_tiles = [float(row["max_tile"]) for row in episodes]
    total_think_ms = [float(row["total_think_ms"]) for row in episodes]
    total_nodes = [float(row["total_nodes"]) for row in episodes]
    total_batches = [float(row["total_batches"]) for row in episodes]
    tt_hit_rates = [float(row["mean_tt_hit_rate"]) for row in episodes]
    nps = [float(row["mean_nps"]) for row in episodes]
    win_2048 = [float(row["max_tile"] >= 2048) for row in episodes]
    win_4096 = [float(row["max_tile"] >= 4096) for row in episodes]
    win_1024 = [float(row["max_tile"] >= 1024) for row in episodes]
    win_8192 = [float(row["max_tile"] >= 8192) for row in episodes]
    score_ci = confidence_interval(scores)
    return {
        "analysis_unit": "training_seed",
        "sweep_name": config.get("sweep_name", ""),
        "condition": config["condition"],
        "training_seed": config["training_seed"],
        "depth": config["depth"],
        "model_path": config.get("model_path", ""),
        "model_sha256": config.get("model_sha256", ""),
        "n_episodes": len(episodes),
        "mean_score": score_ci["mean"],
        "win_rate_1024": math.fsum(win_1024) / len(win_1024),
        "mean_steps": math.fsum(steps) / len(steps),
        "mean_max_tile": math.fsum(max_tiles) / len(max_tiles),
        "win_rate_2048": math.fsum(win_2048) / len(win_2048),
        "win_rate_4096": math.fsum(win_4096) / len(win_4096),
        "win_rate_8192": math.fsum(win_8192) / len(win_8192),
        "mean_total_think_ms": math.fsum(total_think_ms) / len(total_think_ms),
        "mean_total_nodes": math.fsum(total_nodes) / len(total_nodes),
        "mean_total_batches": math.fsum(total_batches) / len(total_batches),
        "mean_tt_hit_rate": math.fsum(tt_hit_rates) / len(tt_hit_rates),
        "mean_nps": math.fsum(nps) / len(nps),
        "episode_score_sd_descriptive": score_ci["sd"],
        "episode_uncertainty_unit": "episode_descriptive_only",
        "episode_score_ci95_low_descriptive": score_ci["ci95_low"],
        "episode_score_ci95_high_descriptive": score_ci["ci95_high"],
        "episode_win_rate_2048_ci95_low_descriptive": confidence_interval(win_2048)[
            "ci95_low"
        ],
        "episode_win_rate_2048_ci95_high_descriptive": confidence_interval(win_2048)[
            "ci95_high"
        ],
        "episode_win_rate_4096_ci95_low_descriptive": confidence_interval(win_4096)[
            "ci95_low"
        ],
        "episode_win_rate_4096_ci95_high_descriptive": confidence_interval(win_4096)[
            "ci95_high"
        ],
        "episode_win_rate_1024_ci95_low_descriptive": confidence_interval(win_1024)[
            "ci95_low"
        ],
        "episode_win_rate_1024_ci95_high_descriptive": confidence_interval(win_1024)[
            "ci95_high"
        ],
        "episode_win_rate_8192_ci95_low_descriptive": confidence_interval(win_8192)[
            "ci95_low"
        ],
        "episode_win_rate_8192_ci95_high_descriptive": confidence_interval(win_8192)[
            "ci95_high"
        ],
    }


def validate_complete_design(
    model_metrics: Sequence[Mapping],
    *,
    expected_seeds: Iterable[int] = EXPECTED_TRAINING_SEEDS,
) -> None:
    """Reject incomplete seed sets or unpaired D4/no-D4 conditions."""
    expected = set(expected_seeds)
    if not expected:
        raise ValueError("expected training seed set cannot be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in expected):
        raise ValueError("expected training seeds must be integers")
    if not model_metrics:
        raise ValueError("statistical analysis requires model metrics")

    grouped: dict[tuple[str, int], dict[int, Mapping]] = defaultdict(dict)
    for model in model_metrics:
        condition = model.get("condition")
        depth = model.get("depth")
        seed = model.get("training_seed")
        if not isinstance(condition, str) or condition not in {"d4", "no_d4"}:
            raise ValueError(f"unsupported statistical condition: {condition!r}")
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise ValueError(f"invalid statistical depth: {depth!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"invalid training seed: {seed!r}")
        key = (condition, depth)
        if seed in grouped[key]:
            raise ValueError(f"duplicate model/training seed: {key + (seed,)}")
        grouped[key][seed] = model

    depths_by_condition = {
        condition: {depth for current_condition, depth in grouped if current_condition == condition}
        for condition in ("d4", "no_d4")
    }
    if depths_by_condition["d4"] != depths_by_condition["no_d4"]:
        raise ValueError("D4/no-D4 depth sets are not paired")
    if not depths_by_condition["d4"]:
        raise ValueError("D4/no-D4 pairing is missing")

    for key, models in grouped.items():
        actual = set(models)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"incomplete training seeds for {key}: missing={missing}, extra={extra}"
            )


def paired_d4_no_d4_effects(
    model_metrics: Sequence[Mapping],
    *,
    expected_seeds: Iterable[int] = EXPECTED_TRAINING_SEEDS,
) -> tuple[list[dict], list[dict]]:
    """Return per-seed deltas and Student-t effects for D4 minus No-D4."""
    expected = tuple(sorted(set(expected_seeds)))
    validate_complete_design(model_metrics, expected_seeds=expected)
    grouped = _index_models(model_metrics)
    effect_rows: list[dict] = []
    delta_rows: list[dict] = []
    depths = sorted({model["depth"] for model in model_metrics})
    for depth in depths:
        for metric in STATISTICAL_METRICS:
            deltas = []
            for seed in expected:
                d4 = grouped[("d4", depth, seed)][metric]
                no_d4 = grouped[("no_d4", depth, seed)][metric]
                delta = float(d4) - float(no_d4)
                deltas.append(delta)
                delta_rows.append({
                    "analysis": "d4_vs_no_d4",
                    "comparison": "d4_minus_no_d4",
                    "condition": "d4_minus_no_d4",
                    "depth": depth,
                    "depth_a": "",
                    "depth_b": "",
                    "training_seed": seed,
                    "metric": metric,
                    "delta": delta,
                    "unit": "training_seed",
                })
            effect_rows.append(_effect_row(
                analysis="d4_vs_no_d4",
                comparison="d4_minus_no_d4",
                condition="d4_minus_no_d4",
                depth=depth,
                depth_a="",
                depth_b="",
                metric=metric,
                values=deltas,
            ))
    return effect_rows, delta_rows


def model_level_confidence_intervals(
    model_metrics: Sequence[Mapping],
    *,
    expected_seeds: Iterable[int] = EXPECTED_TRAINING_SEEDS,
) -> list[dict]:
    """Calculate uncertainty across model estimates within each condition/depth."""
    expected = tuple(sorted(set(expected_seeds)))
    validate_complete_design(model_metrics, expected_seeds=expected)
    grouped = _index_models(model_metrics)
    rows = []
    for condition in ("d4", "no_d4"):
        for depth in sorted({model["depth"] for model in model_metrics}):
            for metric in STATISTICAL_METRICS:
                interval = confidence_interval([
                    float(grouped[(condition, depth, seed)][metric])
                    for seed in expected
                ])
                rows.append({
                    "analysis": "model_level",
                    "condition": condition,
                    "depth": depth,
                    "metric": metric,
                    "n_models": interval["n"],
                    "df": interval["df"],
                    "mean": interval["mean"],
                    "sd": interval["sd"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                    "unit": "training_seed",
                    "ci_method": "student_t_95",
                })
    return rows


def cross_depth_effects(
    model_metrics: Sequence[Mapping],
    *,
    expected_seeds: Iterable[int] = EXPECTED_TRAINING_SEEDS,
) -> tuple[list[dict], list[dict]]:
    """Return paired effects for every ordered depth pair within each condition."""
    expected = tuple(sorted(set(expected_seeds)))
    validate_complete_design(model_metrics, expected_seeds=expected)
    grouped = _index_models(model_metrics)
    effect_rows: list[dict] = []
    delta_rows: list[dict] = []
    depths = sorted({model["depth"] for model in model_metrics})
    for condition in ("d4", "no_d4"):
        for depth_a, depth_b in itertools.combinations(depths, 2):
            for metric in STATISTICAL_METRICS:
                deltas = []
                for seed in expected:
                    delta = (
                        float(grouped[(condition, depth_b, seed)][metric])
                        - float(grouped[(condition, depth_a, seed)][metric])
                    )
                    deltas.append(delta)
                    delta_rows.append({
                        "analysis": "cross_depth",
                        "comparison": "depth_b_minus_depth_a",
                        "condition": condition,
                        "depth": "",
                        "depth_a": depth_a,
                        "depth_b": depth_b,
                        "training_seed": seed,
                        "metric": metric,
                        "delta": delta,
                        "unit": "training_seed",
                    })
                effect_rows.append(_effect_row(
                    analysis="cross_depth",
                    comparison="depth_b_minus_depth_a",
                    condition=condition,
                    depth="",
                    depth_a=depth_a,
                    depth_b=depth_b,
                    metric=metric,
                    values=deltas,
                ))
    return effect_rows, delta_rows


def _index_models(model_metrics: Sequence[Mapping]) -> dict[tuple[str, int, int], Mapping]:
    return {
        (model["condition"], model["depth"], model["training_seed"]): model
        for model in model_metrics
    }


def _effect_row(
    *,
    analysis: str,
    comparison: str,
    condition: str,
    depth,
    depth_a,
    depth_b,
    metric: str,
    values: Sequence[float],
) -> dict:
    interval = confidence_interval(values)
    return {
        "analysis": analysis,
        "comparison": comparison,
        "condition": condition,
        "depth": depth,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "metric": metric,
        "n_models": interval["n"],
        "df": interval["df"],
        "mean_delta": interval["mean"],
        "sd_delta": interval["sd"],
        "ci95_low": interval["ci95_low"],
        "ci95_high": interval["ci95_high"],
        "unit": "training_seed",
    }


def _student_t_cdf(value: float, df: int) -> float:
    if value == 0:
        return 0.5
    x = df / (df + value * value)
    tail = 0.5 * _regularized_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if value > 0 else tail


def _regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = max(abs(d), tiny) if d == 0 else d
    d = 1.0 / d
    result = d
    for iteration in range(1, 201):
        doubled = 2 * iteration
        numerator = iteration * (b - iteration) * x / ((a + doubled - 1.0) * (a + doubled))
        d = 1.0 + numerator * d
        d = max(abs(d), tiny) if d == 0 else d
        c = 1.0 + numerator / c
        c = max(abs(c), tiny) if c == 0 else c
        d = 1.0 / d
        result *= d * c
        numerator = (
            -(a + iteration) * (a + b + iteration) * x
            / ((a + doubled) * (a + doubled + 1.0))
        )
        d = 1.0 + numerator * d
        d = max(abs(d), tiny) if d == 0 else d
        c = 1.0 + numerator / c
        c = max(abs(c), tiny) if c == 0 else c
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < 3e-14:
            break
    return result
