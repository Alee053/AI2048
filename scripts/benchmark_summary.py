"""Summary aggregation from collected episode rows."""
from __future__ import annotations

import math
import statistics
from typing import Any


def compute_summary_from_rows(rows, config, total_time_s):
    """Compute summary.json from collected episode rows."""
    if not rows:
        return {
            "benchmark_schema_version": "1.0.0",
            "run_name": config.get("run_name", ""),
            "status": config.get("status", "unknown"),
            "interrupted": config.get("interrupted", False),
            "n_completed": 0,
            "n_runs_requested": config.get("n_runs", 0),
            "config": _config_subset(config),
            "metrics": {},
            "win_rates": {},
            "max_tile_dist": {},
        }

    scores = [r["score"] for r in rows]
    max_tiles = [r["max_tile"] for r in rows]
    steps = [r["steps"] for r in rows]

    metrics: dict = {
        "avg_score": float(statistics.fmean(scores)),
        "std_score": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        "min_score": min(scores),
        "max_score": max(scores),
        "median_score": float(statistics.median(scores)),
        "p25_score": float(_percentile(scores, 25)),
        "p75_score": float(_percentile(scores, 75)),
        "avg_steps": float(statistics.fmean(steps)),
        "min_steps": min(steps),
        "max_steps": max(steps),
        "median_steps": float(statistics.median(steps)),
        "total_time_s": float(total_time_s),
        "total_wall_time_s": float(config.get("total_wall_time_s", total_time_s)),
        "avg_time_per_game_s": float(total_time_s / len(rows)),
        "games_per_sec": float(len(rows) / total_time_s) if total_time_s > 0 else 0.0,
    }
    if len(scores) > 1:
        score_std = statistics.pstdev(scores)
        score_se = score_std / (len(scores) ** 0.5)
        metrics["score_ci95_low"] = float(metrics["avg_score"] - 1.96 * score_se)
        metrics["score_ci95_high"] = float(metrics["avg_score"] + 1.96 * score_se)

    use_search = config.get("use_expectimax", False)
    if use_search and any(r.get("total_think_ms") for r in rows):
        metrics["avg_think_ms"] = float(statistics.fmean(
            r["total_think_ms"] for r in rows
        ))
        metrics["avg_nodes_visited"] = float(statistics.fmean(
            r["total_nodes"] for r in rows
        ))
        metrics["avg_batches_eval"] = float(statistics.fmean(
            r["total_batches"] for r in rows
        ))
        metrics["avg_tt_collisions"] = float(statistics.fmean(
            r["total_tt_collisions"] for r in rows
        ))
        metrics["avg_tt_same_key_overwrites"] = float(statistics.fmean(
            r["total_tt_same_key_overwrites"] for r in rows
        ))
        metrics["avg_moves_resolved"] = float(statistics.fmean(
            r["total_moves_resolved"] for r in rows
        ))
        metrics["avg_moves_unresolved"] = float(statistics.fmean(
            r["total_moves_unresolved"] for r in rows
        ))
        metrics["avg_cap_hits"] = float(statistics.fmean(
            r["total_cap_hits"] for r in rows
        ))
        metrics["avg_alpha_beta_cuts"] = float(statistics.fmean(
            r["total_alpha_beta_cuts"] for r in rows
        ))
        metrics["avg_chance_nodes"] = float(statistics.fmean(
            r["total_chance_nodes"] for r in rows
        ))
        metrics["avg_max_nodes"] = float(statistics.fmean(
            r["total_max_nodes"] for r in rows
        ))
        nps_values = [r["mean_nps"] for r in rows if r["mean_nps"] > 0]
        if nps_values:
            metrics["avg_nodes_per_sec"] = float(statistics.fmean(nps_values))
        tt_rates = [r["mean_tt_hit_rate"] for r in rows if r["mean_tt_hit_rate"] > 0]
        if tt_rates:
            metrics["avg_tt_hit_rate"] = float(statistics.fmean(tt_rates))
        chance_values = [r["mean_chance_value"] for r in rows if r.get("mean_chance_value")]
        if chance_values:
            metrics["avg_chance_value"] = float(statistics.fmean(chance_values))

    win_rates = {}
    for t in (1024, 2048, 4096, 8192):
        wins = sum(1 for v in max_tiles if v >= t)
        win_rates[f"win_rate_{t}+"] = wins / len(rows)
    max_tile_dist = {}
    for t in sorted(set(max_tiles)):
        max_tile_dist[str(t)] = max_tiles.count(t)

    return {
        "benchmark_schema_version": "1.0.0",
        "run_name": config.get("run_name", ""),
        "config": _config_subset(config),
        "metrics": metrics,
        "win_rates": win_rates,
        "max_tile_dist": max_tile_dist,
    }


def _config_subset(config):
    return {
        "model_path": config.get("model_path"),
        "model_md5": config.get("model_md5"),
        "depth": config.get("depth"),
        "use_expectimax": config.get("use_expectimax"),
        "n_workers": config.get("n_workers"),
        "device": config.get("device"),
        "log_moves": config.get("log_moves"),
        "base_eval_seed": config.get("base_eval_seed"),
    }


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))