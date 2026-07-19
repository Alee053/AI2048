"""
Post-processing aggregator for AI2048 benchmark results.

Produces summary.csv and paper_figures/ with violin plots, bar charts, and heatmaps
across seeds and depths for a given sweep.

Usage:
    python scripts/aggregate.py data/benchmarks/ --sweep sweep-v1
    python scripts/aggregate.py data/benchmarks/ --sweep sweep-v1 --win-threshold 4096

Output: summary.csv + paper_figures/ in the benchmark root directory.
"""

import argparse
import os
import re
import sys
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

from scripts.benchmark_summary import compute_summary_from_rows
from scripts.benchmark_io import EPISODE_COLUMNS

SUPPORTED_SCHEMA_MAJOR = 1

_REQUIRED_PAPER_PROVENANCE = (
    "git_commit", "model_sha256", "effective_config_sha256", "uv_lock_sha256",
    "native_extension_sha256", "train_seed", "base_eval_seed", "python_version",
    "torch_version", "sb3_version", "cuda_runtime", "gpu_name", "compiler",
    "search_depth", "search_canonicalization", "search_transposition_table",
    "search_batch_size", "depth", "use_expectimax",
)
_PAIRED_PROVENANCE = (
    "git_commit", "model_sha256", "effective_config_sha256", "uv_lock_sha256",
    "native_extension_sha256", "device", "python_version", "torch_version",
    "sb3_version", "cuda_runtime", "gpu_name", "compiler",
    "search_canonicalization", "search_transposition_table", "search_batch_size",
)
_INT_EPISODE_FIELDS = {
    "episode_idx", "worker_id", "train_seed", "eval_seed", "requested_depth",
    "effective_depth", "score", "max_tile", "max_log_tile", "steps", "total_nodes",
    "total_batches", "total_tt_lookups", "total_tt_hits", "total_tt_collisions",
    "total_tt_same_key_overwrites", "total_moves_resolved", "total_moves_unresolved",
    "total_cap_hits", "total_alpha_beta_cuts", "total_chance_nodes", "total_max_nodes",
    "min_empty_cells",
}
_FLOAT_EPISODE_FIELDS = {
    "episode_time_s", "mean_move_time_ms", "median_move_time_ms", "p95_move_time_ms",
    "max_move_time_ms", "total_think_ms", "mean_chance_value", "mean_empty_cells",
    "mean_merge_score", "mean_nps", "mean_tt_hit_rate", "mean_nodes_per_batch_call",
}


def validate_paper_run(run_dir: str | Path) -> dict:
    """Validate one completed paper-grade run before it can be aggregated."""
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.json")
    summary = _load_json(run_dir / "summary.json")
    episodes = _load_episode_rows(run_dir / "episodes.csv")

    if _schema_major(config.get("benchmark_schema_version")) != SUPPORTED_SCHEMA_MAJOR:
        raise ValueError(f"{run_dir}: unsupported config schema version")
    if (
        _schema_major(summary.get("benchmark_schema_version")) != SUPPORTED_SCHEMA_MAJOR
        or summary.get("benchmark_schema_version") != config.get("benchmark_schema_version")
    ):
        raise ValueError(f"{run_dir}: unsupported or mismatched summary schema version")

    if config.get("status") != "completed" or config.get("interrupted"):
        raise ValueError(f"{run_dir}: status is not completed")
    if not config.get("paper_grade") or config.get("git_dirty"):
        raise ValueError(f"{run_dir}: run is not paper-grade")
    missing = [key for key in _REQUIRED_PAPER_PROVENANCE if key not in config or config[key] is None]
    nonempty = (
        "git_commit", "model_sha256", "effective_config_sha256", "uv_lock_sha256",
        "native_extension_sha256", "compiler", "search_canonicalization",
    )
    missing.extend(key for key in nonempty if not config.get(key) and key not in missing)
    if missing:
        raise ValueError(f"{run_dir}: missing paper provenance: {', '.join(missing)}")

    requested = int(config["n_runs"])
    if len(episodes) != requested or config.get("n_completed") != requested:
        raise ValueError(f"{run_dir}: episode row count does not match requested runs")
    if summary.get("n_completed") != requested or summary.get("n_runs_requested") != requested:
        raise ValueError(f"{run_dir}: summary row count does not match requested runs")
    if any(_schema_major(row["schema_version"]) != SUPPORTED_SCHEMA_MAJOR for row in episodes):
        raise ValueError(f"{run_dir}: unsupported episode schema version")

    run_id = config.get("run_id")
    if run_id in (None, "") or any(row.get("run_id") != run_id for row in episodes):
        raise ValueError(f"{run_dir}: episode run_id does not match config")
    indices = [row["episode_idx"] for row in episodes]
    if len(indices) != len(set(indices)) or set(indices) != set(range(requested)):
        raise ValueError(f"{run_dir}: duplicate or missing episode_idx values")
    seeds = [row["eval_seed"] for row in episodes]
    expected_seeds = {int(config["base_eval_seed"]) + index for index in range(requested)}
    if len(seeds) != len(set(seeds)) or set(seeds) != expected_seeds:
        raise ValueError(f"{run_dir}: duplicate or missing eval_seed values")
    if any(row.get("train_seed") != config["train_seed"] for row in episodes):
        raise ValueError(f"{run_dir}: episode training seed does not match config")
    expected_depth = int(config["depth"])
    expected_search = bool(config["use_expectimax"])
    if any(row["requested_depth"] != expected_depth for row in episodes):
        raise ValueError(f"{run_dir}: episode requested_depth does not match config")
    if any(row["effective_depth"] != expected_depth for row in episodes):
        raise ValueError(f"{run_dir}: episode effective_depth does not match config")
    if any(row["use_expectimax"] != expected_search for row in episodes):
        raise ValueError(f"{run_dir}: episode use_expectimax does not match config")
    if any(row["total_cap_hits"] for row in episodes):
        raise ValueError(f"{run_dir}: episodes contain search cap hits")
    if any(row["total_moves_unresolved"] for row in episodes):
        raise ValueError(f"{run_dir}: episodes contain unresolved search moves")

    total_time_s = float(summary.get("metrics", {}).get("total_time_s", 0.0))
    expected_summary = compute_summary_from_rows(episodes, config, total_time_s)
    expected_summary.update({
        "status": "completed", "interrupted": False,
        "n_completed": requested, "n_runs_requested": requested,
    })
    for key in expected_summary:
        if key not in summary or not _equivalent(expected_summary[key], summary[key]):
            raise ValueError(f"{run_dir}: summary.json does not match episodes.csv ({key})")
    return {"config": config, "summary": summary, "episodes": episodes}


def validate_paired_paper_runs(run_dirs) -> list[dict]:
    """Require paired conditions to share eval seeds and execution provenance."""
    runs = [validate_paper_run(run_dir) for run_dir in run_dirs]
    if len(runs) < 2:
        return runs
    baseline = runs[0]
    baseline_seeds = {row["eval_seed"] for row in baseline["episodes"]}
    for run in runs[1:]:
        if {row["eval_seed"] for row in run["episodes"]} != baseline_seeds:
            raise ValueError("paired runs have mismatched eval seed sets")
        for key in _PAIRED_PROVENANCE:
            if run["config"].get(key) != baseline["config"].get(key):
                raise ValueError(f"paired runs differ in {key}")
    return runs


def _load_json(path: Path) -> dict:
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable JSON") from exc


def _load_episode_rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = [column for column in EPISODE_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"{path}: missing required episode CSV columns: {', '.join(missing)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"{path}: unreadable episodes CSV") from exc
    for row in rows:
        for key in _INT_EPISODE_FIELDS:
            if key in row and row[key] not in (None, ""):
                row[key] = int(row[key])
        for key in _FLOAT_EPISODE_FIELDS:
            if key in row and row[key] not in (None, ""):
                row[key] = float(row[key])
        row["use_expectimax"] = _parse_bool(row["use_expectimax"], path)
    return rows


def _parse_bool(value, path: Path) -> bool:
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"{path}: invalid boolean use_expectimax value")


def _schema_major(version) -> int | None:
    try:
        return int(str(version).split(".", maxsplit=1)[0])
    except (TypeError, ValueError):
        return None


def _equivalent(expected, actual) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _equivalent(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _equivalent(left, right) for left, right in zip(expected, actual)
        )
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return expected == actual


def summary_fieldnames(rows: list[dict]) -> list[str]:
    """Return the ordered union of all summary row fields."""
    return list(dict.fromkeys(key for row in rows for key in row))


def _check_schema_versions(benchmark_dir, sweep_name):
    """Walk {sweep_name}_depth* folders and check config.json schema versions.

    Returns a list of (path, version) tuples for any run whose major
    version differs from SUPPORTED_SCHEMA_MAJOR. Config lives at the
    depth-folder root (the producer writes {sweep}_depth{N}/config.json).
    """
    pattern = re.compile(rf"{re.escape(sweep_name)}_depth(\d+)")
    violations = []
    for entry in os.scandir(benchmark_dir):
        if not entry.is_dir():
            continue
        if not pattern.match(entry.name):
            continue
        cfg = Path(entry.path) / "config.json"
        if not cfg.exists():
            # Try subdir fallback for legacy multi-seed sweeps.
            for sub in os.scandir(entry.path):
                if not sub.is_dir():
                    continue
                legacy_cfg = Path(sub.path) / "config.json"
                if legacy_cfg.exists():
                    cfg = legacy_cfg
                    break
            else:
                continue
        try:
            with open(cfg) as f:
                cfg_data = json.load(f)
            ver = cfg_data.get("benchmark_schema_version", "")
            major = int(ver.split(".")[0]) if ver else 0
            if major != SUPPORTED_SCHEMA_MAJOR:
                violations.append((str(cfg.parent), ver))
        except Exception:
            violations.append((str(cfg.parent), "unreadable"))
    return violations


def discover_depth_folders(benchmark_dir: str, sweep_name: str, legacy: bool = False) -> dict:
    """Find all {sweep_name}_depth{N} folders and their result files.

    Supports both legacy layouts (results_seed_N.json per seed) and the
    new flat layout where each depth folder contains episodes.csv directly.
    Falls back to treating benchmark_dir as a single depth=0 run when no
    sweep subfolders are present (single-run smoke case).

    When legacy=True, only results_seed_N.json files are accepted and the
    episodes.csv fallback is skipped.
    """
    pattern = re.compile(rf"{re.escape(sweep_name)}_depth(\d+)")
    depth_folders = {}

    for entry in os.scandir(benchmark_dir):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        depth = int(m.group(1))
        legacy_files = sorted([
            f.path for f in os.scandir(entry.path)
            if re.match(r"results_seed_\d+\.json", f.name)
        ])
        if legacy_files:
            depth_folders[depth] = {
                "path": entry.path,
                "results": legacy_files,
            }
            continue
        if legacy:
            continue
        ep_csv = Path(entry.path) / "episodes.csv"
        if ep_csv.exists():
            depth_folders[depth] = {
                "path": entry.path,
                "results": [str(ep_csv)],
            }

    # Single-run fallback: benchmark_dir itself contains episodes.csv.
    if not depth_folders and not legacy:
        direct_csv = Path(benchmark_dir) / "episodes.csv"
        if direct_csv.exists():
            depth_folders[0] = {
                "path": benchmark_dir,
                "results": [str(direct_csv)],
            }
    return depth_folders


def load_result_file(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_win_rates(raw_data: list, thresholds: list) -> dict:
    """Compute win rate (fraction >= threshold) for each threshold."""
    n = len(raw_data)
    result = {}
    for t in thresholds:
        result[f"win_rate_{t}"] = (sum(1 for r in raw_data if r["max_tile"] >= t) / n) if n else 0.0
    return result


def compute_max_tile_eq_pct(raw_data: list, thresholds: list) -> dict:
    """Compute fraction of episodes where max_tile == threshold."""
    n = len(raw_data)
    result = {}
    for t in thresholds:
        result[f"max_tile_eq_{t}_pct"] = (sum(1 for r in raw_data if r["max_tile"] == t) / n) if n else 0.0
    return result


def aggregate_depth(depth_results: list) -> dict:
    """Aggregate metrics across seeds for a single depth."""
    avg_scores = [r["metrics"]["avg_score"] for r in depth_results]
    std_scores = [r["metrics"]["std_score"] for r in depth_results]
    min_scores = [r["metrics"]["min_score"] for r in depth_results]
    max_scores = [r["metrics"]["max_score"] for r in depth_results]
    avg_steps = [r["metrics"]["avg_steps"] for r in depth_results]

    n_seeds = len(depth_results)
    n_episodes = depth_results[0]["config"]["n_runs"]

    result = {
        "mean_score": np.mean(avg_scores),
        "std_score": np.mean(std_scores),  # avg of per-seed stds
        "min_score": np.mean(min_scores),
        "max_score": np.mean(max_scores),
        "mean_steps": np.mean(avg_steps),
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
    }

    # Aggregate search metrics if present
    search_keys = ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]
    for key in search_keys:
        vals = [r["metrics"][key] for r in depth_results if key in r["metrics"]]
        if vals:
            result[key] = np.mean(vals)

    return result


def compute_ci(mean: float, std: float, n: int, z: float = 1.96) -> tuple:
    """Compute 95% CI given mean, std, and n samples."""
    se = std / np.sqrt(n)
    return mean - z * se, mean + z * se


def _pstdev(values):
    """Population standard deviation. Used for per-seed std in summary rows."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def load_episodes_csv(run_dir):
    """Load episodes.csv from a run folder as a list of dicts.

    Returns an empty list if the file doesn't exist or pandas is missing.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: aggregate.py requires pandas for the new CSV layout.")
        print("Install with: uv add pandas")
        sys.exit(1)
    csv_path = Path(run_dir) / "episodes.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=str, help="Root folder containing {sweep_name}_depth* subfolders")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep name to aggregate")
    parser.add_argument("--win-threshold", type=int, default=None, help="Report single win threshold (default: all observed)")
    parser.add_argument("--output", type=str, default=None, help="Override output directory (default: benchmark_dir)")
    parser.add_argument("--legacy", action="store_true",
                        help="Read legacy results_seed_N.json files instead of the new CSV layout.")
    parser.add_argument("--paper-mode", "--strict", dest="paper_mode", action="store_true",
                        help="Reject incomplete, non-paper-grade, or unpaired result artifacts.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    benchmark_dir = args.benchmark_dir
    sweep_name = args.sweep
    output_dir = Path(args.output or benchmark_dir)

    # Schema-version check: aggregate.py only accepts major version 1.
    if not args.legacy:
        schema_violations = _check_schema_versions(benchmark_dir, sweep_name)
        if schema_violations:
            print("Error: unsupported benchmark_schema_version in some run folders:")
            for path, ver in schema_violations:
                print(f"  {path}: {ver}")
            print("aggregate.py only accepts major version 1.x.x.")
            print("Re-run with --legacy to consume the old JSON layout.")
            sys.exit(2)

    # Discover
    depth_folders = discover_depth_folders(benchmark_dir, sweep_name, legacy=args.legacy)
    if not depth_folders:
        print(f"Error: no depth folders found for sweep '{sweep_name}' in {benchmark_dir}")
        print(f"  Expected pattern: {sweep_name}_depth<N>")
        return 1

    if args.paper_mode:
        try:
            validate_paired_paper_runs(
                [info["path"] for _, info in sorted(depth_folders.items())]
            )
        except ValueError as exc:
            print(f"Error: paper aggregation rejected: {exc}")
            return 2

    print(f"Found depths: {sorted(depth_folders.keys())}")
    for depth, info in sorted(depth_folders.items()):
        print(f"  depth {depth}: {len(info['results'])} seed result(s)")

    THRESHOLDS = [1024, 2048, 4096, 8192]
    REPORT_THRESHOLDS = THRESHOLDS if args.win_threshold is None else [args.win_threshold]

    # Collect all rows for summary.csv
    summary_rows = []
    cross_depth_rows = []

    for depth, info in sorted(depth_folders.items()):
        depth_results = []
        for result_path in info["results"]:
            basename = os.path.basename(result_path)
            seed_m = re.search(r"results_seed_(\d+)\.json", basename)
            if seed_m is not None:
                seed_n = int(seed_m.group(1))
            elif basename == "episodes.csv":
                # New flat layout: treat each depth folder as a single seed=0 run.
                seed_n = 0
            else:
                continue
            if basename.endswith(".json"):
                result = load_result_file(result_path)
                raw = result.get("raw_data", [])
            else:
                result = {}
                raw = []

            win_rates = compute_win_rates(raw, REPORT_THRESHOLDS)
            tile_eq_pcts = compute_max_tile_eq_pct(raw, THRESHOLDS)

            row = {
                "sweep_name": sweep_name,
                "depth": depth,
                "seed": seed_n,
            }
            run_dir = os.path.dirname(result_path)
            episodes = load_episodes_csv(run_dir) if not args.legacy else []

            if episodes:
                # New CSV layout
                scores = [int(r["score"]) for r in episodes]
                steps_ = [int(r["steps"]) for r in episodes]
                max_tiles = [int(r["max_tile"]) for r in episodes]
                row.update({
                    "avg_score": float(sum(scores) / len(scores)),
                    "std_score": float(_pstdev(scores)),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "avg_steps": float(sum(steps_) / len(steps_)),
                })
                # Include search-mode metrics from episodes.csv when present.
                for key, col in [
                    ("avg_think_ms", "total_think_ms"),
                    ("avg_nodes_visited", "total_nodes"),
                    ("avg_batches_eval", "total_batches"),
                    ("avg_tt_collisions", "total_tt_collisions"),
                    ("avg_tt_same_key_overwrites", "total_tt_same_key_overwrites"),
                    ("avg_moves_resolved", "total_moves_resolved"),
                    ("avg_moves_unresolved", "total_moves_unresolved"),
                    ("avg_cap_hits", "total_cap_hits"),
                    ("avg_alpha_beta_cuts", "total_alpha_beta_cuts"),
                    ("avg_chance_nodes", "total_chance_nodes"),
                    ("avg_max_nodes", "total_max_nodes"),
                    ("avg_nodes_per_sec", "mean_nps"),
                    ("avg_tt_hit_rate", "mean_tt_hit_rate"),
                    ("avg_chance_value", "mean_chance_value"),
                ]:
                    vals = [r[col] for r in episodes if r.get(col) is not None and not (isinstance(r[col], float) and (r[col] != r[col]))]
                    if vals:
                        row[key] = float(sum(vals) / len(vals))
                # Reuse the existing win-rate/tile-eq computation by passing
                # synthetic raw_data-like list.
                win_rates = compute_win_rates(
                    [{"score": s, "max_tile": mt} for s, mt in zip(scores, max_tiles)],
                    REPORT_THRESHOLDS,
                )
                tile_eq_pcts = compute_max_tile_eq_pct(
                    [{"max_tile": mt} for mt in max_tiles],
                    THRESHOLDS,
                )
            else:
                # Legacy JSON layout
                row.update({
                    "avg_score": result["metrics"]["avg_score"],
                    "std_score": result["metrics"]["std_score"],
                    "min_score": result["metrics"]["min_score"],
                    "max_score": result["metrics"]["max_score"],
                    "avg_steps": result["metrics"]["avg_steps"],
                })
                for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
                    if key in result["metrics"]:
                        row[key] = result["metrics"][key]

            row.update(win_rates)
            row.update(tile_eq_pcts)
            summary_rows.append(row)
            depth_results.append(result)

        # Cross-seed aggregate row for this depth (legacy only; new flat
        # layout contributes a single per-seed row that already encodes
        # the metrics).
        legacy_depth_results = [r for r in depth_results if r.get("metrics")]
        if legacy_depth_results:
            agg = aggregate_depth(legacy_depth_results)
            ci_lower, ci_upper = compute_ci(agg["mean_score"], agg["std_score"], agg["n_seeds"])
            agg_row = {
                "sweep_name": sweep_name,
                "depth": depth,
                "seed": "aggregate",
                "avg_score": agg["mean_score"],
                "std_score": agg["std_score"],
                "min_score": agg["min_score"],
                "max_score": agg["max_score"],
                "avg_steps": agg["mean_steps"],
            }
            # Aggregate search metrics across seeds
            for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
                vals = [r["metrics"][key] for r in legacy_depth_results if key in r["metrics"]]
                if vals:
                    agg_row[key] = round(np.mean(vals), 3 if key == "avg_think_ms" else 2 if key == "avg_tt_hit_rate" else 1)
            # Compute aggregate win rates from pooled raw data
            all_raw = [r for res in legacy_depth_results for r in res.get("raw_data", [])]
            agg_win_rates = compute_win_rates(all_raw, REPORT_THRESHOLDS)
            agg_tile_eq = compute_max_tile_eq_pct(all_raw, THRESHOLDS)
            agg_row.update(agg_win_rates)
            agg_row.update(agg_tile_eq)
            summary_rows.append(agg_row)

            # Cross-depth CI table
            ci_row = {
                "depth": depth,
                "mean_score": agg["mean_score"],
                "std_score": agg["std_score"],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_seeds": agg["n_seeds"],
                "n_episodes": agg["n_seeds"] * agg["n_episodes"],
            }
            # Include search metrics in CI table
            for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
                if key in agg:
                    ci_row[key] = agg[key]
            cross_depth_rows.append(ci_row)

    # Write summary.csv
    csv_path = output_dir / "summary.csv"
    if summary_rows:
        fieldnames = summary_fieldnames(summary_rows)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary written to {csv_path}")

    # Write cross-depth CI table
    ci_csv_path = output_dir / "cross_depth_ci_table.csv"
    if cross_depth_rows:
        # Dynamically build fieldnames from all keys present in rows
        ci_fields = ["depth", "mean_score", "std_score", "ci_lower", "ci_upper", "n_seeds", "n_episodes"]
        for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
            if any(key in row for row in cross_depth_rows):
                ci_fields.append(key)
        with open(ci_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ci_fields)
            writer.writeheader()
            writer.writerows(cross_depth_rows)
        print(f"Cross-depth CI table written to {ci_csv_path}")

    # Generate paper figures
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        print("Warning: matplotlib not found. Skipping paper figures.")

    if MATPLOTLIB_AVAILABLE:
        figures_dir = output_dir / "paper_figures"
        figures_dir.mkdir(exist_ok=True)

        def _load_scores_for_path(result_path):
            """Return a list of {seed, scores, raw} entries for one path."""
            basename = os.path.basename(result_path)
            if basename.endswith(".json"):
                result = load_result_file(result_path)
                seed_m = re.search(r"results_seed_(\d+)\.json", basename)
                seed = seed_m.group(1) if seed_m else "0"
                raw = result.get("raw_data", [])
                scores = [r["score"] for r in raw]
                return [{"seed": seed, "scores": scores, "raw": raw}]
            if basename == "episodes.csv":
                run_dir = os.path.dirname(result_path)
                episodes = load_episodes_csv(run_dir)
                raw = [{"score": int(r["score"]), "max_tile": int(r["max_tile"])} for r in episodes]
                scores = [int(r["score"]) for r in episodes]
                return [{"seed": "0", "scores": scores, "raw": raw}]
            return []

        for depth, info in sorted(depth_folders.items()):
            seed_data = []
            all_raw = []
            for result_path in info["results"]:
                entries = _load_scores_for_path(result_path)
                seed_data.extend(entries)
                for e in entries:
                    all_raw.extend(e["raw"])

            if not seed_data:
                continue

            # 1. Violin plot of score distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            parts = ax.violinplot([d["scores"] for d in seed_data],
                                  positions=list(range(len(seed_data))),
                                  showmeans=True, showmedians=True)
            ax.set_xticks(range(len(seed_data)))
            ax.set_xticklabels([d["seed"] for d in seed_data])
            ax.set_xlabel("Seed")
            ax.set_ylabel("Score")
            ax.set_title(f"Score Distribution — Depth {depth}")
            plt.savefig(figures_dir / f"violin_score_depth{depth}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 2. Bar chart of win rates (per seed + aggregate)
            win_rates_per_seed = []
            for d in seed_data:
                raw = d["raw"]
                wr = sum(1 for r in raw if r["max_tile"] >= 2048) / len(raw) if raw else 0
                win_rates_per_seed.append(wr)

            agg_wr = sum(1 for r in all_raw if r["max_tile"] >= 2048) / len(all_raw) if all_raw else 0

            fig, ax = plt.subplots(figsize=(10, 6))
            x = list(range(len(win_rates_per_seed)))
            bars = ax.bar(x, win_rates_per_seed, color="steelblue", alpha=0.7)
            ax.axhline(agg_wr, color="orange", linestyle="--", linewidth=2, label=f"Aggregate: {agg_wr:.2%}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Seed {d['seed']}" for d in seed_data])
            ax.set_ylabel("Win Rate (>=2048)")
            ax.set_title(f"Win Rate per Seed — Depth {depth}")
            ax.legend()
            plt.savefig(figures_dir / f"bar_winrate_depth{depth}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 3. Heatmap of max tile frequency across all seeds and depths
        all_depths = sorted(depth_folders.keys())
        tile_values = [1024, 2048, 4096, 8192, 16384]
        heatmap_data = np.zeros((len(all_depths), len(tile_values)))

        for di, depth in enumerate(all_depths):
            for result_path in depth_folders[depth]["results"]:
                entries = _load_scores_for_path(result_path)
                for e in entries:
                    raw = e["raw"]
                    n = len(raw)
                    for ti, tile in enumerate(tile_values):
                        count = sum(1 for r in raw if r["max_tile"] == tile)
                        heatmap_data[di, ti] += count / n if n else 0

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(tile_values)))
        ax.set_xticklabels(tile_values)
        ax.set_yticks(range(len(all_depths)))
        ax.set_yticklabels([f"Depth {d}" for d in all_depths])
        ax.set_xlabel("Max Tile")
        ax.set_ylabel("Configuration")
        plt.colorbar(im, ax=ax, label="Fraction of episodes")
        ax.set_title(f"Max Tile Frequency Heatmap — {sweep_name}")
        plt.savefig(figures_dir / "heatmap_max_tile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Paper figures written to {figures_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
