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
from pathlib import Path
from collections import defaultdict

import numpy as np

SUPPORTED_SCHEMA_MAJOR = 1


def _check_schema_versions(benchmark_dir, sweep_name):
    """Walk {sweep_name}_depth* folders and check config.json schema versions.

    Returns a list of (path, version) tuples for any run whose major
    version differs from SUPPORTED_SCHEMA_MAJOR.
    """
    pattern = re.compile(rf"{re.escape(sweep_name)}_depth(\d+)")
    violations = []
    for entry in os.scandir(benchmark_dir):
        if not entry.is_dir():
            continue
        if not pattern.match(entry.name):
            continue
        for sub in os.scandir(entry.path):
            if not sub.is_dir():
                continue
            cfg = Path(sub.path) / "config.json"
            if not cfg.exists():
                continue
            try:
                with open(cfg) as f:
                    cfg_data = json.load(f)
                ver = cfg_data.get("benchmark_schema_version", "")
                major = int(ver.split(".")[0]) if ver else 0
                if major != SUPPORTED_SCHEMA_MAJOR:
                    violations.append((sub.path, ver))
            except Exception:
                violations.append((sub.path, "unreadable"))
    return violations


def discover_depth_folders(benchmark_dir: str, sweep_name: str) -> dict:
    """Find all {sweep_name}_depth{N} folders and their result files."""
    pattern = re.compile(rf"{re.escape(sweep_name)}_depth(\d+)")
    depth_folders = {}

    for entry in os.scandir(benchmark_dir):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        depth = int(m.group(1))
        result_files = sorted([
            f.path for f in os.scandir(entry.path)
            if re.match(r"results_seed_\d+\.json", f.name)
        ])
        if result_files:
            depth_folders[depth] = {
                "path": entry.path,
                "results": result_files
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
        result[f"win_rate_{t}"] = sum(1 for r in raw_data if r["max_tile"] >= t) / n
    return result


def compute_max_tile_eq_pct(raw_data: list, thresholds: list) -> dict:
    """Compute fraction of episodes where max_tile == threshold."""
    n = len(raw_data)
    result = {}
    for t in thresholds:
        result[f"max_tile_eq_{t}_pct"] = sum(1 for r in raw_data if r["max_tile"] == t) / n
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=str, help="Root folder containing {sweep_name}_depth* subfolders")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep name to aggregate")
    parser.add_argument("--win-threshold", type=int, default=None, help="Report single win threshold (default: all observed)")
    parser.add_argument("--output", type=str, default=None, help="Override output directory (default: benchmark_dir)")
    parser.add_argument("--legacy", action="store_true",
                        help="Read legacy results_seed_N.json files instead of the new CSV layout.")
    args = parser.parse_args()

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
    depth_folders = discover_depth_folders(benchmark_dir, sweep_name)
    if not depth_folders:
        print(f"Error: no depth folders found for sweep '{sweep_name}' in {benchmark_dir}")
        print(f"  Expected pattern: {sweep_name}_depth<N>")
        sys.exit(1)

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
            seed_m = re.search(r"results_seed_(\d+)\.json", os.path.basename(result_path))
            if seed_m is None:
                continue
            seed_n = int(seed_m.group(1))
            result = load_result_file(result_path)
            raw = result.get("raw_data", [])

            win_rates = compute_win_rates(raw, REPORT_THRESHOLDS)
            tile_eq_pcts = compute_max_tile_eq_pct(raw, THRESHOLDS)

            row = {
                "sweep_name": sweep_name,
                "depth": depth,
                "seed": seed_n,
                "avg_score": result["metrics"]["avg_score"],
                "std_score": result["metrics"]["std_score"],
                "min_score": result["metrics"]["min_score"],
                "max_score": result["metrics"]["max_score"],
                "avg_steps": result["metrics"]["avg_steps"],
            }
            # Include search metrics if present
            for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
                if key in result["metrics"]:
                    row[key] = result["metrics"][key]
            row.update(win_rates)
            row.update(tile_eq_pcts)
            summary_rows.append(row)
            depth_results.append(result)

        # Cross-seed aggregate row for this depth
        agg = aggregate_depth(depth_results)
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
            vals = [r["metrics"][key] for r in depth_results if key in r["metrics"]]
            if vals:
                agg_row[key] = round(np.mean(vals), 3 if key == "avg_think_ms" else 2 if key == "avg_tt_hit_rate" else 1)
        # Compute aggregate win rates from pooled raw data
        all_raw = [r for res in depth_results for r in res.get("raw_data", [])]
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
        fieldnames = list(summary_rows[0].keys())
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

        for depth, info in sorted(depth_folders.items()):
            # Load all seed results for this depth
            seed_data = []
            for result_path in info["results"]:
                result = load_result_file(result_path)
                seed_m = re.search(r"results_seed_(\d+)\.json", os.path.basename(result_path))
                if seed_m is None:
                    continue
                seed_n = seed_m.group(1)
                scores = [r["score"] for r in result.get("raw_data", [])]
                seed_data.append({"seed": seed_n, "scores": scores})

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
            depth_results = []
            all_raw = []
            for result_path in info["results"]:
                result = load_result_file(result_path)
                depth_results.append(result)
                all_raw.extend(result.get("raw_data", []))

            win_rates_per_seed = []
            for result in depth_results:
                raw = result.get("raw_data", [])
                wr = sum(1 for r in raw if r["max_tile"] >= 2048) / len(raw) if raw else 0
                win_rates_per_seed.append(wr)

            agg_wr = sum(1 for r in all_raw if r["max_tile"] >= 2048) / len(all_raw) if all_raw else 0

            fig, ax = plt.subplots(figsize=(10, 6))
            x = list(range(len(win_rates_per_seed)))
            bars = ax.bar(x, win_rates_per_seed, color="steelblue", alpha=0.7)
            ax.axhline(agg_wr, color="orange", linestyle="--", linewidth=2, label=f"Aggregate: {agg_wr:.2%}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Seed {i}" for i in range(len(win_rates_per_seed))])
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
                result = load_result_file(result_path)
                raw = result.get("raw_data", [])
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


if __name__ == "__main__":
    main()