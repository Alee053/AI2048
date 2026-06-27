"""Benchmark CLI: orchestrates workers, writes CSV + JSON outputs.

Usage:
    python scripts/benchmark.py <model_path> [OPTIONS]

Examples:
    python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v3.zip \\
        --n-runs 100 --depth 3 --workers 4
    python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v3.zip \\
        --n-runs 50 --depth 3 --log-moves --yes-large-move-log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import torch


def _tqdm_iter(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    disable = not sys.stdout.isatty()
    return tqdm(iterable, disable=disable, **kwargs)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark a trained 2048 agent with optional expectimax search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model_path", type=str, nargs="?", default=None,
                   help="Path to trained model .zip file (not required when --model-dir is set)")
    p.add_argument("--n-runs", type=int, default=100,
                   help="Number of episodes to simulate (default: 100)")
    p.add_argument("--depth", type=int, default=0,
                   help="Expectimax search depth; 0 = raw policy (default: 0)")
    p.add_argument("--output", type=str, default=None,
                   help="Run name (folder name under data/benchmarks). Default: run_<timestamp>")
    p.add_argument("--device", type=str, default="auto",
                   choices=("cpu", "cuda", "auto"),
                   help="Device for model inference (default: auto)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes (default: 1). "
                        "Recommended: --device cpu --workers 4-8 for throughput; "
                        "--device cuda --workers 1 for paper-grade latency.")
    p.add_argument("--log-moves", action="store_true",
                   help="Write per-move CSV (moves.csv). Off by default.")
    p.add_argument("--yes-large-move-log", action="store_true",
                   help="Acknowledge >5M row estimate when combined with --log-moves.")
    p.add_argument("--base-eval-seed", type=int, default=None,
                   help="Root seed for deterministic per-episode eval seeds. "
                        "Default: random, surfaced in config.json.")
    p.add_argument("--train-seed", type=int, default=None,
                   help="Train seed (recorded in config.json for sweep runs).")
    p.add_argument("--model-version", type=str, default=None,
                   help="Free-form version label, recorded in config.json.")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Directory containing seed_N/ subdirs for multi-seed benchmarking.")
    p.add_argument("--parallel", action="store_true",
                   help="Run seed benchmarks in parallel (multi-seed mode only).")
    return p.parse_args(argv)


def _git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=os.getcwd(),
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, cwd=os.getcwd(),
        )
        return bool(out.strip())
    except Exception:
        return False


def _cuda_info(device):
    if device != "cuda":
        return None, None
    try:
        name = torch.cuda.get_device_name(0)
        runtime = torch.version.cuda or ""
        return name, runtime
    except Exception:
        return None, None


def build_config(args, run_name, env_seed_base, eval_seed_strategy, started_at_iso):
    cuda_name, cuda_runtime = _cuda_info(args.device)
    return {
        "benchmark_schema_version": "1.0.0",
        "run_name": run_name,
        "model_path": str(args.model_path),
        "model_md5": "",
        "model_version": args.model_version or "",
        "train_seed": args.train_seed,
        "env_seed_base": env_seed_base,
        "n_runs": args.n_runs,
        "n_workers": args.workers,
        "device": args.device,
        "cuda_device_name": cuda_name,
        "cuda_runtime": cuda_runtime,
        "depth": args.depth,
        "use_expectimax": args.depth > 0,
        "log_moves": bool(args.log_moves),
        "base_eval_seed": env_seed_base if args.base_eval_seed is not None else None,
        "eval_seed_strategy": eval_seed_strategy,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "started_at_iso": started_at_iso,
        "finished_at_iso": "",
        "total_wall_time_s": 0.0,
        "interrupted": False,
        "status": "running",
    }


def main(argv=None):
    args = parse_args(argv)

    if args.model_dir:
        from scripts.benchmark_multi_seed import benchmark_multi_seed
        return benchmark_multi_seed(
            args.model_dir, args.n_runs, args.depth, args.device,
            args.output or f"run_{int(time.time())}",
            args.parallel,
        )

    if not args.model_path:
        print("Error: model_path is required (or use --model-dir for multi-seed)")
        return 1
    if not os.path.exists(args.model_path):
        print(f"Error: model file not found: {args.model_path}")
        return 1

    from scripts.benchmark_runner import run_benchmark

    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())