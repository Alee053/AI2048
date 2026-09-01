"""Benchmark CLI: orchestrates workers, writes CSV + JSON outputs.

Usage:
    python scripts/benchmark.py <model_path> [OPTIONS]

Examples:
    python scripts/benchmark.py data/archive/v3-100m/models/release/Hybrid-PPO-Expectimax-v3.zip \\
        --n-runs 100 --depth 3 --workers 4
    python scripts/benchmark.py data/archive/v3-100m/models/release/Hybrid-PPO-Expectimax-v3.zip \\
        --n-runs 50 --depth 3 --log-moves --yes-large-move-log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Allow `python scripts/benchmark.py` to resolve sibling `scripts.*` imports
# without requiring `python -m scripts.benchmark`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from scripts.benchmark_provenance import (
    SEARCH_BATCH_SIZE,
    SEARCH_CANONICALIZATION,
    SEARCH_TRANSPOSITION_TABLE,
    collect_runtime_provenance,
)
from scripts.benchmark_io import EPISODE_SCHEMA_VERSION
from scripts.paper_provenance import validate_benchmark_training_binding


_PAPER_REQUIRED_PROVENANCE = (
    "model_sha256", "effective_config_sha256", "uv_lock_sha256",
    "native_extension_sha256", "python_version", "torch_version",
    "sb3_version", "compiler",
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return parsed


def _tqdm_iter(iterable=None, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable if iterable is not None else iter([])
    disable = not sys.stdout.isatty()
    if iterable is None:
        return tqdm(disable=disable, **kwargs)
    return tqdm(iterable, disable=disable, **kwargs)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark a trained 2048 agent with optional expectimax search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model_path", type=str, nargs="?", default=None,
                   help="Path to trained model .zip file")
    p.add_argument("--n-runs", type=_positive_int, default=100,
                   help="Number of episodes to simulate (default: 100)")
    p.add_argument("--depth", type=int, default=0,
                   help="Expectimax search depth; 0 = raw policy (default: 0)")
    p.add_argument("--output", type=str, default=None,
                   help="Run name (folder name under data/benchmarks). Default: run_<timestamp>")
    p.add_argument(
        "--sweep-name", type=str, default=None,
        help="Manifest metadata used by aggregate.py to select this experiment.",
    )
    p.add_argument("--device", type=str, default="auto",
                   choices=("cpu", "cuda", "auto"),
                   help="Device for model inference (default: auto)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes (default: 1). "
                        "Recommended: --device cpu --workers 4-8 for throughput; "
                        "--device cuda --workers 1 for paper-grade latency.")
    p.add_argument(
        "--worker-timeout", type=_positive_float, default=300.0,
        help="Maximum worker inactivity time in seconds (default: 300)",
    )
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
    p.add_argument("--paper-mode", action="store_true",
                   help="Require a clean git worktree and emit paper-grade provenance.")
    p.add_argument("--allow-dirty-paper-run", action="store_true",
                   help="Allow --paper-mode on a dirty worktree, marking the run non-paper-grade.")
    p.add_argument("--effective-config", type=str, default=None,
                   help="Path to the resolved training configuration used for this model.")
    p.add_argument(
        "--training-manifest", type=str, default=None,
        help="Path to the training_manifest.json bound to this model.",
    )
    return p.parse_args(argv)


def _git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=_REPO_ROOT,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, cwd=_REPO_ROOT,
        )
        return bool(out.strip())
    except Exception:
        return None


def _cuda_info(device):
    if device != "cuda":
        return None, None
    try:
        name = torch.cuda.get_device_name(0)
        runtime = torch.version.cuda or ""
        return name, runtime
    except Exception:
        return None, None


def validate_paper_mode(args) -> None:
    """Reject a dirty paper run unless the explicit non-paper override is used."""
    if not args.paper_mode:
        return
    if args.train_seed is None:
        raise ValueError("--paper-mode requires --train-seed")
    if args.base_eval_seed is None:
        raise ValueError("--paper-mode requires --base-eval-seed")
    git_commit = _git_commit()
    git_dirty = _git_dirty()
    _validate_paper_git_state(git_commit, git_dirty, args.allow_dirty_paper_run)
    effective_config = resolve_effective_config_path(
        args.model_path, args.effective_config,
    )
    if effective_config is None:
        raise ValueError(
            "--paper-mode requires a model-adjacent effective config or an "
            "existing --effective-config artifact."
        )
    _validate_paper_training_seed(effective_config, args.train_seed)
    args.effective_config = str(effective_config)
    training_manifest = resolve_training_manifest_path(
        args.model_path, getattr(args, "training_manifest", None),
    )
    if training_manifest is None:
        raise ValueError(
            "--paper-mode requires a model-adjacent training_manifest.json or "
            "an existing --training-manifest artifact."
        )
    args.training_manifest = str(training_manifest)
    args._paper_provenance = _collect_paper_provenance(
        args.model_path, args.effective_config, args.training_manifest,
        expected_train_seed=args.train_seed,
        require_paper_grade=not args.allow_dirty_paper_run,
    )


def _validate_paper_git_state(git_commit, git_dirty, allow_dirty_paper_run: bool) -> None:
    if not _is_git_commit(git_commit) or git_dirty is None:
        raise ValueError("--paper-mode requires a verifiable git identity or status")
    if git_dirty and not allow_dirty_paper_run:
        raise ValueError(
            "--paper-mode requires a clean git worktree; use "
            "--allow-dirty-paper-run to record a non-paper-grade run."
        )


def _validate_paper_training_seed(effective_config: Path, train_seed: int) -> None:
    """Require the persisted resolved config to identify the benchmarked model."""
    try:
        with effective_config.open() as stream:
            resolved_config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "--paper-mode requires a readable resolved effective_config JSON"
        ) from exc
    if (
        not isinstance(resolved_config, dict)
        or resolved_config.get("root_training_seed") != train_seed
    ):
        raise ValueError(
            "--paper-mode requires effective_config root_training_seed to equal "
            "--train-seed"
        )


def _collect_paper_provenance(
    model_path: str,
    effective_config: str,
    training_manifest: str,
    *,
    expected_train_seed: int | None = None,
    require_paper_grade: bool = False,
) -> dict:
    provenance = collect_runtime_provenance(
        model_path=model_path, effective_config=effective_config,
    )
    missing = [key for key in _PAPER_REQUIRED_PROVENANCE if not provenance.get(key)]
    if missing:
        raise ValueError(
            "--paper-mode could not produce required provenance: "
            + ", ".join(missing)
        )
    binding = validate_benchmark_training_binding(
        model_path,
        training_manifest,
        expected_train_seed=expected_train_seed,
        expected_effective_config_path=effective_config,
        recorded=provenance,
        require_paper_grade=require_paper_grade,
    )
    return {**provenance, **binding}


def _is_git_commit(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def resolve_effective_config_path(
    model_path: str | None, explicit_path: str | None,
) -> Path | None:
    """Prefer an explicit resolved config, otherwise use the model's sibling."""
    candidate = Path(explicit_path) if explicit_path else (
        Path(model_path).parent / "effective_config.json" if model_path else None
    )
    return candidate.resolve() if candidate is not None and candidate.is_file() else None


def resolve_training_manifest_path(
    model_path: str | None, explicit_path: str | None,
) -> Path | None:
    """Prefer an explicit training manifest, otherwise use the model's sibling."""
    candidate = Path(explicit_path) if explicit_path else (
        Path(model_path).parent / "training_manifest.json" if model_path else None
    )
    return candidate.resolve() if candidate is not None and candidate.is_file() else None


def _condition_from_effective_config(path: Path | None) -> tuple[str | None, bool | None]:
    if path is None:
        return None, None
    try:
        with path.open() as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(config, dict):
        return None, None
    env_kwargs = config.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        return None, None
    d4_augment = env_kwargs.get("d4_augment")
    if type(d4_augment) is not bool:
        return None, None
    return ("d4" if d4_augment else "no_d4"), d4_augment


def build_config(args, run_name, env_seed_base, eval_seed_strategy, started_at_iso):
    cuda_name, cuda_runtime = _cuda_info(args.device)
    effective_config = resolve_effective_config_path(
        args.model_path, args.effective_config,
    )
    training_manifest = resolve_training_manifest_path(
        args.model_path, getattr(args, "training_manifest", None),
    )
    git_dirty = _git_dirty()
    git_commit = _git_commit()
    if args.paper_mode:
        _validate_paper_git_state(git_commit, git_dirty, args.allow_dirty_paper_run)
        if effective_config is None:
            raise ValueError(
                "--paper-mode requires a model-adjacent effective config or an "
                "existing --effective-config artifact."
            )
        if args.train_seed is None:
            raise ValueError("--paper-mode requires --train-seed")
        if args.base_eval_seed is None:
            raise ValueError("--paper-mode requires --base-eval-seed")
        if training_manifest is None:
            raise ValueError(
                "--paper-mode requires a model-adjacent training_manifest.json or "
                "an existing --training-manifest artifact."
            )
        provenance = getattr(args, "_paper_provenance", None)
        if provenance is None:
            provenance = _collect_paper_provenance(
                args.model_path,
                str(effective_config),
                str(training_manifest),
                expected_train_seed=args.train_seed,
                require_paper_grade=not args.allow_dirty_paper_run,
            )
    else:
        provenance = collect_runtime_provenance(
            model_path=args.model_path,
            effective_config=str(effective_config) if effective_config else None,
        )
        if training_manifest is not None and effective_config is not None:
            provenance = {
                **provenance,
                **_collect_paper_provenance(
                    args.model_path,
                    str(effective_config) if effective_config else "",
                    str(training_manifest),
                    expected_train_seed=args.train_seed,
                ),
            }
    condition, d4_augment = _condition_from_effective_config(effective_config)
    worker_timeout = float(getattr(args, "worker_timeout", 300.0))
    config = {
        "benchmark_schema_version": EPISODE_SCHEMA_VERSION,
        "run_name": run_name,
        "sweep_name": getattr(args, "sweep_name", None) or run_name,
        "model_path": str(Path(args.model_path).resolve()),
        "model_md5": "",
        "model_version": args.model_version or "",
        "train_seed": args.train_seed,
        "training_seed": args.train_seed,
        "condition": condition,
        "d4_augment": d4_augment,
        "env_seed_base": env_seed_base,
        "n_runs": args.n_runs,
        "n_workers": args.workers,
        "worker_timeout": worker_timeout,
        "worker_inactivity_timeout": worker_timeout,
        "worker_timeout_scope": "per-worker-inactivity-between-episode-heartbeats",
        "device": args.device,
        "cuda_device_name": cuda_name,
        "cuda_runtime": cuda_runtime,
        "depth": args.depth,
        "use_expectimax": args.depth > 0,
        "log_moves": bool(args.log_moves),
        "base_eval_seed": env_seed_base,
        "eval_seed_strategy": eval_seed_strategy,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "paper_mode": bool(args.paper_mode),
        "paper_grade": bool(
            args.paper_mode and _is_git_commit(git_commit) and git_dirty is False
            and not args.allow_dirty_paper_run and effective_config is not None
        ),
        "search_depth": args.depth,
        "search_canonicalization": SEARCH_CANONICALIZATION,
        "search_transposition_table": SEARCH_TRANSPOSITION_TABLE,
        "search_batch_size": SEARCH_BATCH_SIZE,
        "started_at_iso": started_at_iso,
        "finished_at_iso": "",
        "total_wall_time_s": 0.0,
        "interrupted": False,
        "status": "running",
    }
    config.update(provenance)
    return config


def main(argv=None):
    args = parse_args(argv)

    try:
        validate_paper_mode(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if not args.model_path:
        print("Error: model_path is required")
        return 1
    if not os.path.exists(args.model_path):
        print(f"Error: model file not found: {args.model_path}")
        return 1

    from scripts.benchmark_runner import run_benchmark

    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
