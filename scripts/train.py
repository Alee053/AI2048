"""
Training script for 2048 AI agent.

Usage:
    python scripts/train.py --config <path_to_yaml>

Required Config Keys:
    - project_name: W&B project identifier
    - run_name: Unique experiment name
    - total_timesteps: Training steps (e.g., 200_000_000)
    - n_envs: Parallel environment count
    - ppo_params: Dict of PPO hyperparameters

Examples:
    # Fresh training
    python scripts/train.py --config configs/train/hybrid_ppo_v1.yaml
    
    # Resume from checkpoint
    python scripts/train.py --config configs/train/resume_training.yaml
"""

import os
import time
import json as _json
import subprocess
import yaml
import argparse
import wandb
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from numpy.random import SeedSequence
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.agent.architecture import CustomCNN
from twenty_forty_eight_ai.agent.callbacks import WandbLoggingCallback
from twenty_forty_eight_ai.agent.policy import ValueNormalizedMaskablePolicy
try:
    from scripts.benchmark_provenance import collect_runtime_provenance, sha256_file
except ModuleNotFoundError:  # Support `python scripts/train.py`.
    from benchmark_provenance import collect_runtime_provenance, sha256_file
from twenty_forty_eight_ai.utils.effective_config import (
    D4_SEED_DERIVATION,
    V3_EXPERIMENT_CONDITIONS,
    V3_TRAINING_SEEDS,
    derive_d4_rank_seed_sequences,
    materialize_training_config,
    validate_v3_experiment_config,
)

SweepStatusPath = "sweep_status.json"
EFFECTIVE_CONFIG_FILENAME = "effective_config.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_sweep_status(output_dir: str) -> dict:
    path = os.path.join(output_dir, SweepStatusPath)
    if os.path.exists(path):
        with open(path, "r") as f:
            return _json.load(f)
    return {}


def _save_sweep_status(output_dir: str, status: dict) -> None:
    path = os.path.join(output_dir, SweepStatusPath)
    with open(path, "w") as f:
        _json.dump(status, f, indent=2)


def _init_sweep_status(sweep_name: str, total_seeds: int) -> dict:
    return {
        "sweep_name": sweep_name,
        "total_seeds": total_seeds,
        "seeds": {str(i): {"status": "pending"} for i in range(total_seeds)}
    }


def _update_sweep_status(output_dir: str, seed: int, status: str, error: str = None) -> None:
    sweep = _load_sweep_status(output_dir)
    if str(seed) in sweep.get("seeds", {}):
        sweep["seeds"][str(seed)]["status"] = status
        if error:
            sweep["seeds"][str(seed)]["error"] = error
    else:
        sweep["seeds"][str(seed)] = {"status": status, "error": error} if error else {"status": status}
    _save_sweep_status(output_dir, sweep)


def set_global_seed(seed: int) -> None:
    """Set all random seeds for deterministic, reproducible runs."""
    import random
    import numpy as np
    import torch

    # Must be set before any CUDA ops; avoids "deterministic" RuntimeError
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Some ops don't have deterministic fallback


def collect_git_provenance() -> dict[str, str | bool]:
    """Collect the commit and real porcelain worktree state for a run."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_status_porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Unable to collect git provenance") from exc
    if not git_commit:
        raise RuntimeError("Unable to collect git commit SHA")
    return {
        "git_commit": git_commit,
        "git_status_porcelain": git_status_porcelain,
        "git_dirty": bool(git_status_porcelain),
    }


def native_extension_identity() -> dict[str, str]:
    """Identify the loaded C++ extension and verify its on-disk digest."""
    try:
        from twenty_forty_eight_ai.utils import searcher

        extension_path = Path(searcher._impl.__file__).resolve()
    except (AttributeError, ImportError, OSError) as exc:
        raise RuntimeError("Unable to identify the loaded C++ extension") from exc
    extension_sha256 = sha256_file(extension_path)
    if not extension_sha256:
        raise RuntimeError("Unable to hash the loaded C++ extension")
    return {"path": str(extension_path), "sha256": extension_sha256}


def relevant_package_versions() -> dict[str, str]:
    """Return package versions that affect the v3 training runtime."""
    packages = ("gymnasium", "numpy", "numba", "sb3-contrib", "pybind11")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = ""
    return versions


def seed_from_config(config: dict) -> int:
    """Get seed from config dict, defaulting to 0."""
    return config.get("seed", 0)


def resume_settings(config: dict) -> tuple[bool, str | None]:
    """Return validated resume settings, defaulting to fresh training."""
    load_model = bool(config.get("load_model", False))
    checkpoint_path = config.get("checkpoint_path")
    if load_model and not checkpoint_path:
        raise ValueError("load_model=true requires a non-empty checkpoint_path")
    return load_model, checkpoint_path


def make_training_env_factories(
    env_kwargs: dict, d4_rank_seed_sequences: list[SeedSequence]
) -> list:
    """Build monitored rank factories with reproducible, distinct D4 streams."""
    return [
        lambda d4_seed_sequence=d4_seed_sequence: Monitor(
            Game2048Env(**env_kwargs, d4_seed=d4_seed_sequence)
        )
        for d4_seed_sequence in d4_rank_seed_sequences
    ]


def resolve_training_config(config: dict) -> tuple[dict, list[SeedSequence]]:
    """Materialize training defaults and reproducible D4 provenance for W&B."""
    seed = seed_from_config(config)
    effective_config = materialize_training_config(config)
    d4_rank_seed_sequences = derive_d4_rank_seed_sequences(
        seed, effective_config['n_envs']
    )
    effective_config.update(
        root_training_seed=seed,
        d4_seed_derivation=D4_SEED_DERIVATION,
        d4_rank_spawn_keys=[
            list(sequence.spawn_key) for sequence in d4_rank_seed_sequences
        ],
    )
    return effective_config, d4_rank_seed_sequences


def select_training_policy(config: dict):
    """Use the normalized critic only for the v3 experiments."""
    if config.get("run_name") in V3_EXPERIMENT_CONDITIONS:
        return ValueNormalizedMaskablePolicy
    return "CnnPolicy"


def validate_v3_seed_sweep(config: dict, requested_seed_count: int) -> None:
    """Require the configured four-seed sweep for either v3 condition."""
    if config.get("run_name") not in V3_EXPERIMENT_CONDITIONS:
        return
    validate_v3_experiment_config(config)
    if requested_seed_count != len(V3_TRAINING_SEEDS):
        raise ValueError(
            "v3 training requires requested seed count to match training_seeds "
            f"({len(V3_TRAINING_SEEDS)} seeds)"
        )


def persist_effective_config(model_dir: str | Path, effective_config: dict) -> Path:
    """Write the resolved training configuration beside checkpoints and final model."""
    path = Path(model_dir) / EFFECTIVE_CONFIG_FILENAME
    with path.open("w") as stream:
        _json.dump(effective_config, stream, indent=2, sort_keys=True)
    return path


def persist_training_manifest(model_dir: str | Path, model_path: str, model, effective_config: dict) -> Path:
    """Persist immutable provenance bound to the final saved model."""
    effective_config_path = Path(model_dir) / EFFECTIVE_CONFIG_FILENAME
    if not effective_config_path.is_file():
        raise RuntimeError("Effective config must exist before writing training manifest")
    env_kwargs = effective_config.get("env_kwargs")
    d4_augment = env_kwargs.get("d4_augment") if isinstance(env_kwargs, dict) else None
    if type(d4_augment) is not bool:
        raise RuntimeError("Effective config must record a boolean D4 condition")

    git_provenance = collect_git_provenance()
    runtime_provenance = collect_runtime_provenance(
        model_path=model_path,
        effective_config=str(effective_config_path),
    )
    extension = native_extension_identity()
    package_versions = relevant_package_versions()
    required_runtime_fields = (
        "effective_config_sha256",
        "python_version",
        "torch_version",
        "sb3_version",
    )
    missing_runtime_fields = [
        field for field in required_runtime_fields if not runtime_provenance.get(field)
    ]
    missing_package_versions = [
        package for package, package_version in package_versions.items()
        if not package_version
    ]
    if missing_runtime_fields or missing_package_versions:
        missing = missing_runtime_fields + missing_package_versions
        raise RuntimeError("Missing training provenance: " + ", ".join(missing))

    manifest = {
        **git_provenance,
        "effective_config": effective_config,
        "effective_config_path": str(effective_config_path),
        "training_seed": effective_config["root_training_seed"],
        "d4_augment": d4_augment,
        "d4_condition": "d4" if d4_augment else "no_d4",
        "versions": {
            "python": runtime_provenance["python_version"],
            "torch": runtime_provenance["torch_version"],
            "stable-baselines3": runtime_provenance["sb3_version"],
            **package_versions,
        },
        "runtime": {
            "cuda_runtime": runtime_provenance.get("cuda_runtime", ""),
            "gpu_name": runtime_provenance.get("gpu_name", ""),
            "compiler": runtime_provenance.get("compiler", ""),
        },
        "native_extension": extension,
        "root_training_seed": effective_config["root_training_seed"],
        "effective_config_sha256": runtime_provenance["effective_config_sha256"],
        "uv_lock_sha256": runtime_provenance.get("uv_lock_sha256", ""),
        "native_extension_sha256": extension["sha256"],
        "python_version": runtime_provenance["python_version"],
        "torch_version": runtime_provenance["torch_version"],
        "sb3_version": runtime_provenance["sb3_version"],
        "model_sha256": sha256_file(model_path),
        "final_timestep": model.num_timesteps,
    }
    path = Path(model_dir) / "training_manifest.json"
    with path.open("w") as stream:
        _json.dump(manifest, stream, indent=2, sort_keys=True)
    return path


def train(config: dict):
    """Run training loop."""
    seed = seed_from_config(config)
    effective_config, d4_rank_seed_sequences = resolve_training_config(config)
    run_name = f"{config['run_name']}-seed{seed}"
    run = wandb.init(
        project=config['project_name'],
        config=effective_config,
        name=run_name,
        save_code=True,
    )
    set_global_seed(seed)

    model_dir = os.path.join(config['output_dir'], "models", config['run_name'])
    os.makedirs(model_dir, exist_ok=True)
    persist_effective_config(model_dir, effective_config)

    # Callbacks
    wandb_callback = WandbLoggingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config['save_interval'] // config['n_envs'], 1),
        save_path=model_dir,
        name_prefix="rl_model"
    )
    callbacks = [wandb_callback, checkpoint_callback]

    # Environment & Model
    env_kwargs = effective_config['env_kwargs']
    vec_env = DummyVecEnv(
        make_training_env_factories(env_kwargs, d4_rank_seed_sequences)
    )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config['features_dim']),
    )

    should_load_model, checkpoint_path = resume_settings(config)

    if should_load_model:
        print(f"Loading model from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint path not found at {checkpoint_path}. Exiting.")
            return

        model = MaskablePPO.load(checkpoint_path, env=vec_env, verbose=1)

        # Calculate remaining steps
        current_steps = model.num_timesteps
        total_steps = config['total_timesteps']
        remaining_steps = total_steps - current_steps

        # Resumed LR schedule
        lr_config = config['ppo_params']['learning_rate']
        if lr_config['type'] == 'linear_decay':
            def resumed_lr_schedule(progress_remaining: float) -> float:
                progress_this_run = 1.0 - progress_remaining
                steps_this_run = progress_this_run * remaining_steps
                current_global_steps = current_steps + steps_this_run
                global_progress_remaining = 1.0 - (current_global_steps / total_steps)
                return global_progress_remaining * lr_config['initial_value']

            model.learning_rate = resumed_lr_schedule

        print(f"Resuming training for an additional {remaining_steps} steps.")
        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=False
        )

    else:
        print("Starting a new training run from scratch.")
        ppo_params = config['ppo_params'].copy()

        # New LR schedule
        lr_config = ppo_params.pop('learning_rate')
        if lr_config['type'] == 'linear_decay':
            ppo_params['learning_rate'] = lambda p: p * lr_config['initial_value']

        model = MaskablePPO(
            select_training_policy(config), vec_env, policy_kwargs=policy_kwargs,
            verbose=1, seed=seed, **ppo_params
        )
        model.learn(
            total_timesteps=config['total_timesteps'],
            reset_num_timesteps=True,
            callback=callbacks,
            progress_bar=False
        )

    # Save model
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    persist_training_manifest(model_dir, final_model_path, model, effective_config)
    print(f"Final model saved to: {final_model_path}")
    print("Training complete!")


def main_with_sweep(config: dict):
    sweep_cfg = config.get("__sweep", {})
    if not sweep_cfg.get("enabled"):
        train(config)
        return

    n_seeds = sweep_cfg["n_seeds"]
    validate_v3_seed_sweep(config, n_seeds)
    sweep_name = config.get("run_name", f"sweep_{int(time.time())}")
    output_dir = os.path.join(config["output_dir"], "models", sweep_name)

    if sweep_cfg["parallel"]:
        print("Warning: parallel training not recommended on single-GPU systems.")
        print("Each seed run should be launched independently with --seed.")

    if sweep_cfg["dry_run"]:
        _print_dry_run(config, sweep_name, n_seeds)
        return

    os.makedirs(output_dir, exist_ok=True)

    if sweep_cfg["resume"]:
        sweep_status = _load_sweep_status(output_dir)
        if not sweep_status:
            print(f"No sweep status found at {output_dir}, initializing new sweep.")
            sweep_status = _init_sweep_status(sweep_name, n_seeds)
            _save_sweep_status(output_dir, sweep_status)
    else:
        sweep_status = _init_sweep_status(sweep_name, n_seeds)
        _save_sweep_status(output_dir, sweep_status)

    for seed_i in range(n_seeds):
        status = sweep_status.get("seeds", {}).get(str(seed_i), {}).get("status")
        if status == "completed":
            print(f"Seed {seed_i}: already completed, skipping.")
            continue

        print(f"\n=== Starting seed {seed_i}/{n_seeds-1} ===")
        _update_sweep_status(output_dir, seed_i, "running")

        seed_config = config.copy()
        seed_config["seed"] = seed_i
        seed_config["run_name"] = f"{sweep_name}-seed{seed_i}"

        try:
            train(seed_config)
            _update_sweep_status(output_dir, seed_i, "completed")
            print(f"Seed {seed_i}: completed.")
        except Exception as e:
            _update_sweep_status(output_dir, seed_i, "failed", str(e))
            print(f"Seed {seed_i}: FAILED — {e}")
            if sweep_cfg["resume"]:
                print("Continuing to next seed (resume mode)...")
            else:
                raise
        finally:
            wandb.finish()


def _print_dry_run(config: dict, sweep_name: str, n_seeds: int):
    print(f"Sweep: {sweep_name} ({n_seeds} seeds, sequential)")
    print(f"W&B group: {sweep_name}")
    print(f"Config: {config.get('_config_path', 'unknown')}")
    print()

    for i in range(n_seeds):
        seed_output = os.path.join(config["output_dir"], "models", sweep_name, f"seed_{i}")
        cmd = (f"python scripts/train.py --config <config> --seed {i} "
               f"--output-dir {seed_output} --run-name {sweep_name}-seed{i}")
        print(f"Seed {i} │ run: {sweep_name}-seed{i} │ output: {seed_output} │ cmd: {cmd}")

    print("\nDry run complete. No jobs launched.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Set global random seed.")
    parser.add_argument("--seed-sweep", type=int, default=None, metavar="N",
                        help="Launch N sequential training runs with seeds [0..N-1].")
    parser.add_argument("--resume-sweep", action="store_true",
                        help="Resume a sweep: skip completed seeds, re-run failed/pending.")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel execution (not recommended for single-GPU training).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sweep plan without launching any jobs.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    # Apply CLI seed override
    if args.seed is not None:
        config_data["seed"] = args.seed

    # Apply sweep CLI overrides
    config_data["__sweep"] = {
        "enabled": args.seed_sweep is not None,
        "n_seeds": args.seed_sweep,
        "resume": args.resume_sweep,
        "parallel": args.parallel,
        "dry_run": args.dry_run,
    }

    # Track config path
    config_data["_config_path"] = args.config

    try:
        main_with_sweep(config_data)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting.")
    finally:
        if wandb.run is not None:
            wandb.finish()
