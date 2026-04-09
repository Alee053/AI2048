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
import yaml
import argparse
import wandb
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.agent.architecture import CustomCNN
from twenty_forty_eight_ai.agent.callbacks import WandbLoggingCallback

SweepStatusPath = "sweep_status.json"


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


def seed_from_config(config: dict) -> int:
    """Get seed from config dict, defaulting to 0."""
    return config.get("seed", 0)


def train(config: dict):
    """Run training loop."""
    seed = seed_from_config(config)
    run_name = f"{config['run_name']}-seed{seed}"
    run = wandb.init(
        project=config['project_name'],
        config=config,
        name=run_name,
        save_code=True,
    )
    set_global_seed(seed)

    model_dir = os.path.join(config['output_dir'], "models", config['run_name'])
    os.makedirs(model_dir, exist_ok=True)

    # Callbacks
    wandb_callback = WandbLoggingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config['save_interval'] // config['n_envs'], 1),
        save_path=model_dir,
        name_prefix="rl_model"
    )
    callbacks = [wandb_callback, checkpoint_callback]

    # Environment & Model
    vec_env = make_vec_env(Game2048Env, n_envs=config['n_envs'])
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config['features_dim']),
    )

    should_load_model = config['load_model'] and config['checkpoint_path'] is not None

    if should_load_model:
        print(f"Loading model from: {config['checkpoint_path']}")
        if not os.path.exists(config['checkpoint_path']):
            print(f"Error: Checkpoint path not found at {config['checkpoint_path']}. Exiting.")
            return

        model = MaskablePPO.load(config['checkpoint_path'], env=vec_env, verbose=1)

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
            "CnnPolicy", vec_env, policy_kwargs=policy_kwargs,
            verbose=1, **ppo_params
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
    print(f"Final model saved to: {final_model_path}")
    print("Training complete!")


def main_with_sweep(config: dict):
    sweep_cfg = config.get("__sweep", {})
    if not sweep_cfg.get("enabled"):
        train(config)
        return

    n_seeds = sweep_cfg["n_seeds"]
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