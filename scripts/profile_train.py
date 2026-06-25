#!/usr/bin/env python
"""
Training pipeline profiler — identifies the slowest component.

Usage:
    uv run python scripts/profile_train.py

Runs a short training burst then prints top time-consuming functions.
"""

import cProfile
import pstats
import io
import time
import numpy as np
import yaml
from stable_baselines3.common.env_util import make_vec_env

from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.agent.architecture import CustomCNN
from twenty_forty_eight_ai.agent.callbacks import WandbLoggingCallback
from sb3_contrib import MaskablePPO

# --- Config (minimal for quick profile) ------------------------------------
CONFIG = {
    "total_timesteps": 50_000,
    "n_envs": 4,
    "save_interval": 100_000,
    "features_dim": 256,
    "ppo_params": {
        "n_steps": 512,
        "gamma": 0.95,
        "ent_coef": 6.68e-6,
        "learning_rate": {"type": "linear_decay", "initial_value": 2.5e-4},
        "clip_range": 0.2,
        "batch_size": 4096,
        "n_epochs": 4,
    },
}


def time_component(name: str, fn, *args, **kwargs):
    """Time a single component call."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return elapsed, result


def profile_pipeline(config: dict):
    """Run a profiled training burst and print stats."""
    env_kwargs = dict(config.get("env_kwargs", {}))
    env_kwargs.setdefault("d4_augment", True)
    vec_env = make_vec_env(Game2048Env, n_envs=config["n_envs"], env_kwargs=env_kwargs)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config["features_dim"]),
    )

    ppo_params = config["ppo_params"].copy()
    lr_config = ppo_params.pop("learning_rate")
    if lr_config["type"] == "linear_decay":
        ppo_params["learning_rate"] = lambda p: p * lr_config["initial_value"]

    model = MaskablePPO(
        "CnnPolicy", vec_env, policy_kwargs=policy_kwargs,
        verbose=0, **ppo_params
    )

    print(f"Profiling {config['total_timesteps']:,} timesteps ({config['n_envs']} envs)...")
    print(f"  rollout buffer: {config['ppo_params']['n_steps']} steps x {config['n_envs']} envs = "
          f"{config['ppo_params']['n_steps'] * config['n_envs']} samples/rollout")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    model.learn(total_timesteps=config["total_timesteps"], reset_num_timesteps=True)

    profiler.disable()

    # --- Print top cumulative time -----------------------------------------
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    print(stream.getvalue())

    # --- Bottleneck summary ------------------------------------------------
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")  # self time (excl. subcalls)
    stats2.print_stats(20)
    lines = stream2.getvalue().splitlines()

    print("=" * 70)
    print("SLOWEST FUNCTIONS (self time, excl. subcalls):")
    print("=" * 70)
    for line in lines[2:22]:
        print(line)

    vec_env.close()


if __name__ == "__main__":
    profile_pipeline(CONFIG)
