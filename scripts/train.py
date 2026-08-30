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
import math
import json as _json
import re
import shlex
import subprocess
import yaml
import argparse
import wandb
from collections.abc import Mapping
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
from twenty_forty_eight_ai.agent.ppo import ValueHeadLRMaskablePPO
try:
    from scripts.benchmark_provenance import (
        collect_runtime_provenance,
        sha256_file,
        validate_artifact_sha256,
    )
except ModuleNotFoundError:  # Support `python scripts/train.py`.
    from benchmark_provenance import (
        collect_runtime_provenance,
        sha256_file,
        validate_artifact_sha256,
    )
from twenty_forty_eight_ai.utils.effective_config import (
    D4_SEED_DERIVATION,
    V3_EXPERIMENT_DEFINITION,
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
    """Capture the source Git state before a run creates its artifacts."""
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
    git_dirty_at_start = bool(git_status_porcelain)
    return {
        "git_commit": git_commit,
        "git_commit_at_start": git_commit,
        "git_status_porcelain": git_status_porcelain,
        "git_status_at_start": git_status_porcelain,
        "git_dirty": git_dirty_at_start,
        "git_dirty_at_start": git_dirty_at_start,
    }


def _normalize_git_start_provenance(
    git_provenance: Mapping[str, object],
) -> dict[str, str | bool]:
    """Normalize the initial Git snapshot and preserve legacy manifest aliases."""
    git_commit_at_start = git_provenance.get(
        "git_commit_at_start", git_provenance.get("git_commit")
    )
    git_status_at_start = git_provenance.get(
        "git_status_at_start", git_provenance.get("git_status_porcelain")
    )
    git_dirty_at_start = git_provenance.get(
        "git_dirty_at_start", git_provenance.get("git_dirty")
    )
    if not isinstance(git_commit_at_start, str) or not git_commit_at_start:
        raise RuntimeError("git_commit_at_start must be a non-empty string")
    if not isinstance(git_status_at_start, str):
        raise RuntimeError("git_status_at_start must be a string")
    if not isinstance(git_dirty_at_start, bool):
        raise RuntimeError("git_dirty_at_start must be a boolean")
    if git_dirty_at_start != bool(git_status_at_start):
        raise RuntimeError("git_dirty_at_start does not match git_status_at_start")
    return {
        "git_commit": git_commit_at_start,
        "git_commit_at_start": git_commit_at_start,
        "git_status_porcelain": git_status_at_start,
        "git_status_at_start": git_status_at_start,
        "git_dirty": git_dirty_at_start,
        "git_dirty_at_start": git_dirty_at_start,
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


def qualified_class_name(value) -> str:
    """Return a stable module-qualified class name for an object or class."""
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def seed_from_config(config: dict) -> int:
    """Get seed from config dict, defaulting to 0."""
    return config.get("seed", 0)


def linear_decay_learning_rate(
    initial_value: float, current_timestep: float, total_timesteps: float
) -> float:
    """Return a linearly decayed learning rate that never becomes negative."""
    computed_lr = initial_value * (1.0 - current_timestep / total_timesteps)
    return max(0.0, computed_lr)


def make_linear_decay_schedule(initial_value: float, total_timesteps: float):
    """Build an SB3 schedule from the remaining-progress convention."""
    return lambda progress_remaining: linear_decay_learning_rate(
        initial_value,
        (1.0 - progress_remaining) * total_timesteps,
        total_timesteps,
    )


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
    """Resolve the policy class from the materialized experiment definition."""
    definition = config.get("experiment_definition")
    if isinstance(definition, Mapping) and definition.get("name") == "v3":
        if definition.get("policy_class") != V3_EXPERIMENT_DEFINITION["policy_class"]:
            raise ValueError("Unsupported v3 policy class in experiment definition")
        return ValueNormalizedMaskablePolicy
    return "CnnPolicy"


def select_training_ppo(config: dict):
    """Resolve the PPO algorithm class from the materialized definition."""
    definition = config.get("experiment_definition")
    if isinstance(definition, Mapping) and definition.get("name") == "v3":
        if definition.get("ppo_class") != V3_EXPERIMENT_DEFINITION["ppo_class"]:
            raise ValueError("Unsupported v3 PPO class in experiment definition")
        return ValueHeadLRMaskablePPO
    return MaskablePPO


def select_value_head_lr_multiplier(config: dict) -> float:
    """Return the configured value-head multiplier for the selected PPO class."""
    definition = config.get("experiment_definition")
    if isinstance(definition, Mapping) and definition.get("name") == "v3":
        multiplier = float(definition.get("value_head_lr_multiplier", float("nan")))
        if multiplier != V3_EXPERIMENT_DEFINITION["value_head_lr_multiplier"]:
            raise ValueError("Unsupported v3 value-head multiplier")
        return multiplier
    return 1.0


def build_fresh_model(
    effective_config: dict,
    vec_env,
    policy_kwargs: dict,
    ppo_params: dict,
    seed: int,
):
    """Construct a fresh model from the materialized experiment definition."""
    ppo_class = select_training_ppo(effective_config)
    ppo_kwargs = {}
    if ppo_class is ValueHeadLRMaskablePPO:
        ppo_kwargs["value_head_lr_multiplier"] = (
            select_value_head_lr_multiplier(effective_config)
        )
    return ppo_class(
        select_training_policy(effective_config),
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        **ppo_kwargs,
        **ppo_params,
    )


def validate_v3_seed_sweep(
    config: dict, requested_seed_count: int, *, resume: bool = False
) -> None:
    """Require the configured four-seed sweep for either v3 condition."""
    validate_v3_experiment_config(config)
    if not _is_v3_experiment(config):
        return
    if resume:
        raise ValueError("official v3 seed sweeps cannot use --resume-sweep")
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


def _is_v3_experiment(config: Mapping[str, object]) -> bool:
    definition = config.get("experiment_definition")
    return isinstance(definition, Mapping) and definition.get("name") == "v3"


def _require_empty_output_dir(path: str | Path, description: str) -> None:
    path = Path(path)
    if not path.exists():
        return
    if not path.is_dir() or any(path.iterdir()):
        raise ValueError(f"{description} must be absent or empty: {path}")


def persist_training_manifest(
    model_dir: str | Path,
    model_path: str,
    model,
    effective_config: dict,
    *,
    git_provenance: Mapping[str, object],
) -> Path:
    """Persist immutable provenance bound to the final saved model."""
    effective_config_path = Path(model_dir) / EFFECTIVE_CONFIG_FILENAME
    if not effective_config_path.is_file():
        raise RuntimeError("Effective config must exist before writing training manifest")
    try:
        with effective_config_path.open() as stream:
            persisted_config = _json.load(stream)
    except (OSError, _json.JSONDecodeError) as exc:
        raise RuntimeError("Effective config is not readable") from exc
    if persisted_config != effective_config:
        raise RuntimeError("Effective config on disk differs from in-memory config")
    validate_v3_experiment_config(effective_config)
    git_provenance = _normalize_git_start_provenance(git_provenance)
    env_kwargs = effective_config.get("env_kwargs")
    d4_augment = env_kwargs.get("d4_augment") if isinstance(env_kwargs, dict) else None
    if type(d4_augment) is not bool:
        raise RuntimeError("Effective config must record a boolean D4 condition")

    resolved_model_path = Path(model_path).resolve()
    resolved_effective_config_path = effective_config_path.resolve()
    expected_model_path = Path(model_dir).resolve() / "final_model.zip"
    expected_effective_config_path = Path(model_dir).resolve() / EFFECTIVE_CONFIG_FILENAME
    if resolved_model_path != expected_model_path:
        raise RuntimeError("Training manifest must bind model_dir/final_model.zip")
    if resolved_effective_config_path != expected_effective_config_path:
        raise RuntimeError(
            "Training manifest must bind model_dir/effective_config.json"
        )
    runtime_provenance = collect_runtime_provenance(
        model_path=str(resolved_model_path),
        effective_config=str(resolved_effective_config_path),
    )
    extension = native_extension_identity()
    extension_path = Path(extension["path"]).resolve()
    model_sha256 = sha256_file(resolved_model_path)
    effective_config_sha256 = sha256_file(resolved_effective_config_path)
    native_extension_sha256 = sha256_file(extension_path)
    package_versions = relevant_package_versions()
    required_runtime_fields = (
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
    if (
        missing_runtime_fields
        or missing_package_versions
        or not model_sha256
        or not effective_config_sha256
        or not native_extension_sha256
        or not runtime_provenance.get("uv_lock_sha256")
    ):
        missing = missing_runtime_fields + missing_package_versions
        if not model_sha256:
            missing.append("model_sha256")
        if not effective_config_sha256:
            missing.append("effective_config_sha256")
        if not native_extension_sha256:
            missing.append("native_extension_sha256")
        if not runtime_provenance.get("uv_lock_sha256"):
            missing.append("uv_lock_sha256")
        raise RuntimeError("Missing training provenance: " + ", ".join(missing))
    if not isinstance(git_provenance.get("git_dirty_at_start"), bool):
        raise RuntimeError("git_dirty_at_start must be a boolean")
    if not isinstance(git_provenance.get("git_status_at_start"), str):
        raise RuntimeError("git_status_at_start must be a string")
    if not re.fullmatch(r"[0-9a-f]{40}", git_provenance.get("git_commit", "")):
        raise RuntimeError("git_commit must be a 40-character hexadecimal SHA")
    if git_provenance["git_commit"] != git_provenance["git_commit_at_start"]:
        raise RuntimeError("git_commit does not match git_commit_at_start")
    if git_provenance["git_status_porcelain"] != git_provenance["git_status_at_start"]:
        raise RuntimeError("git_status_porcelain does not match git_status_at_start")
    if git_provenance["git_dirty"] != git_provenance["git_dirty_at_start"]:
        raise RuntimeError("git_dirty does not match git_dirty_at_start")
    if git_provenance["git_dirty_at_start"] != bool(
        git_provenance.get("git_status_at_start")
    ):
        raise RuntimeError("git_dirty_at_start does not match git_status_at_start")
    for field, actual in (
        ("model_sha256", model_sha256),
        ("effective_config_sha256", effective_config_sha256),
        ("native_extension_sha256", native_extension_sha256),
    ):
        recorded = runtime_provenance.get(field)
        if recorded and recorded != actual:
            raise RuntimeError(f"Runtime provenance {field} does not match the artifact")

    definition = effective_config.get("experiment_definition", {})
    is_v3 = definition.get("name") == "v3"
    policy_class = qualified_class_name(model.policy)
    ppo_class = qualified_class_name(model)
    value_head_lr_multiplier = float(
        getattr(model, "value_head_lr_multiplier", definition.get(
            "value_head_lr_multiplier", 1.0
        ))
    )
    if is_v3:
        if policy_class != definition["policy_class"]:
            raise RuntimeError(
                "Training policy does not match effective experiment definition"
            )
        if ppo_class != definition["ppo_class"]:
            raise RuntimeError(
                "Training PPO class does not match effective experiment definition"
            )
        if value_head_lr_multiplier != float(definition["value_head_lr_multiplier"]):
            raise RuntimeError(
                "Training value-head multiplier does not match effective experiment definition"
            )

    fresh_training = (
        effective_config.get("load_model") is False
        and effective_config.get("checkpoint_path") is None
    )
    if is_v3 and not fresh_training:
        raise RuntimeError("v3 training manifest requires a fresh training run")
    git_dirty = git_provenance["git_dirty_at_start"]

    manifest = {
        **git_provenance,
        "effective_config": effective_config,
        "effective_config_path": str(resolved_effective_config_path),
        "training_seed": effective_config["root_training_seed"],
        "d4_augment": d4_augment,
        "d4_condition": "d4" if d4_augment else "no_d4",
        "condition": "d4" if d4_augment else "no_d4",
        "model_path": str(resolved_model_path),
        "model_sha256": model_sha256,
        "policy_class": policy_class,
        "ppo_class": ppo_class,
        "value_head_lr_multiplier": value_head_lr_multiplier,
        "fresh_training": fresh_training,
        "load_model": effective_config.get("load_model"),
        "checkpoint_path": effective_config.get("checkpoint_path"),
        "paper_grade": git_dirty is False,
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
        "native_extension": {
            "path": str(extension_path),
            "sha256": native_extension_sha256,
        },
        "uv_lock_path": str((REPO_ROOT / "uv.lock").resolve()),
        "root_training_seed": effective_config["root_training_seed"],
        "effective_config_sha256": effective_config_sha256,
        "uv_lock_sha256": runtime_provenance.get("uv_lock_sha256", ""),
        "native_extension_sha256": native_extension_sha256,
        "python_version": runtime_provenance["python_version"],
        "torch_version": runtime_provenance["torch_version"],
        "sb3_version": runtime_provenance["sb3_version"],
        "final_timestep": int(model.num_timesteps),
    }
    path = Path(model_dir) / "training_manifest.json"
    with path.open("w") as stream:
        _json.dump(manifest, stream, indent=2, sort_keys=True)
    return path


def validate_training_manifest(manifest_path: str | Path) -> dict:
    """Verify a model-bound training manifest and return its parsed contents."""
    path = Path(manifest_path)
    manifest_dir = path.resolve().parent
    try:
        with path.open() as stream:
            manifest = _json.load(stream)
    except (OSError, _json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable training manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Training manifest must be a JSON object: {path}")

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def manifest_path(field: str) -> Path:
        value = manifest.get(field)
        require(isinstance(value, str) and value, f"manifest {field} is missing")
        return Path(value)

    model_path = manifest_path("model_path")
    effective_config_path = manifest_path("effective_config_path")
    native_extension = manifest.get("native_extension", {})
    require(isinstance(native_extension, dict), "manifest native extension is missing")
    native_extension_path = native_extension.get("path")
    require(
        isinstance(native_extension_path, str) and native_extension_path,
        "manifest native extension path is missing",
    )
    native_extension_path = Path(native_extension_path)
    uv_lock_path_value = manifest.get("uv_lock_path")
    require(
        isinstance(uv_lock_path_value, str) and uv_lock_path_value,
        "manifest uv.lock path is missing",
    )
    uv_lock_path = Path(uv_lock_path_value)
    require(
        model_path == model_path.parent / "final_model.zip",
        "manifest model_path must point to final_model.zip",
    )
    require(
        effective_config_path == effective_config_path.parent / EFFECTIVE_CONFIG_FILENAME,
        "manifest effective_config_path must be adjacent effective_config.json",
    )
    require(
        model_path.parent == manifest_dir
        and effective_config_path.parent == manifest_dir,
        "manifest, model, and effective config must share a directory",
    )
    require(model_path.is_absolute(), "manifest model_path must be absolute")
    require(effective_config_path.is_absolute(), "manifest effective_config_path must be absolute")
    require(native_extension_path.is_absolute(), "manifest native extension path must be absolute")
    require(uv_lock_path.is_absolute(), "manifest uv.lock path must be absolute")
    require(model_path == model_path.resolve(), "manifest model_path must be canonical")
    require(
        effective_config_path == effective_config_path.resolve(),
        "manifest effective_config_path must be canonical",
    )
    require(
        native_extension_path == native_extension_path.resolve(),
        "manifest native extension path must be canonical",
    )
    require(uv_lock_path == uv_lock_path.resolve(), "manifest uv.lock path must be canonical")
    require(
        uv_lock_path == (REPO_ROOT / "uv.lock").resolve(),
        "manifest uv.lock path must point to repository uv.lock",
    )
    for artifact_path, expected_hash, label in (
        (model_path, manifest.get("model_sha256"), "model_sha256"),
        (effective_config_path, manifest.get("effective_config_sha256"), "effective_config_sha256"),
        (native_extension_path, native_extension.get("sha256"), "native_extension_sha256"),
        (uv_lock_path, manifest.get("uv_lock_sha256"), "uv_lock_sha256"),
    ):
        require(
            isinstance(expected_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None,
            f"{label} is missing or invalid",
        )
        try:
            validate_artifact_sha256(artifact_path, expected_hash, label=label)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    native_sha256 = sha256_file(native_extension_path)
    require(
        native_sha256 == manifest.get("native_extension_sha256"),
        "native_extension_sha256 does not match native extension",
    )

    try:
        with effective_config_path.open() as stream:
            effective_config = _json.load(stream)
    except (OSError, _json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable effective config: {effective_config_path}") from exc
    require(
        manifest.get("effective_config") == effective_config,
        "manifest effective_config does not match effective config file",
    )
    validate_v3_experiment_config(effective_config)
    is_v3 = _is_v3_experiment(effective_config)
    if is_v3:
        git_commit_at_start = manifest.get("git_commit_at_start")
        git_status_at_start = manifest.get("git_status_at_start")
        git_dirty_at_start = manifest.get("git_dirty_at_start")
    else:
        git_commit_at_start = manifest.get(
            "git_commit_at_start", manifest.get("git_commit")
        )
        git_status_at_start = manifest.get(
            "git_status_at_start", manifest.get("git_status_porcelain")
        )
        git_dirty_at_start = manifest.get(
            "git_dirty_at_start", manifest.get("git_dirty")
        )
    require(
        isinstance(git_commit_at_start, str)
        and re.fullmatch(r"[0-9a-f]{40}", git_commit_at_start) is not None,
        "manifest git_commit_at_start is missing",
    )
    require(
        isinstance(git_status_at_start, str),
        "manifest git_status_at_start is invalid",
    )
    require(
        isinstance(git_dirty_at_start, bool),
        "manifest git_dirty_at_start is invalid",
    )
    require(
        manifest.get("git_commit") == git_commit_at_start,
        "manifest git_commit does not match git_commit_at_start",
    )
    require(
        manifest.get("git_status_porcelain") == git_status_at_start,
        "manifest git_status_porcelain does not match git_status_at_start",
    )
    require(
        manifest.get("git_dirty") == git_dirty_at_start,
        "manifest git_dirty does not match git_dirty_at_start",
    )
    require(
        git_dirty_at_start == bool(git_status_at_start),
        "manifest git_dirty_at_start does not match git_status_at_start",
    )
    for field in ("python_version", "torch_version", "sb3_version"):
        require(
            isinstance(manifest.get(field), str) and bool(manifest[field]),
            f"manifest {field} is missing",
        )
    versions = manifest.get("versions")
    require(isinstance(versions, dict), "manifest versions are missing")
    require(versions.get("python") == manifest["python_version"], "python version mismatch")
    require(versions.get("torch") == manifest["torch_version"], "torch version mismatch")
    require(
        versions.get("stable-baselines3") == manifest["sb3_version"],
        "sb3 version mismatch",
    )
    for dependency in ("gymnasium", "numpy", "numba", "sb3-contrib", "pybind11"):
        require(
            isinstance(versions.get(dependency), str) and bool(versions[dependency]),
            f"manifest {dependency} version is missing",
        )
    runtime = manifest.get("runtime")
    require(isinstance(runtime, dict), "manifest runtime provenance is missing")
    require(
        isinstance(runtime.get("compiler"), str) and bool(runtime["compiler"]),
        "manifest compiler provenance is missing",
    )
    final_timestep = manifest.get("final_timestep")
    require(
        isinstance(final_timestep, int)
        and not isinstance(final_timestep, bool)
        and final_timestep >= 0,
        "manifest final_timestep is invalid",
    )

    definition = effective_config.get("experiment_definition", {})
    if definition.get("name") == "v3":
        require(type(manifest.get("d4_augment")) is bool, "manifest d4_augment is invalid")
        require(isinstance(manifest.get("condition"), str), "manifest condition is invalid")
        require(isinstance(manifest.get("d4_condition"), str), "manifest d4_condition is invalid")
        require(isinstance(manifest.get("policy_class"), str), "manifest policy class is invalid")
        require(isinstance(manifest.get("ppo_class"), str), "manifest PPO class is invalid")
        multiplier = manifest.get("value_head_lr_multiplier")
        require(
            type(multiplier) in (int, float) and math.isfinite(float(multiplier)),
            "manifest value-head multiplier is invalid",
        )
        require(
            manifest.get("condition") == manifest.get("d4_condition"),
            "manifest condition fields disagree",
        )
        expected_condition = "d4" if effective_config["env_kwargs"]["d4_augment"] else "no_d4"
        require(
            manifest.get("d4_augment") == effective_config["env_kwargs"]["d4_augment"],
            "manifest d4_augment does not match effective config",
        )
        require(
            manifest.get("condition") == expected_condition,
            "manifest condition does not match effective config",
        )
        require(
            manifest.get("training_seed") == effective_config["root_training_seed"],
            "manifest training seed does not match effective config",
        )
        require(
            manifest.get("root_training_seed") == effective_config["root_training_seed"],
            "manifest root training seed does not match effective config",
        )
        require(
            manifest.get("root_training_seed") == effective_config.get("seed"),
            "manifest root training seed does not match config seed",
        )
        require(
            manifest.get("policy_class") == definition["policy_class"],
            "manifest policy class does not match effective config",
        )
        require(
            manifest.get("ppo_class") == definition["ppo_class"],
            "manifest PPO class does not match effective config",
        )
        require(
            manifest.get("value_head_lr_multiplier")
            == float(definition["value_head_lr_multiplier"]),
            "manifest value-head multiplier does not match effective config",
        )
        require(manifest.get("fresh_training") is True, "v3 manifest is not fresh training")
        require(manifest.get("load_model") is False, "v3 manifest load_model must be false")
        require(manifest.get("checkpoint_path") is None, "v3 manifest contains a checkpoint")

        try:
            loaded_model = ValueHeadLRMaskablePPO.load(model_path, device="cpu")
        except Exception as exc:
            raise ValueError("v3 model artifact could not be loaded") from exc
        require(
            qualified_class_name(loaded_model) == manifest.get("ppo_class"),
            "manifest PPO class does not match saved model",
        )
        require(
            qualified_class_name(loaded_model.policy) == manifest.get("policy_class"),
            "manifest policy class does not match saved model",
        )
        require(
            float(getattr(loaded_model, "value_head_lr_multiplier", float("nan")))
            == float(manifest["value_head_lr_multiplier"]),
            "manifest value-head multiplier does not match saved model",
        )
        actual_final_timestep = int(loaded_model.num_timesteps)
        require(
            actual_final_timestep == manifest["final_timestep"],
            "manifest final_timestep does not match saved model",
        )
        configured_timesteps = effective_config.get("total_timesteps")
        if configured_timesteps is not None:
            require(
                actual_final_timestep >= int(configured_timesteps),
                "saved model did not reach configured total_timesteps",
            )

    require(
        manifest.get("paper_grade") is (git_dirty_at_start is False),
        "manifest paper_grade does not match git_dirty_at_start",
    )
    return manifest


def train(
    config: dict,
    *,
    git_provenance_at_start: Mapping[str, object] | None = None,
):
    """Run training loop."""
    seed = seed_from_config(config)
    effective_config, d4_rank_seed_sequences = resolve_training_config(config)
    seed_suffix = f"-seed{seed}"
    run_name = config["run_name"]
    if not run_name.endswith(seed_suffix):
        run_name = f"{run_name}{seed_suffix}"
    model_dir = os.path.join(config['output_dir'], "models", config['run_name'])
    if _is_v3_experiment(effective_config):
        _require_empty_output_dir(model_dir, "v3 fresh training output directory")
    if git_provenance_at_start is None:
        git_provenance_at_start = collect_git_provenance()
    run = wandb.init(
        project=config['project_name'],
        config=effective_config,
        name=run_name,
        save_code=True,
    )
    set_global_seed(seed)

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
                return linear_decay_learning_rate(
                    lr_config['initial_value'], current_global_steps, total_steps
                )

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
            ppo_params['learning_rate'] = make_linear_decay_schedule(
                lr_config['initial_value'], config['total_timesteps']
            )

        model = build_fresh_model(
            effective_config,
            vec_env,
            policy_kwargs,
            ppo_params,
            seed,
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
    manifest_path = persist_training_manifest(
        model_dir,
        final_model_path,
        model,
        effective_config,
        git_provenance=git_provenance_at_start,
    )
    validate_training_manifest(manifest_path)
    print(f"Final model saved to: {final_model_path}")
    print("Training complete!")


def main_with_sweep(config: dict):
    sweep_cfg = config.get("__sweep", {})
    if not sweep_cfg.get("enabled"):
        if _is_v3_experiment(config) and sweep_cfg.get("resume"):
            raise ValueError("official v3 training cannot use --resume-sweep")
        train(config)
        return

    n_seeds = sweep_cfg["n_seeds"]
    is_v3 = _is_v3_experiment(config)
    validate_v3_seed_sweep(config, n_seeds, resume=bool(sweep_cfg["resume"]))
    sweep_name = config.get("run_name", f"sweep_{int(time.time())}")
    output_dir = os.path.join(config["output_dir"], "models", sweep_name)

    if sweep_cfg["parallel"]:
        print("Warning: parallel training not recommended on single-GPU systems.")
        print("Each seed run should be launched independently with --seed.")

    if sweep_cfg["dry_run"]:
        _print_dry_run(config, sweep_name, n_seeds)
        return

    if is_v3:
        _require_empty_output_dir(output_dir, "v3 fresh sweep output directory")
        for seed_i in range(n_seeds):
            _require_empty_output_dir(
                os.path.join(
                    config["output_dir"], "models", f"{sweep_name}-seed{seed_i}"
                ),
                "v3 fresh seed output directory",
            )
        git_provenance_at_start = collect_git_provenance()
    else:
        git_provenance_at_start = None
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
            if is_v3:
                train(
                    seed_config,
                    git_provenance_at_start=git_provenance_at_start,
                )
            else:
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
    effective_config = materialize_training_config(config)
    sweep_cfg = config.get("__sweep", {})
    resume = bool(sweep_cfg.get("resume"))
    parallel = bool(sweep_cfg.get("parallel"))
    mode = "Resume" if resume else "Fresh"
    execution = "sequential (parallel requested)" if parallel else "sequential"
    print(f"Sweep: {sweep_name} ({n_seeds} seeds, {execution})")
    print(f"W&B group: {sweep_name}")
    config_path = config.get("_config_path", "<config>")
    print(f"Config: {config_path}")
    command = (
        f"python scripts/train.py --config {shlex.quote(str(config_path))} "
        f"--seed-sweep {n_seeds}"
    )
    if resume:
        command += " --resume-sweep"
    if parallel:
        command += " --parallel"
    print(f"{mode} launch command: {command}")
    status_path = os.path.join(
        config["output_dir"], "models", sweep_name, SweepStatusPath,
    )
    status_action = "loaded" if resume else "reinitialized"
    print(f"{mode} sweep status path ({status_action}): {status_path}")
    print()

    for i in range(n_seeds):
        seed_run_name = f"{sweep_name}-seed{i}"
        model_dir = os.path.join(config["output_dir"], "models", seed_run_name)
        wandb_run_name = seed_run_name
        print(
            f"Seed {i} │ seed={i} │ run={seed_run_name} │ "
            f"W&B name={wandb_run_name} │ "
            f"d4_augment={effective_config['env_kwargs']['d4_augment']} │ "
            f"total_timesteps={effective_config['total_timesteps']} │ "
            f"fresh={not resume} │ model_dir={model_dir} │ "
            f"final_model={os.path.join(model_dir, 'final_model.zip')}"
        )

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
