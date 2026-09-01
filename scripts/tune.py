"""The official PRE-FREEZE v3 paired tuner.

Each Optuna trial samples one PPO configuration and trains it from scratch in
both conditions.  The two conditions share the training seed, evaluation
seeds, budget, architecture, reward implementation, and evaluator; only
``env_kwargs.d4_augment`` changes.

The tuner deliberately has a separate configuration/materialization path from
the official 200M v3 validator. Its artifacts live under
``data/tuning/v3/<study>/`` and are diagnostic PRE-FREEZE outputs, never paper
results by themselves.

Usage::

    uv run python scripts/tune.py \
        --config configs/tune/bayesian_opt_search.yaml
    uv run python scripts/tune.py \
        --config configs/tune/bayesian_opt_search.yaml --dry-run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import optuna
import torch
import wandb
import yaml
from optuna.exceptions import TrialPruned
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ``python scripts/tune.py`` puts ``scripts/`` rather than the repository root
# first on sys.path. Keep the documented direct CLI invocation working while
# retaining normal package imports for pytest and module execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train as training
from scripts.benchmark_io import EpisodeResult
from scripts.benchmark_provenance import (
    collect_runtime_provenance,
    sha256_file,
    validate_artifact_sha256,
)
from twenty_forty_eight_ai.agent.architecture import CustomCNN
from twenty_forty_eight_ai.agent.policy import ValueNormalizedMaskablePolicy
from twenty_forty_eight_ai.agent.ppo import ValueHeadLRMaskablePPO
from twenty_forty_eight_ai.evaluation.benchmarker import (
    BenchmarkEpisodeError,
    Benchmarker,
)
from twenty_forty_eight_ai.evaluation.value_evaluator import D4ValueEvaluator
from twenty_forty_eight_ai.utils.effective_config import (
    D4_SEED_DERIVATION,
    V3_EXPERIMENT_DEFINITION,
    compare_behavioral_configs,
    derive_d4_rank_seed_sequences,
    load_effective_config,
    validate_v3_experiment_config,
)

PROTOCOL_STATUS = "PRE-FREEZE"
TUNING_PROTOCOL_VERSION = "v3-stage1-paired-1"
CONDITIONS = ("d4", "no_d4")
STAGE1_BUDGET = 20_000_000
STAGE1_CHECKPOINTS = (5_000_000, 10_000_000, 20_000_000)
V3_REPRESENTATION = "integer_tile_exponents_0..15"
V3_REWARD_FUNCTION = "twenty_forty_eight_ai.env.reward.calculate_reward"
V3_EVALUATOR_CLASS = (
    "twenty_forty_eight_ai.evaluation.value_evaluator.D4ValueEvaluator"
)
V3_BENCHMARK_PIPELINE = "twenty_forty_eight_ai.evaluation.benchmarker.Benchmarker"
SEARCH_PARAMETER_NAMES = (
    "learning_rate",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "vf_coef",
    "clip_range",
)
_STUDY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CRITIC_METRIC_FIELDS = (
    "explained_variance",
    "value_std",
    "value_max_abs",
)


def _capture_rng_state() -> tuple[Any, tuple[Any, ...], torch.Tensor, list[torch.Tensor] | None]:
    """Capture process RNGs so paired condition training cannot cross-contaminate."""
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), cuda_states


def _restore_rng_state(
    state: tuple[Any, tuple[Any, ...], torch.Tensor, list[torch.Tensor] | None]
) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state[3])


class TuningError(RuntimeError):
    """Base class for explicit tuning configuration/runtime errors."""


class InvalidTrialError(TuningError):
    """A trial failed validity checks and must be recorded as a failed trial."""


class StudyConfigurationMismatch(TuningError):
    """A persisted study cannot be resumed with a different immutable config."""


class ConditionSession(Protocol):
    """Small seam used by the paired runner and its dry/unit-test doubles."""

    condition: str

    def advance_to(self, target_timesteps: int) -> int: ...

    def checkpoint_path(self, target_timesteps: int) -> Path: ...

    def training_metrics(self, target_timesteps: int) -> Mapping[str, Any] | "CheckpointMetrics": ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CheckpointMetrics:
    """Metrics required to accept one paired checkpoint."""

    checkpoint_timesteps: int
    actual_timesteps: int
    depth0_mean_score: float
    depth1_mean_score: float
    depth1_minus_depth0: float
    explained_variance: float
    value_std: float
    value_max_abs: float
    episode_length: float
    training_reward: float
    cap_count: int
    unresolved_count: int
    invalid_count: int


@dataclass(frozen=True)
class StudySpec:
    """Validated immutable inputs for one Optuna study."""

    config: dict[str, Any]
    config_path: Path
    study_name: str
    storage_root: Path
    study_dir: Path
    db_path: Path
    n_trials: int
    timeout_seconds: float | None
    training_seed: int
    trial_budget: int
    checkpoints: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    evaluation_depths: tuple[int, ...]
    evaluation_device: str
    search_space: dict[str, dict[str, Any]]
    sampler_config: dict[str, Any]
    pruner_config: dict[str, Any]
    fixed: dict[str, Any]
    base_configs: dict[str, Path]
    base_config_data: dict[str, dict[str, Any]]
    base_config_hashes: dict[str, str]
    wandb_config: dict[str, Any]
    critic_max_abs: float
    study_config: dict[str, Any]
    study_config_sha256: str


def _json_ready(value: Any) -> Any:
    """Convert nested config values to deterministic JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def study_config_hash(config: Mapping[str, Any]) -> str:
    """Hash the canonical immutable study configuration."""
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _relative_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def load_tuning_yaml(path: str | Path) -> dict[str, Any]:
    """Load and minimally type-check a PRE-FREEZE tuner YAML file."""
    config_path = Path(path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("tuning config must contain a mapping")
    return copy.deepcopy(dict(config))


def _require_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _resolve_repo_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _validate_search_space(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ValueError("search_space must be a mapping")
    if set(raw.keys()) != set(SEARCH_PARAMETER_NAMES):
        raise ValueError(
            "search_space must contain exactly: "
            + ", ".join(SEARCH_PARAMETER_NAMES)
        )

    result: dict[str, dict[str, Any]] = {}
    for name in SEARCH_PARAMETER_NAMES:
        definition = raw[name]
        if not isinstance(definition, Mapping):
            raise ValueError(f"search_space.{name} must be a mapping")
        kind = definition.get("type")
        if kind == "float":
            low = _require_finite_float(definition.get("low"), f"{name}.low")
            high = _require_finite_float(definition.get("high"), f"{name}.high")
            baseline = _require_finite_float(
                definition.get("baseline"), f"{name}.baseline"
            )
            if low > high or not low <= baseline <= high:
                raise ValueError(f"{name} baseline/bounds are inconsistent")
            log_space = definition.get("log", False)
            if type(log_space) is not bool:
                raise ValueError(f"{name}.log must be boolean")
            if low <= 0.0 and log_space:
                raise ValueError(f"{name} log-space lower bound must be positive")
            result[name] = {
                "type": "float",
                "low": low,
                "high": high,
                "log": log_space,
                "baseline": baseline,
            }
        elif kind == "categorical":
            choices = list(definition.get("choices", []))
            if not choices:
                raise ValueError(f"{name}.choices must not be empty")
            if "baseline" not in definition or definition["baseline"] not in choices:
                raise ValueError(f"{name} baseline must be one of choices")
            result[name] = {
                "type": "categorical",
                "choices": choices,
                "baseline": definition["baseline"],
            }
        else:
            raise ValueError(f"{name} has unsupported search-space type: {kind}")
    return result


def _validate_sampler(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("type") != "TPESampler":
        raise ValueError("sampler.type must be TPESampler")
    seed = _require_int(raw.get("seed"), "sampler.seed")
    if seed < 0:
        raise ValueError("sampler.seed must be non-negative")
    return {"type": "TPESampler", "seed": seed}


def _validate_pruner(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("type") != "MedianPruner":
        raise ValueError("pruner.type must be MedianPruner")
    result = {
        "type": "MedianPruner",
        "n_startup_trials": _require_int(
            raw.get("n_startup_trials"), "pruner.n_startup_trials"
        ),
        "n_warmup_steps": _require_int(
            raw.get("n_warmup_steps"), "pruner.n_warmup_steps"
        ),
        "interval_steps": _require_int(
            raw.get("interval_steps"), "pruner.interval_steps", positive=True
        ),
        "n_min_trials": _require_int(
            raw.get("n_min_trials"), "pruner.n_min_trials", positive=True
        ),
    }
    if result["n_startup_trials"] < 0 or result["n_warmup_steps"] < 0:
        raise ValueError("pruner startup/warmup values must be non-negative")
    return result


def _validate_official_base_configs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], dict[str, str]]:
    raw_paths = config.get("base_configs")
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != set(CONDITIONS):
        raise ValueError("base_configs must define d4 and no_d4 paths")

    paths = {
        condition: _resolve_repo_path(raw_paths[condition], label=f"base_configs.{condition}")
        for condition in CONDITIONS
    }
    data = {condition: load_effective_config(path) for condition, path in paths.items()}
    for condition in CONDITIONS:
        validate_v3_experiment_config(data[condition])

    diff = compare_behavioral_configs(data["d4"], data["no_d4"])
    if diff != {"env_kwargs.d4_augment": (True, False)}:
        raise ValueError(
            "official v3 base configs must differ only by env_kwargs.d4_augment"
        )
    hashes = {condition: sha256_file(path) for condition, path in paths.items()}
    if not all(hashes.values()):
        raise ValueError("unable to hash both official v3 base configs")
    return paths, data, hashes


def build_study_spec(
    config: Mapping[str, Any], *, config_path: str | Path
) -> StudySpec:
    """Validate and materialize the immutable inputs for a tuning study."""
    raw = copy.deepcopy(dict(config))
    if raw.get("protocol_status") != PROTOCOL_STATUS:
        raise ValueError("tuning protocol_status must be PRE-FREEZE")
    if raw.get("protocol_version") != TUNING_PROTOCOL_VERSION:
        raise ValueError("unsupported tuning protocol_version")

    study_name = raw.get("study_name")
    if not isinstance(study_name, str) or not _STUDY_NAME_PATTERN.fullmatch(study_name):
        raise ValueError("study_name must be a safe path component")

    n_trials = _require_int(raw.get("n_trials"), "n_trials", positive=True)
    timeout_hours = raw.get("timeout_hours")
    timeout_seconds = None
    if timeout_hours is not None:
        timeout_seconds = _require_finite_float(timeout_hours, "timeout_hours") * 3600.0
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_hours must be positive when configured")

    training_seed = _require_int(raw.get("training_seed"), "training_seed")
    if training_seed != 0:
        raise ValueError("Stage 1 requires training_seed=0")

    trial_config = raw.get("trial")
    if not isinstance(trial_config, Mapping):
        raise ValueError("trial must be a mapping")
    trial_budget = _require_int(
        trial_config.get("total_timesteps"), "trial.total_timesteps", positive=True
    )
    if trial_budget != STAGE1_BUDGET:
        raise ValueError(
            f"Stage 1 trial.total_timesteps must equal {STAGE1_BUDGET}"
        )
    checkpoint_values = tuple(
        _require_int(value, "trial.checkpoint_timesteps", positive=True)
        for value in trial_config.get("checkpoint_timesteps", [])
    )
    if checkpoint_values != STAGE1_CHECKPOINTS:
        raise ValueError(
            "Stage 1 checkpoint_timesteps must be exactly "
            f"{list(STAGE1_CHECKPOINTS)}"
        )

    base_paths, base_data, base_hashes = _validate_official_base_configs(raw)
    fixed_raw = raw.get("fixed")
    if not isinstance(fixed_raw, Mapping):
        raise ValueError("fixed must be a mapping")
    fixed = dict(fixed_raw)
    expected_fixed = {
        "n_steps": base_data["d4"]["ppo_params"]["n_steps"],
        "batch_size": base_data["d4"]["ppo_params"]["batch_size"],
        "n_epochs": base_data["d4"]["ppo_params"]["n_epochs"],
        "n_envs": base_data["d4"]["n_envs"],
        "features_dim": base_data["d4"]["features_dim"],
        "value_head_lr_multiplier": V3_EXPERIMENT_DEFINITION[
            "value_head_lr_multiplier"
        ],
        "policy_class": V3_EXPERIMENT_DEFINITION["policy_class"],
        "ppo_class": V3_EXPERIMENT_DEFINITION["ppo_class"],
        "representation": V3_REPRESENTATION,
        "reward": V3_REWARD_FUNCTION,
        "evaluator": V3_EVALUATOR_CLASS,
        "benchmark_pipeline": V3_BENCHMARK_PIPELINE,
    }
    if fixed != expected_fixed:
        raise ValueError(
            "fixed tuning values must match the current official v3 architecture/config"
        )

    search_space = _validate_search_space(raw.get("search_space"))
    sampler_config = _validate_sampler(raw.get("sampler"))
    pruner_config = _validate_pruner(raw.get("pruner"))

    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")
    evaluation_seeds = tuple(
        _require_int(seed, "evaluation.seeds", positive=True)
        for seed in evaluation.get("seeds", [])
    )
    if not evaluation_seeds or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("evaluation.seeds must contain distinct seeds")
    if training_seed in evaluation_seeds:
        raise ValueError("evaluation seeds must be separate from training_seed")
    evaluation_depths = tuple(evaluation.get("depths", []))
    if evaluation_depths != (0, 1):
        raise ValueError("evaluation.depths must be exactly [0, 1]")
    evaluation_device = evaluation.get("device", "auto")
    if not isinstance(evaluation_device, str) or not evaluation_device:
        raise ValueError("evaluation.device must be a non-empty string")

    critic_config = raw.get("critic")
    if not isinstance(critic_config, Mapping):
        raise ValueError("critic must be a mapping")
    critic_max_abs = _require_finite_float(
        critic_config.get("max_abs_value"), "critic.max_abs_value", positive=True
    )

    wandb_config = raw.get("wandb", {})
    if not isinstance(wandb_config, Mapping):
        raise ValueError("wandb must be a mapping")
    wandb_config = dict(wandb_config)
    if wandb_config.get("enabled", True):
        if not isinstance(wandb_config.get("project"), str) or not wandb_config["project"]:
            raise ValueError("wandb.project is required when W&B is enabled")

    storage_root = Path(raw.get("storage_root", "data/tuning/v3"))
    if not storage_root.is_absolute():
        storage_root = REPO_ROOT / storage_root
    storage_root = storage_root.resolve()
    study_dir = storage_root / study_name
    db_path = study_dir / "study.db"

    resolved_config_path = Path(config_path).resolve()
    git_provenance = training.collect_git_provenance()
    native_identity = training.native_extension_identity()
    uv_lock_sha = sha256_file(REPO_ROOT / "uv.lock")
    if not uv_lock_sha:
        raise ValueError("unable to hash uv.lock for study provenance")

    study_config = {
        "protocol_status": PROTOCOL_STATUS,
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "study_name": study_name,
        "config_path": _relative_repo_path(resolved_config_path),
        "base_configs": {
            condition: {
                "path": _relative_repo_path(base_paths[condition]),
                "sha256": base_hashes[condition],
            }
            for condition in CONDITIONS
        },
        "search_space": search_space,
        "sampler": sampler_config,
        "pruner": pruner_config,
        "trial": {
            "total_timesteps": trial_budget,
            "checkpoint_timesteps": list(checkpoint_values),
        },
        "execution": {
            "n_trials": n_trials,
            "timeout_hours": timeout_hours,
        },
        "training_seed": training_seed,
        "evaluation": {
            "seeds": list(evaluation_seeds),
            "depths": list(evaluation_depths),
            "device": evaluation_device,
        },
        "fixed": fixed,
        "critic": {"max_abs_value": critic_max_abs},
        "wandb": wandb_config,
        "provenance": {
            "git_commit": git_provenance["git_commit"],
            "git_dirty_at_spec": git_provenance["git_dirty_at_start"],
            "native_extension_sha256": native_identity["sha256"],
            "native_extension_path": native_identity["path"],
            "uv_lock_sha256": uv_lock_sha,
        },
    }

    return StudySpec(
        config=raw,
        config_path=resolved_config_path,
        study_name=study_name,
        storage_root=storage_root,
        study_dir=study_dir,
        db_path=db_path,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        training_seed=training_seed,
        trial_budget=trial_budget,
        checkpoints=checkpoint_values,
        evaluation_seeds=evaluation_seeds,
        evaluation_depths=evaluation_depths,
        evaluation_device=evaluation_device,
        search_space=search_space,
        sampler_config=sampler_config,
        pruner_config=pruner_config,
        fixed=fixed,
        base_configs=base_paths,
        base_config_data=base_data,
        base_config_hashes=base_hashes,
        wandb_config=wandb_config,
        critic_max_abs=critic_max_abs,
        study_config=study_config,
        study_config_sha256=study_config_hash(study_config),
    )


def ensure_study_config(study_dir: str | Path, config: Mapping[str, Any]) -> str:
    """Create the immutable study config, or reject a resume mismatch."""
    directory = Path(study_dir)
    path = directory / "study_config.json"
    hash_path = directory / "study_config.sha256"
    expected_bytes = _canonical_json(config)
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    if path.exists():
        try:
            existing_bytes = path.read_bytes()
        except OSError as exc:
            raise StudyConfigurationMismatch(
                f"unable to read immutable study config: {path}"
            ) from exc
        if hashlib.sha256(existing_bytes).hexdigest() != expected_hash:
            raise StudyConfigurationMismatch(
                "existing study has an incompatible immutable configuration"
            )
        if hash_path.exists():
            try:
                stored_hash = hash_path.read_text().strip()
            except OSError as exc:
                raise StudyConfigurationMismatch(
                    f"unable to read immutable study config hash: {hash_path}"
                ) from exc
            if stored_hash != expected_hash:
                raise StudyConfigurationMismatch(
                    "existing study has an incompatible immutable configuration hash"
                )
        else:
            hash_path.write_text(f"{expected_hash}\n")
        return expected_hash

    if hash_path.exists():
        raise StudyConfigurationMismatch(
            "immutable study config hash exists but study_config.json is missing"
        )
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected_bytes)
    hash_path.write_text(f"{expected_hash}\n")
    return expected_hash


def make_sampler(spec: StudySpec):
    if spec.sampler_config["type"] != "TPESampler":
        raise ValueError("unsupported sampler")
    return optuna.samplers.TPESampler(seed=spec.sampler_config["seed"])


def make_pruner(spec: StudySpec):
    if spec.pruner_config["type"] != "MedianPruner":
        raise ValueError("unsupported pruner")
    return optuna.pruners.MedianPruner(
        n_startup_trials=spec.pruner_config["n_startup_trials"],
        n_warmup_steps=spec.pruner_config["n_warmup_steps"],
        interval_steps=spec.pruner_config["interval_steps"],
        n_min_trials=spec.pruner_config["n_min_trials"],
    )


def default_sampled_params(spec: StudySpec) -> dict[str, Any]:
    """Return the documented v3 baseline as a deterministic dry-run sample."""
    return {
        name: definition["baseline"]
        for name, definition in spec.search_space.items()
    }


def _validate_sampled_params(
    sampled_params: Mapping[str, Any], search_space: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(sampled_params) != set(search_space):
        raise InvalidTrialError("sampled hyperparameters do not match the search space")
    normalized: dict[str, Any] = {}
    for name, definition in search_space.items():
        value = sampled_params[name]
        if definition["type"] == "categorical":
            if value not in definition["choices"]:
                raise InvalidTrialError(f"sampled {name} is outside categorical choices")
            normalized[name] = value
            continue
        value = _require_finite_float(value, f"sampled {name}")
        if not definition["low"] <= value <= definition["high"]:
            raise InvalidTrialError(f"sampled {name} is outside configured bounds")
        normalized[name] = value
    return normalized


def sample_hyperparameters(
    trial: optuna.Trial, spec: StudySpec
) -> dict[str, Any]:
    """Sample exactly the six allowed PPO hyperparameters, once per trial."""
    sampled: dict[str, Any] = {}
    for name, definition in spec.search_space.items():
        if definition["type"] == "categorical":
            sampled[name] = trial.suggest_categorical(name, definition["choices"])
        else:
            sampled[name] = trial.suggest_float(
                name,
                definition["low"],
                definition["high"],
                log=definition["log"],
            )
    return _validate_sampled_params(sampled, spec.search_space)


def make_clamped_linear_schedule(initial_value: float, total_timesteps: int):
    """Return a global-timestep linear schedule clamped at zero.

    ``OnPolicyAlgorithm.learn`` can be stopped and resumed at each checkpoint.
    Therefore the schedule keeps its global timestep independently of SB3's
    per-call progress argument.
    """
    initial = _require_finite_float(initial_value, "learning_rate", positive=True)
    total = _require_int(total_timesteps, "total_timesteps", positive=True)

    class ClampedLinearSchedule:
        def __init__(self) -> None:
            self.current_timestep = 0

        def __call__(self, _progress_remaining: float) -> float:
            return training.linear_decay_learning_rate(
                initial, self.current_timestep, total
            )

    return ClampedLinearSchedule()


def materialize_tuning_config(
    spec: StudySpec,
    sampled_params: Mapping[str, Any],
    *,
    condition: str,
    output_dir: str | Path,
    trial_number: int = 0,
) -> dict[str, Any]:
    """Create a short, explicit v3 tuning config without changing base YAML."""
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    sampled = _validate_sampled_params(sampled_params, spec.search_space)
    effective = copy.deepcopy(spec.base_config_data[condition])
    validate_v3_experiment_config(effective)

    effective["run_name"] = f"{spec.study_name}-trial-{trial_number}-{condition}"
    effective["output_dir"] = str(Path(output_dir).resolve())
    effective["total_timesteps"] = spec.trial_budget
    effective["seed"] = spec.training_seed
    # A tuning condition is one explicit fresh seed, not the official
    # four-seed paper matrix inherited by the base config.
    effective["training_seeds"] = [spec.training_seed]
    seed_sequences = derive_d4_rank_seed_sequences(
        spec.training_seed, spec.fixed["n_envs"]
    )
    effective["root_training_seed"] = spec.training_seed
    effective["d4_seed_derivation"] = D4_SEED_DERIVATION
    effective["d4_rank_spawn_keys"] = [
        list(sequence.spawn_key) for sequence in seed_sequences
    ]
    effective["load_model"] = False
    effective["checkpoint_path"] = None
    effective["n_envs"] = spec.fixed["n_envs"]
    effective["features_dim"] = spec.fixed["features_dim"]
    effective["env_kwargs"] = copy.deepcopy(effective.get("env_kwargs", {}))
    effective["env_kwargs"]["d4_augment"] = condition == "d4"
    effective["ppo_params"] = {
        "n_steps": spec.fixed["n_steps"],
        "gamma": sampled["gamma"],
        "gae_lambda": sampled["gae_lambda"],
        "ent_coef": sampled["ent_coef"],
        "vf_coef": sampled["vf_coef"],
        "learning_rate": {
            "type": "linear_decay",
            "initial_value": sampled["learning_rate"],
        },
        "clip_range": sampled["clip_range"],
        "batch_size": spec.fixed["batch_size"],
        "n_epochs": spec.fixed["n_epochs"],
    }
    effective["evaluation"] = {
        "depths": list(spec.evaluation_depths),
        "seeds": list(spec.evaluation_seeds),
        "device": spec.evaluation_device,
    }
    effective["tuning"] = {
        "protocol_status": PROTOCOL_STATUS,
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "study_name": spec.study_name,
        "trial_number": trial_number,
        "condition": condition,
        "training_seed": spec.training_seed,
        "trial_budget": spec.trial_budget,
        "checkpoint_timesteps": list(spec.checkpoints),
        "sampled_hyperparameters": sampled,
        "fixed": copy.deepcopy(spec.fixed),
        "base_config": _relative_repo_path(spec.base_configs[condition]),
        "base_config_sha256": spec.base_config_hashes[condition],
        "paper_grade": False,
    }
    return effective


def condition_behavioral_diff(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Compare only execution behavior, excluding tuning identifiers/metadata."""
    keys = (
        "total_timesteps",
        "n_envs",
        "features_dim",
        "seed",
        "training_seeds",
        "load_model",
        "checkpoint_path",
        "ppo_params",
        "env_kwargs",
        "experiment_definition",
    )
    first_view = {key: first.get(key) for key in keys}
    second_view = {key: second.get(key) for key in keys}
    return compare_behavioral_configs(first_view, second_view)


def build_v3_tuning_model(effective_config: Mapping[str, Any], vec_env, *, seed: int):
    """Construct the exact v3 policy/PPO pair for a tuning run."""
    definition = effective_config.get("experiment_definition")
    if definition != V3_EXPERIMENT_DEFINITION:
        raise InvalidTrialError("tuning config does not declare the exact v3 definition")
    # Keep an injected schedule object shared with the session. A deep copy
    # here would silently disconnect the global-timestep schedule from the
    # callback that advances it at each checkpoint.
    params = dict(effective_config["ppo_params"])
    learning_rate = params.get("learning_rate")
    if isinstance(learning_rate, Mapping):
        if learning_rate.get("type") != "linear_decay":
            raise InvalidTrialError("tuning learning_rate must be linear_decay")
        params["learning_rate"] = make_clamped_linear_schedule(
            learning_rate["initial_value"], effective_config["total_timesteps"]
        )
    elif not callable(learning_rate):
        raise InvalidTrialError("tuning learning_rate must be a schedule")

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": effective_config["features_dim"]
        },
    }
    model = training.build_fresh_model(
        dict(effective_config),
        vec_env,
        policy_kwargs,
        params,
        seed,
    )
    if type(model) is not ValueHeadLRMaskablePPO:
        raise InvalidTrialError("tuning did not construct ValueHeadLRMaskablePPO")
    if type(model.policy) is not ValueNormalizedMaskablePolicy:
        raise InvalidTrialError(
            "tuning did not construct ValueNormalizedMaskablePolicy"
        )
    if float(model.value_head_lr_multiplier) != 10.0:
        raise InvalidTrialError("tuning value-head LR multiplier is not fixed at 10")
    return model


class StopAtTimestepCallback(BaseCallback):
    """Stop one SB3 learn segment at a reproducible checkpoint boundary."""

    def __init__(self, target_timesteps: int, schedule, verbose: int = 0):
        super().__init__(verbose)
        self.target_timesteps = target_timesteps
        self.schedule = schedule

    def _on_step(self) -> bool:
        current = int(self.model.num_timesteps)
        self.schedule.current_timestep = current
        return current < self.target_timesteps


def _critic_probe_metrics(model) -> tuple[float, float]:
    """Measure the current critic on a fixed, non-gameplay probe corpus."""
    probe_boards = np.asarray(
        [
            np.zeros((4, 4), dtype=np.float32),
            np.eye(4, dtype=np.float32),
            np.fliplr(np.eye(4, dtype=np.float32)) * 4.0,
            np.asarray(
                [[1, 2, 3, 4], [4, 3, 2, 1], [0, 1, 0, 2], [2, 0, 4, 0]],
                dtype=np.float32,
            ),
        ],
    )[:, np.newaxis, :, :]
    observations = torch.as_tensor(probe_boards, device=model.policy.device)
    with torch.no_grad():
        values = model.policy.predict_values(observations).detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan"), float("nan")
    return float(np.std(values)), float(np.max(np.abs(values)))


class ConditionTrainingSession:
    """One fresh v3 model/env pair that advances through Stage 1 checkpoints."""

    def __init__(
        self,
        condition: str,
        spec: StudySpec,
        sampled_params: Mapping[str, Any],
        condition_dir: str | Path,
        trial_number: int,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unsupported condition: {condition}")
        self.condition = condition
        self.spec = spec
        self.sampled_params = dict(sampled_params)
        self.condition_dir = Path(condition_dir)
        if self.condition_dir.exists() and any(self.condition_dir.iterdir()):
            raise InvalidTrialError(
                f"tuning condition artifact directory is not empty: {self.condition_dir}"
            )
        self.condition_dir.mkdir(parents=True, exist_ok=True)
        self.effective_config = materialize_tuning_config(
            spec,
            sampled_params,
            condition=condition,
            output_dir=self.condition_dir,
            trial_number=trial_number,
        )
        self.effective_config_path = self.condition_dir / "effective_config.json"
        _write_json(self.effective_config_path, self.effective_config)
        self.model = None
        self.vec_env = None
        self.wandb_run = None
        self.schedule = None
        self._rng_state = None
        self.closed = False
        try:
            training.set_global_seed(spec.training_seed)
            seed_sequences = derive_d4_rank_seed_sequences(
                spec.training_seed, spec.fixed["n_envs"]
            )
            self.vec_env = DummyVecEnv(
                training.make_training_env_factories(
                    self.effective_config["env_kwargs"], seed_sequences
                )
            )
            model_config = copy.deepcopy(self.effective_config)
            self.schedule = make_clamped_linear_schedule(
                self.sampled_params["learning_rate"], spec.trial_budget
            )
            model_config["ppo_params"]["learning_rate"] = self.schedule
            self.model = build_v3_tuning_model(
                model_config, self.vec_env, seed=spec.training_seed
            )
            # Capture before W&B initialization so tracker internals cannot
            # alter the training stream for either condition.
            self._rng_state = _capture_rng_state()
            self.wandb_run = _start_wandb_run(
                spec,
                trial_number,
                condition,
                self.effective_config,
            )
            self._write_manifest(status="running")
        except Exception:
            self.close()
            raise

    def advance_to(self, target_timesteps: int) -> int:
        if self.model is None or self.schedule is None:
            raise InvalidTrialError("training session is not initialized")
        current = int(self.model.num_timesteps)
        if target_timesteps <= current:
            raise InvalidTrialError(
                f"checkpoint {target_timesteps} is not ahead of {current}"
            )
        if self._rng_state is None:
            raise InvalidTrialError("training RNG state is not initialized")
        _restore_rng_state(self._rng_state)
        try:
            self.schedule.current_timestep = current
            callback = StopAtTimestepCallback(target_timesteps, self.schedule)
            self.model.learn(
                total_timesteps=target_timesteps - current,
                callback=callback,
                reset_num_timesteps=current == 0,
                progress_bar=False,
                tb_log_name=f"{self.spec.study_name}-trial-{self.condition}",
            )
        finally:
            self._rng_state = _capture_rng_state()
        actual = int(self.model.num_timesteps)
        self.schedule.current_timestep = actual
        if actual < target_timesteps:
            raise InvalidTrialError(
                f"training stopped before checkpoint {target_timesteps}: {actual}"
            )
        return actual

    def checkpoint_path(self, target_timesteps: int) -> Path:
        if self.model is None:
            raise InvalidTrialError("training session is not initialized")
        stem = self.condition_dir / f"checkpoint_{target_timesteps}_steps"
        self.model.save(str(stem))
        path = stem.with_suffix(".zip")
        validate_artifact_sha256(path, label="tuning checkpoint")
        return path

    def final_model_path(self) -> Path:
        if self.model is None:
            raise InvalidTrialError("training session is not initialized")
        stem = self.condition_dir / "final_model"
        self.model.save(str(stem))
        path = stem.with_suffix(".zip")
        validate_artifact_sha256(path, label="tuning final model")
        return path

    def training_metrics(self, target_timesteps: int) -> dict[str, Any]:
        if self.model is None:
            raise InvalidTrialError("training session is not initialized")
        if not self.model.ep_info_buffer:
            raise InvalidTrialError(
                f"no training episode metrics at checkpoint {target_timesteps}"
            )
        rewards = [float(info["r"]) for info in self.model.ep_info_buffer]
        lengths = [float(info["l"]) for info in self.model.ep_info_buffer]
        if not rewards or not lengths:
            raise InvalidTrialError("training episode metrics are empty")
        logger_values = getattr(self.model.logger, "name_to_value", {})
        explained_variance = logger_values.get("train/explained_variance", float("nan"))
        value_std, value_max_abs = _critic_probe_metrics(self.model)
        return {
            "actual_timesteps": int(self.model.num_timesteps),
            "explained_variance": float(explained_variance),
            "value_std": value_std,
            "value_max_abs": value_max_abs,
            "episode_length": float(np.mean(lengths)),
            "training_reward": float(np.mean(rewards)),
            "cap_count": 0,
            "unresolved_count": 0,
            "invalid_count": 0,
        }

    def _write_manifest(self, *, status: str, **extra: Any) -> None:
        manifest = build_tuning_manifest(
            study_name=self.spec.study_name,
            trial_number=int(self.effective_config["tuning"]["trial_number"]),
            condition=self.condition,
            sampled_params=self.sampled_params,
            training_seed=self.spec.training_seed,
            evaluation_seeds=self.spec.evaluation_seeds,
            effective_config=self.effective_config,
            status=status,
            **extra,
        )
        manifest["effective_config_path"] = str(self.effective_config_path.resolve())
        manifest["effective_config_sha256"] = sha256_file(self.effective_config_path)
        manifest["study_config_sha256"] = self.spec.study_config_sha256
        manifest["base_config_sha256"] = self.spec.base_config_hashes[self.condition]
        if self.wandb_run is not None:
            manifest["wandb_run_id"] = str(getattr(self.wandb_run, "id", ""))
        model_path = extra.get("model_path")
        if model_path:
            model_path = Path(model_path).resolve()
            manifest["model_path"] = str(model_path)
            manifest["model_sha256"] = validate_artifact_sha256(
                model_path, label="tuning model"
            )
            provenance = _runtime_provenance(
                model_path, self.effective_config_path
            )
            expected = self.spec.study_config["provenance"]
            for field in ("git_commit", "native_extension_sha256", "uv_lock_sha256"):
                if provenance.get(field) != expected[field]:
                    raise InvalidTrialError(
                        f"provenance {field} changed during the study"
                    )
            if bool(provenance.get("git_dirty_at_start")) != bool(
                expected["git_dirty_at_spec"]
            ):
                raise InvalidTrialError("Git cleanliness changed during the study")
            manifest["provenance"] = provenance
        _write_json(self.condition_dir / "manifest.json", manifest)

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(dict(metrics), step=step)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.vec_env is not None:
                self.vec_env.close()
        finally:
            if self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass
            self.model = None
            self.vec_env = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _start_wandb_run(
    spec: StudySpec,
    trial_number: int,
    condition: str,
    effective_config: Mapping[str, Any],
):
    if not spec.wandb_config.get("enabled", True):
        return None
    kwargs = {
        "project": spec.wandb_config["project"],
        "group": spec.study_name,
        "name": f"{spec.study_name}-trial-{trial_number}-{condition}",
        "job_type": "optuna",
        "tags": ["pre-freeze", "optuna", f"trial-{trial_number}", condition],
        "config": {
            "protocol_status": PROTOCOL_STATUS,
            "study_name": spec.study_name,
            "trial_number": trial_number,
            "condition": condition,
            "training_seed": spec.training_seed,
            "evaluation_seeds": list(spec.evaluation_seeds),
            "study_config_sha256": spec.study_config_sha256,
            "sampled_hyperparameters": effective_config["tuning"][
                "sampled_hyperparameters"
            ],
            "effective_config": effective_config,
        },
    }
    mode = spec.wandb_config.get("mode")
    if mode:
        kwargs["mode"] = mode
    try:
        run = wandb.init(**kwargs)
        if run is None or not getattr(run, "id", None):
            raise InvalidTrialError("W&B initialization returned no run id")
        return run
    except Exception as exc:
        raise InvalidTrialError(f"W&B initialization failed: {exc}") from exc


def _runtime_provenance(model_path: Path, effective_config_path: Path) -> dict[str, Any]:
    runtime = collect_runtime_provenance(
        model_path=str(model_path.resolve()),
        effective_config=str(effective_config_path.resolve()),
    )
    git = training.collect_git_provenance()
    native = training.native_extension_identity()
    return {
        **git,
        **runtime,
        "native_extension_path": native["path"],
        "native_extension_sha256": native["sha256"],
    }


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def symmetric_log_objective(
    d4_depth1_mean_score: float, no_d4_depth1_mean_score: float
) -> float:
    """Compute the primary symmetric log-space score for one checkpoint."""
    scores = (d4_depth1_mean_score, no_d4_depth1_mean_score)
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float, np.number))
        or not math.isfinite(float(score))
        or float(score) < 0.0
        for score in scores
    ):
        raise InvalidTrialError("objective scores must be finite non-negative values")
    return float(np.mean([math.log1p(float(score)) for score in scores]))


def validate_checkpoint_metrics(
    metrics: CheckpointMetrics, *, critic_max_abs: float
) -> None:
    """Reject invalid, unresolved, or numerically pathological checkpoints."""
    if not isinstance(metrics, CheckpointMetrics):
        raise InvalidTrialError("checkpoint metrics have an invalid type")
    if metrics.actual_timesteps < metrics.checkpoint_timesteps:
        raise InvalidTrialError("actual timesteps did not reach checkpoint")
    for field in (
        "depth0_mean_score",
        "depth1_mean_score",
        "depth1_minus_depth0",
        "episode_length",
        "training_reward",
        *_CRITIC_METRIC_FIELDS,
    ):
        value = getattr(metrics, field)
        if not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
            raise InvalidTrialError(f"checkpoint metric {field} is non-finite")
    if metrics.depth0_mean_score < 0.0 or metrics.depth1_mean_score < 0.0:
        raise InvalidTrialError("checkpoint score means must be non-negative")
    if metrics.episode_length < 0.0 or metrics.value_std < 0.0:
        raise InvalidTrialError("checkpoint metric has an invalid negative value")
    if metrics.value_max_abs > critic_max_abs or metrics.value_std > critic_max_abs:
        raise InvalidTrialError("critic is pathological at checkpoint")
    for field in ("cap_count", "unresolved_count", "invalid_count"):
        value = getattr(metrics, field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise InvalidTrialError(f"checkpoint count {field} is invalid")
    if any(
        getattr(metrics, field) > 0
        for field in ("cap_count", "unresolved_count", "invalid_count")
    ):
        raise InvalidTrialError(
            "cap/unresolved/invalid counts make the checkpoint invalid"
        )


def _as_metric_mapping(value: Mapping[str, Any] | CheckpointMetrics) -> dict[str, Any]:
    if isinstance(value, CheckpointMetrics):
        return asdict(value)
    if not isinstance(value, Mapping):
        raise InvalidTrialError("metric provider returned a non-mapping")
    return dict(value)


def _combine_checkpoint_metrics(
    training_metrics: Mapping[str, Any] | CheckpointMetrics,
    evaluation_metrics: Mapping[str, Any],
    *,
    checkpoint_timesteps: int,
    actual_timesteps: int,
) -> CheckpointMetrics:
    train = _as_metric_mapping(training_metrics)
    evaluation = _as_metric_mapping(evaluation_metrics)
    depth0 = evaluation.get("depth0_mean_score", train.get("depth0_mean_score"))
    depth1 = evaluation.get("depth1_mean_score", train.get("depth1_mean_score"))
    return CheckpointMetrics(
        checkpoint_timesteps=checkpoint_timesteps,
        actual_timesteps=int(train.get("actual_timesteps", actual_timesteps)),
        depth0_mean_score=float(depth0),
        depth1_mean_score=float(depth1),
        depth1_minus_depth0=float(depth1) - float(depth0),
        explained_variance=float(train["explained_variance"]),
        value_std=float(train["value_std"]),
        value_max_abs=float(train["value_max_abs"]),
        episode_length=float(
            evaluation.get("episode_length", train["episode_length"])
        ),
        training_reward=float(train["training_reward"]),
        cap_count=int(
            evaluation.get("cap_count", train.get("cap_count", 0))
        ),
        unresolved_count=int(
            evaluation.get("unresolved_count", train.get("unresolved_count", 0))
        ),
        invalid_count=int(
            evaluation.get("invalid_count", train.get("invalid_count", 0))
        ),
    )


def evaluate_checkpoint(
    model_path: str | Path,
    condition: str,
    checkpoint_timesteps: int,
    evaluation_seeds: Sequence[int],
    evaluation_device: str,
) -> dict[str, Any]:
    """Evaluate a checkpoint through the real depth-0/depth-1 pipeline."""
    model_path = Path(model_path).resolve()
    validate_artifact_sha256(model_path, label="tuning checkpoint")
    if condition not in CONDITIONS:
        raise InvalidTrialError(f"unsupported evaluation condition: {condition}")

    aggregate: dict[str, Any] = {}
    for depth in (0, 1):
        benchmarker = None
        scores: list[float] = []
        lengths: list[float] = []
        cap_count = 0
        unresolved_count = 0
        try:
            benchmarker = Benchmarker(
                str(model_path),
                use_expectimax=depth == 1,
                search_depth=depth,
                device=evaluation_device,
            )
            if type(benchmarker.model) is not ValueHeadLRMaskablePPO:
                raise InvalidTrialError("evaluation loaded a non-v3 PPO model")
            if type(benchmarker.model.policy) is not ValueNormalizedMaskablePolicy:
                raise InvalidTrialError("evaluation loaded a non-v3 policy")
            if float(benchmarker.model.value_head_lr_multiplier) != 10.0:
                raise InvalidTrialError("evaluation loaded a non-v3 value-head multiplier")
            if type(benchmarker.value_evaluator) is not D4ValueEvaluator:
                raise InvalidTrialError("evaluation loaded a non-v3 D4 evaluator")
            for episode_idx, eval_seed in enumerate(evaluation_seeds):
                try:
                    result: EpisodeResult = benchmarker.run_episode(
                        eval_seed=int(eval_seed),
                        log_moves=False,
                        run_id=(
                            f"optuna-{condition}-trial-checkpoint-"
                            f"{checkpoint_timesteps}-d{depth}"
                        ),
                        episode_idx=episode_idx,
                    )
                except BenchmarkEpisodeError as exc:
                    raise InvalidTrialError(
                        f"evaluation {condition} depth{depth} failed: {exc.reason}"
                    ) from exc
                score = float(result.score)
                steps = float(result.steps)
                if not math.isfinite(score) or score < 0.0:
                    raise InvalidTrialError("evaluation produced an invalid score")
                if not math.isfinite(steps) or steps < 0.0:
                    raise InvalidTrialError("evaluation produced an invalid episode length")
                scores.append(score)
                lengths.append(steps)
                cap_count += int(result.total_cap_hits)
                unresolved_count += int(result.total_moves_unresolved)
            if not scores or not np.all(np.isfinite(scores + lengths)):
                raise InvalidTrialError("evaluation returned no finite metrics")
        except InvalidTrialError:
            raise
        except Exception as exc:
            raise InvalidTrialError(
                f"evaluation {condition} depth{depth} failed: {exc}"
            ) from exc
        finally:
            if benchmarker is not None:
                env = getattr(benchmarker, "env", None)
                if env is not None:
                    env.close()
            benchmarker = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cap_count or unresolved_count:
            raise InvalidTrialError(
                f"evaluation {condition} depth{depth} has cap/unresolved counts"
            )
        aggregate[f"depth{depth}_mean_score"] = float(np.mean(scores))
        if depth == 1:
            aggregate["episode_length"] = float(np.mean(lengths))
        aggregate["cap_count"] = aggregate.get("cap_count", 0) + cap_count
        aggregate["unresolved_count"] = (
            aggregate.get("unresolved_count", 0) + unresolved_count
        )
    aggregate["invalid_count"] = 0
    aggregate["evaluation_seeds"] = list(evaluation_seeds)
    return aggregate


def report_and_maybe_prune(trial, *, step: int, value: float) -> None:
    """Report only an evaluated objective and preserve an explicit prune reason."""
    if not math.isfinite(float(value)):
        raise InvalidTrialError("cannot report a non-finite objective")
    trial.report(float(value), step)
    if trial.should_prune():
        reason = f"checkpoint {step} pruned by MedianPruner"
        trial.set_user_attr("prune_reason", reason)
        raise TrialPruned(reason)


def raise_invalid_trial(reason: str) -> None:
    """Raise the explicit failure type used by Optuna's ``catch`` boundary."""
    raise InvalidTrialError(reason)


def trial_artifact_paths(
    storage_root: str | Path, *, study_name: str, trial_number: int
) -> dict[str, Path]:
    trial_dir = Path(storage_root) / study_name / f"trial_{trial_number}"
    return {
        "study": trial_dir.parent,
        "trial": trial_dir,
        "d4": trial_dir / "d4",
        "no_d4": trial_dir / "no_d4",
    }


def build_tuning_manifest(
    *,
    study_name: str,
    trial_number: int,
    condition: str,
    sampled_params: Mapping[str, Any],
    training_seed: int,
    evaluation_seeds: Sequence[int],
    effective_config: Mapping[str, Any],
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a PRE-FREEZE condition manifest with no paper-grade implication."""
    manifest = {
        "protocol_status": PROTOCOL_STATUS,
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "paper_grade": False,
        "study_name": study_name,
        "trial_number": trial_number,
        "condition": condition,
        "training_seed": training_seed,
        "evaluation_seeds": list(evaluation_seeds),
        "sampled_hyperparameters": dict(sampled_params),
        "effective_config": _json_ready(effective_config),
        "status": status,
    }
    manifest.update(_json_ready(extra))
    return manifest


def _write_trial_manifest(
    trial_dir: Path,
    spec: StudySpec,
    trial_number: int,
    sampled_params: Mapping[str, Any],
    *,
    status: str,
    checkpoint_metrics: Mapping[str, Mapping[int, CheckpointMetrics]],
    objective_by_checkpoint: Mapping[int, float],
    objective: float | None = None,
    reason_key: str | None = None,
    reason: str | None = None,
    final_model_paths: Mapping[str, Path] | None = None,
    wandb_run_ids: Mapping[str, str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "protocol_status": PROTOCOL_STATUS,
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "paper_grade": False,
        "study_name": spec.study_name,
        "trial_number": trial_number,
        "status": status,
        "training_seed": spec.training_seed,
        "evaluation_seeds": list(spec.evaluation_seeds),
        "study_config_sha256": spec.study_config_sha256,
        "base_config_sha256": spec.base_config_hashes,
        "sampled_hyperparameters": dict(sampled_params),
        "checkpoints": {
            condition: {
                str(step): asdict(metrics)
                for step, metrics in values.items()
            }
            for condition, values in checkpoint_metrics.items()
        },
        "objective_by_checkpoint": {
            str(step): value for step, value in objective_by_checkpoint.items()
        },
        "artifact_layout": {
            "trial": str(trial_dir.resolve()),
            "d4": str((trial_dir / "d4").resolve()),
            "no_d4": str((trial_dir / "no_d4").resolve()),
        },
    }
    if objective is not None:
        payload["objective"] = objective
    if wandb_run_ids:
        payload["wandb_run_ids"] = dict(wandb_run_ids)
    if reason_key and reason:
        payload[reason_key] = reason
    if final_model_paths:
        payload["final_models"] = {
            condition: {
                "path": str(path.resolve()),
                "sha256": validate_artifact_sha256(path, label="tuning final model"),
            }
            for condition, path in final_model_paths.items()
        }
    _write_json(trial_dir / "trial_manifest.json", payload)


SessionFactory = Callable[[str, StudySpec, Mapping[str, Any], Path, int], ConditionSession]
EvaluationFunction = Callable[[Path, str, int, Sequence[int], str], Mapping[str, Any]]


def _wandb_run_ids(sessions: Mapping[str, ConditionSession]) -> dict[str, str]:
    result = {}
    for condition, session in sessions.items():
        run = getattr(session, "wandb_run", None)
        run_id = getattr(run, "id", None)
        if run_id:
            result[condition] = str(run_id)
    return result


def _default_session_factory(
    condition: str,
    spec: StudySpec,
    sampled_params: Mapping[str, Any],
    condition_dir: Path,
    trial_number: int,
) -> ConditionTrainingSession:
    return ConditionTrainingSession(
        condition,
        spec,
        sampled_params,
        condition_dir,
        trial_number,
    )


def execute_trial(
    trial: optuna.Trial,
    spec: StudySpec,
    *,
    session_factory: SessionFactory | None = None,
    evaluation_fn: EvaluationFunction | None = None,
) -> float:
    """Run one paired trial, pruning only after both conditions are evaluated."""
    trial_number = int(getattr(trial, "number", 0))
    sampled_params = sample_hyperparameters(trial, spec)
    trial.set_user_attr("sampled_hyperparameters", sampled_params)
    trial.set_user_attr("training_seed", spec.training_seed)
    trial.set_user_attr("evaluation_seeds", list(spec.evaluation_seeds))

    paths = trial_artifact_paths(
        spec.storage_root, study_name=spec.study_name, trial_number=trial_number
    )
    if paths["trial"].exists() and any(paths["trial"].iterdir()):
        raise InvalidTrialError(
            f"tuning trial artifact directory is not empty: {paths['trial']}"
        )
    paths["trial"].mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        paths[condition].mkdir(parents=True, exist_ok=True)
    checkpoint_metrics: dict[str, dict[int, CheckpointMetrics]] = {
        condition: {} for condition in CONDITIONS
    }
    objective_by_checkpoint: dict[int, float] = {}
    sessions: dict[str, ConditionSession] = {}
    session_factory = session_factory or _default_session_factory
    evaluation_fn = evaluation_fn or evaluate_checkpoint
    _write_trial_manifest(
        paths["trial"],
        spec,
        trial_number,
        sampled_params,
        status="running",
        checkpoint_metrics=checkpoint_metrics,
        objective_by_checkpoint=objective_by_checkpoint,
    )
    try:
        for condition in CONDITIONS:
            sessions[condition] = session_factory(
                condition,
                spec,
                sampled_params,
                paths[condition],
                trial_number,
            )
        _write_trial_manifest(
            paths["trial"],
            spec,
            trial_number,
            sampled_params,
            status="running",
            checkpoint_metrics=checkpoint_metrics,
            objective_by_checkpoint=objective_by_checkpoint,
            wandb_run_ids=_wandb_run_ids(sessions),
        )

        for checkpoint in spec.checkpoints:
            for condition in CONDITIONS:
                session = sessions[condition]
                actual_timesteps = session.advance_to(checkpoint)
                model_path = session.checkpoint_path(checkpoint)
                if hasattr(session, "_write_manifest"):
                    # Bind the checkpoint to its effective config/native runtime
                    # before evaluation, so an evaluation failure still leaves
                    # inspectable provenance for the failed artifact.
                    session._write_manifest(
                        status="checkpoint_saved",
                        checkpoint_timesteps=checkpoint,
                        model_path=str(model_path),
                    )
                train_metrics = session.training_metrics(checkpoint)
                eval_metrics = evaluation_fn(
                    model_path,
                    condition,
                    checkpoint,
                    spec.evaluation_seeds,
                    spec.evaluation_device,
                )
                metrics = _combine_checkpoint_metrics(
                    train_metrics,
                    eval_metrics,
                    checkpoint_timesteps=checkpoint,
                    actual_timesteps=actual_timesteps,
                )
                validate_checkpoint_metrics(
                    metrics, critic_max_abs=spec.critic_max_abs
                )
                checkpoint_metrics[condition][checkpoint] = metrics
                _write_json(
                    paths[condition] / "metrics.json",
                    {
                        "protocol_status": PROTOCOL_STATUS,
                        "condition": condition,
                        "checkpoints": {
                            str(step): asdict(value)
                            for step, value in checkpoint_metrics[condition].items()
                        },
                    },
                )
                trial.set_user_attr(
                    f"{condition}_checkpoint_{checkpoint}", asdict(metrics)
                )
                if hasattr(session, "log"):
                    session.log(
                        {
                            "depth0_mean_score": metrics.depth0_mean_score,
                            "depth1_mean_score": metrics.depth1_mean_score,
                            "depth1_minus_depth0": metrics.depth1_minus_depth0,
                            "explained_variance": metrics.explained_variance,
                            "value_std": metrics.value_std,
                            "episode_length": metrics.episode_length,
                            "training_reward": metrics.training_reward,
                            "cap_count": metrics.cap_count,
                            "unresolved_count": metrics.unresolved_count,
                            "invalid_count": metrics.invalid_count,
                        },
                        step=checkpoint,
                    )
                if hasattr(session, "_write_manifest"):
                    session._write_manifest(
                        status="checkpoint_evaluated",
                        checkpoint_timesteps=checkpoint,
                        metrics=asdict(metrics),
                        model_path=str(model_path),
                    )

            objective_value = symmetric_log_objective(
                checkpoint_metrics["d4"][checkpoint].depth1_mean_score,
                checkpoint_metrics["no_d4"][checkpoint].depth1_mean_score,
            )
            objective_by_checkpoint[checkpoint] = objective_value
            trial.set_user_attr(
                f"objective_checkpoint_{checkpoint}", objective_value
            )
            _write_trial_manifest(
                paths["trial"],
                spec,
                trial_number,
                sampled_params,
                status="evaluated",
                checkpoint_metrics=checkpoint_metrics,
                objective_by_checkpoint=objective_by_checkpoint,
                wandb_run_ids=_wandb_run_ids(sessions),
            )
            report_and_maybe_prune(trial, step=checkpoint, value=objective_value)

        final_model_paths = {}
        for condition in CONDITIONS:
            session = sessions[condition]
            if hasattr(session, "final_model_path"):
                final_path = session.final_model_path()
                final_model_paths[condition] = final_path
                if hasattr(session, "_write_manifest"):
                    session._write_manifest(
                        status="completed",
                        objective=objective_by_checkpoint[spec.checkpoints[-1]],
                        model_path=str(final_path),
                    )
        final_objective = objective_by_checkpoint[spec.checkpoints[-1]]
        _write_trial_manifest(
            paths["trial"],
            spec,
            trial_number,
            sampled_params,
            status="completed",
            checkpoint_metrics=checkpoint_metrics,
            objective_by_checkpoint=objective_by_checkpoint,
            objective=final_objective,
            final_model_paths=final_model_paths,
            wandb_run_ids=_wandb_run_ids(sessions),
        )
        trial.set_user_attr("status", "completed")
        return final_objective
    except TrialPruned as exc:
        reason = str(exc) or "pruned after reproducible checkpoint evaluation"
        trial.set_user_attr("status", "pruned")
        _write_trial_manifest(
            paths["trial"],
            spec,
            trial_number,
            sampled_params,
            status="pruned",
            checkpoint_metrics=checkpoint_metrics,
            objective_by_checkpoint=objective_by_checkpoint,
            reason_key="prune_reason",
            reason=reason,
            wandb_run_ids=_wandb_run_ids(sessions),
        )
        raise
    except InvalidTrialError as exc:
        reason = str(exc)
        trial.set_user_attr("status", "failed")
        trial.set_user_attr("failure_reason", reason)
        _write_trial_manifest(
            paths["trial"],
            spec,
            trial_number,
            sampled_params,
            status="failed",
            checkpoint_metrics=checkpoint_metrics,
            objective_by_checkpoint=objective_by_checkpoint,
            reason_key="failure_reason",
            reason=reason,
            wandb_run_ids=_wandb_run_ids(sessions),
        )
        raise
    except Exception as exc:
        reason = f"unexpected trial failure: {exc}"
        trial.set_user_attr("status", "failed")
        trial.set_user_attr("failure_reason", reason)
        _write_trial_manifest(
            paths["trial"],
            spec,
            trial_number,
            sampled_params,
            status="failed",
            checkpoint_metrics=checkpoint_metrics,
            objective_by_checkpoint=objective_by_checkpoint,
            reason_key="failure_reason",
            reason=reason,
            wandb_run_ids=_wandb_run_ids(sessions),
        )
        # Expected runtime/provenance failures are failed Optuna trials, never
        # ordinary pruning. The study-level ``catch`` boundary can continue
        # with the next sample while retaining the failure reason above.
        raise InvalidTrialError(reason) from exc
    finally:
        for session in sessions.values():
            session.close()


def objective(trial: optuna.Trial, spec_or_config: StudySpec | Mapping[str, Any]) -> float:
    """Optuna objective entrypoint for one paired PRE-FREEZE trial."""
    if isinstance(spec_or_config, StudySpec):
        spec = spec_or_config
    else:
        config_path = spec_or_config.get("_config_path", REPO_ROOT / "configs" / "tune" / "bayesian_opt_search.yaml")
        spec = build_study_spec(spec_or_config, config_path=config_path)
    return execute_trial(trial, spec)


def dry_run_trial(spec: StudySpec, *, trial_number: int = 0) -> dict[str, Any]:
    """Materialize one baseline pair in memory without training or writing files."""
    sampled = default_sampled_params(spec)
    conceptual_paths = trial_artifact_paths(
        spec.storage_root, study_name=spec.study_name, trial_number=trial_number
    )
    d4_config = materialize_tuning_config(
        spec,
        sampled,
        condition="d4",
        output_dir=conceptual_paths["d4"],
        trial_number=trial_number,
    )
    no_d4_config = materialize_tuning_config(
        spec,
        sampled,
        condition="no_d4",
        output_dir=conceptual_paths["no_d4"],
        trial_number=trial_number,
    )
    if condition_behavioral_diff(d4_config, no_d4_config) != {
        "env_kwargs.d4_augment": (True, False)
    }:
        raise InvalidTrialError("dry-run pair has a behavioral difference beyond D4")
    return {
        "trial_number": trial_number,
        "sampled_hyperparameters": sampled,
        "d4": d4_config,
        "no_d4": no_d4_config,
    }


def render_dry_run(spec: StudySpec) -> str:
    """Render the execution contract without creating storage or a model."""
    dry_trial = dry_run_trial(spec)
    return "\n".join(
        (
            f"Protocol: {PROTOCOL_STATUS} ({TUNING_PROTOCOL_VERSION})",
            f"Study: {spec.study_name}",
            f"Trials: {spec.n_trials} trials; D4 + No-D4 per trial",
            f"Budget: {spec.trial_budget:,} timesteps per condition",
            f"Checkpoints: {', '.join(f'{step:,}' for step in spec.checkpoints)}",
            f"Training seed: {spec.training_seed}",
            f"Evaluation seeds: {', '.join(map(str, spec.evaluation_seeds))}",
            f"Dry-run trial: trial_{dry_trial['trial_number']} with shared baseline params",
            "Objective: mean(log1p(depth1_mean_score_D4), "
            "log1p(depth1_mean_score_NoD4))",
            f"Estimated cost: {spec.n_trials * 2} full training runs",
            "Estimated evaluation cost: "
            f"{spec.n_trials * len(spec.checkpoints) * 2 * len(spec.evaluation_seeds) * 2} "
            "episodes (depths 0 and 1)",
            f"Storage: {spec.db_path}",
            "No Optuna study, W&B run, training, or artifact is created by this dry-run.",
        )
    )


def _prepare_study(spec: StudySpec):
    if spec.db_path.exists() and not (spec.study_dir / "study_config.json").is_file():
        raise StudyConfigurationMismatch(
            "existing study storage has no immutable study_config.json"
        )
    digest = ensure_study_config(spec.study_dir, spec.study_config)
    _write_json(
        spec.study_dir / "study_manifest.json",
        {
            "protocol_status": PROTOCOL_STATUS,
            "protocol_version": TUNING_PROTOCOL_VERSION,
            "study_name": spec.study_name,
            "study_config_path": str((spec.study_dir / "study_config.json").resolve()),
            "study_config_sha256": digest,
            "storage_path": str(spec.db_path.resolve()),
            "training_seed": spec.training_seed,
            "evaluation_seeds": list(spec.evaluation_seeds),
            "provenance": spec.study_config["provenance"],
            "paper_grade": False,
        },
    )
    storage = f"sqlite:///{spec.db_path}"
    study = optuna.create_study(
        storage=storage,
        study_name=spec.study_name,
        load_if_exists=True,
        direction="maximize",
        sampler=make_sampler(spec),
        pruner=make_pruner(spec),
    )
    stored_digest = study.user_attrs.get("study_config_sha256")
    if stored_digest is not None and stored_digest != digest:
        raise StudyConfigurationMismatch(
            "Optuna study user attributes contain an incompatible config hash"
        )
    if stored_digest is None:
        if study.trials:
            raise StudyConfigurationMismatch(
                "existing Optuna study has trials but no immutable config hash"
            )
        study.set_user_attr("study_config_sha256", digest)
        study.set_user_attr("protocol_status", PROTOCOL_STATUS)
        study.set_user_attr("evaluation_seeds", list(spec.evaluation_seeds))
        study.set_user_attr("training_seed", spec.training_seed)
    return study


def run_study(config_path: str | Path, *, dry_run: bool = False):
    """Load the config and optionally execute the configured study."""
    config = load_tuning_yaml(config_path)
    spec = build_study_spec(config, config_path=config_path)
    if dry_run:
        print(render_dry_run(spec))
        return None

    study = _prepare_study(spec)
    remaining_trials = max(0, spec.n_trials - len(study.trials))
    try:
        if remaining_trials:
            study.optimize(
                lambda trial: objective(trial, spec),
                n_trials=remaining_trials,
                timeout=spec.timeout_seconds,
                n_jobs=1,
                catch=(InvalidTrialError,),
                gc_after_trial=True,
                show_progress_bar=False,
            )
        else:
            print("Study already contains the configured number of trials.")
    except KeyboardInterrupt:
        print("\n--- OPTUNA STUDY INTERRUPTED BY USER ---")

    print("\n--- OPTUNA STUDY COMPLETE ---")
    completed = [trial for trial in study.trials if trial.value is not None]
    if completed:
        best = study.best_trial
        print(f"Best trial for study '{study.study_name}':")
        print(f"  Value: {best.value}")
        print(f"  Params: {best.params}")
    else:
        print("  No completed trials.")
    return study


def main(config_path: str, *, dry_run: bool = False):
    return run_study(config_path, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the PRE-FREEZE v3 tuning YAML configuration.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print the study contract without running Optuna.",
    )
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
