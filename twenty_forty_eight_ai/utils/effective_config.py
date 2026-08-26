"""Resolved training configuration helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


D4_SEED_DERIVATION = "numpy.random.SeedSequence(seed).spawn(n_envs)"
V3_EXPERIMENT_DEFINITION = {
    "name": "v3",
    "policy_class": (
        "twenty_forty_eight_ai.agent.policy.ValueNormalizedMaskablePolicy"
    ),
    "ppo_class": "twenty_forty_eight_ai.agent.ppo.ValueHeadLRMaskablePPO",
    "value_head_lr_multiplier": 10.0,
}
V3_TOTAL_TIMESTEPS = 100_000_000
V3_TRAINING_SEEDS = [0, 1, 2, 3]
_IDENTIFIER_ONLY_KEYS = {"run_name"}
_V3_RUN_NAME_PREFIXES = ("hybrid_ppo_v3", "hybrid_ppo_v3_no_d4")


def is_reserved_v3_run_name(run_name: Any) -> bool:
    """Return whether a run name belongs to the frozen v3 namespace."""
    if not isinstance(run_name, str):
        return False
    return any(
        run_name == prefix
        or run_name.startswith(f"{prefix}-seed")
        for prefix in _V3_RUN_NAME_PREFIXES
    )


def derive_d4_rank_seed_sequences(
    training_seed: int, n_envs: int
) -> list[np.random.SeedSequence]:
    """Derive one deterministic, independent D4 SeedSequence per environment rank."""
    return np.random.SeedSequence(training_seed).spawn(n_envs)


def validate_v3_experiment_config(config: Mapping[str, Any]) -> None:
    """Reject invalid v3 definitions, conditions, resumes, or seed sets."""
    if "experiment_definition" in config and not isinstance(
        config["experiment_definition"], Mapping
    ):
        raise ValueError("experiment_definition must be a mapping")
    definition = config.get("experiment_definition")
    if definition is None:
        if is_reserved_v3_run_name(config.get("run_name")):
            raise ValueError("v3 config must explicitly define experiment_definition")
        return
    if definition.get("name") != "v3":
        if is_reserved_v3_run_name(config.get("run_name")):
            raise ValueError("reserved v3 run name has an unsupported experiment definition")
        raise ValueError("unsupported experiment definition")

    if dict(definition) != V3_EXPERIMENT_DEFINITION:
        raise ValueError("v3 experiment_definition must match the frozen v3 definition")
    if (
        type(config.get("total_timesteps")) is not int
        or config["total_timesteps"] != V3_TOTAL_TIMESTEPS
    ):
        raise ValueError(
            "v3 total_timesteps must equal "
            f"{V3_TOTAL_TIMESTEPS}"
        )

    env_kwargs = config.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping) or "d4_augment" not in env_kwargs:
        raise ValueError("v3 config must explicitly define env_kwargs.d4_augment")
    d4_augment = env_kwargs["d4_augment"]
    if type(d4_augment) is not bool:
        raise ValueError("v3 config env_kwargs.d4_augment must be a boolean")
    if (
        config.get("load_model") is not False
        or "checkpoint_path" not in config
        or config["checkpoint_path"] is not None
    ):
        raise ValueError("v3 experiment must start from scratch with load_model=false")

    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in V3_TRAINING_SEEDS:
        raise ValueError(f"v3 config seed must be one of {V3_TRAINING_SEEDS}")
    if config.get("training_seeds") != V3_TRAINING_SEEDS:
        raise ValueError(
            f"v3 config training_seeds must equal {V3_TRAINING_SEEDS}"
        )


def materialize_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a training config and make its training-only D4 default explicit."""
    effective_config = deepcopy(dict(config))
    validate_v3_experiment_config(effective_config)
    env_kwargs = dict(effective_config.get("env_kwargs", {}))
    env_kwargs.setdefault("d4_augment", True)
    effective_config["env_kwargs"] = env_kwargs
    return effective_config


def load_effective_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML training config with training defaults materialized."""
    with Path(path).open() as config_file:
        config = yaml.safe_load(config_file)
    return materialize_training_config(config)


def compare_behavioral_configs(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return behavioral differences, excluding identifier-only configuration keys."""
    first_values = _flatten_behavioral_config(first)
    second_values = _flatten_behavioral_config(second)
    missing = object()
    return {
        key: (first_values.get(key, missing), second_values.get(key, missing))
        for key in sorted(first_values.keys() | second_values.keys())
        if first_values.get(key, missing) != second_values.get(key, missing)
    }


def _flatten_behavioral_config(
    config: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if key in _IDENTIFIER_ONLY_KEYS:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_behavioral_config(value, path))
        else:
            flattened[path] = value
    return flattened
