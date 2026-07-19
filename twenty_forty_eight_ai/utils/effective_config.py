"""Resolved training configuration helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


D4_SEED_DERIVATION = "numpy.random.SeedSequence(seed).spawn(n_envs)"
_IDENTIFIER_ONLY_KEYS = {"run_name"}


def derive_d4_rank_seed_sequences(
    training_seed: int, n_envs: int
) -> list[np.random.SeedSequence]:
    """Derive one deterministic, independent D4 SeedSequence per environment rank."""
    return np.random.SeedSequence(training_seed).spawn(n_envs)


def materialize_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a training config and make its training-only D4 default explicit."""
    effective_config = deepcopy(dict(config))
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
