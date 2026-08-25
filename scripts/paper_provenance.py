"""Shared validation for paper benchmark and training-manifest bindings."""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from scripts.benchmark_provenance import (
    REPO_ROOT,
    _native_extension_path,
    validate_artifact_sha256,
)


def validate_benchmark_training_binding(
    model_path: str | Path,
    training_manifest_path: str | Path,
    *,
    expected_train_seed: int | None = None,
    expected_effective_config_path: str | Path | None = None,
    recorded: Mapping | None = None,
    require_recorded: bool = False,
    require_paper_grade: bool = False,
) -> dict:
    """Validate a benchmark's model and metadata against its training manifest."""
    manifest_path = Path(training_manifest_path)
    if not manifest_path.is_absolute() or manifest_path != manifest_path.resolve():
        raise ValueError("training manifest path must be an absolute canonical path")
    manifest_sha256 = validate_artifact_sha256(
        manifest_path, label="training manifest",
    )

    # Import lazily because train.py imports this module's hashing helpers.
    from scripts.train import validate_training_manifest

    try:
        manifest = validate_training_manifest(manifest_path)
    except (KeyError, TypeError) as exc:
        raise ValueError("training manifest is malformed") from exc
    if require_paper_grade and not manifest["paper_grade"]:
        raise ValueError("training manifest is not paper-grade")
    benchmark_model_path = Path(model_path).resolve()
    manifest_model_path = Path(manifest["model_path"]).resolve()
    if benchmark_model_path != manifest_model_path:
        raise ValueError(
            "training manifest model_path does not match benchmark model_path"
        )

    effective_config_path = Path(manifest["effective_config_path"]).resolve()
    if expected_effective_config_path is not None:
        if Path(expected_effective_config_path).resolve() != effective_config_path:
            raise ValueError(
                "training manifest effective_config_path does not match benchmark config"
            )

    native_extension = manifest["native_extension"]
    native_extension_path = Path(native_extension["path"]).resolve()
    active_extension_path = _native_extension_path()
    if active_extension_path is None:
        raise ValueError("active native extension is unavailable")
    if native_extension_path != Path(active_extension_path).resolve():
        raise ValueError("training manifest native extension is not the active extension")
    uv_lock_path_value = manifest.get("uv_lock_path")
    if not isinstance(uv_lock_path_value, str) or not uv_lock_path_value:
        raise ValueError("training manifest uv_lock_path is missing")
    uv_lock_path = Path(uv_lock_path_value).resolve()

    model_sha256 = validate_artifact_sha256(
        benchmark_model_path,
        manifest["model_sha256"],
        label="model",
    )
    effective_config_sha256 = validate_artifact_sha256(
        effective_config_path,
        manifest["effective_config_sha256"],
        label="effective config",
    )
    native_extension_sha256 = validate_artifact_sha256(
        native_extension_path,
        manifest["native_extension_sha256"],
        label="native extension",
    )
    uv_lock_sha256 = validate_artifact_sha256(
        uv_lock_path,
        manifest["uv_lock_sha256"],
        label="uv.lock",
    )
    if uv_lock_path != (REPO_ROOT / "uv.lock").resolve():
        raise ValueError("training manifest uv_lock_path does not point to repository uv.lock")

    training_seed = manifest.get("training_seed")
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed not in (0, 1, 2, 3)
    ):
        raise ValueError("training manifest training_seed is invalid")
    if expected_train_seed is not None and expected_train_seed != training_seed:
        raise ValueError(
            "benchmark train_seed does not match training manifest training_seed"
        )

    condition = manifest.get("condition")
    d4_augment = manifest.get("d4_augment")
    if condition not in {"d4", "no_d4"} or type(d4_augment) is not bool:
        raise ValueError("training manifest condition is invalid")
    if condition != ("d4" if d4_augment else "no_d4"):
        raise ValueError("training manifest condition does not match d4_augment")

    binding = {
        "training_manifest_path": str(manifest_path),
        "training_manifest_sha256": manifest_sha256,
        "training_model_sha256": model_sha256,
        "training_seed": training_seed,
        "condition": condition,
        "d4_augment": d4_augment,
        "policy_class": manifest["policy_class"],
        "ppo_class": manifest["ppo_class"],
        "value_head_lr_multiplier": manifest["value_head_lr_multiplier"],
        "training_effective_config_sha256": effective_config_sha256,
        "final_timestep": manifest["final_timestep"],
        "training_git_commit": manifest["git_commit"],
        "training_native_extension_sha256": native_extension_sha256,
        "training_uv_lock_sha256": uv_lock_sha256,
    }

    if recorded is not None:
        if not isinstance(recorded, Mapping):
            raise ValueError("benchmark provenance must be a JSON object")
        observed = {
            "training_manifest_sha256": recorded.get("training_manifest_sha256"),
            "training_model_sha256": recorded.get("training_model_sha256"),
            "training_effective_config_sha256": recorded.get(
                "training_effective_config_sha256"
            ),
            "training_native_extension_sha256": recorded.get(
                "training_native_extension_sha256"
            ),
            "training_uv_lock_sha256": recorded.get("training_uv_lock_sha256"),
            "model_sha256": recorded.get("model_sha256"),
            "effective_config_sha256": recorded.get("effective_config_sha256"),
            "native_extension_sha256": recorded.get("native_extension_sha256"),
            "uv_lock_sha256": recorded.get("uv_lock_sha256"),
            "training_seed": recorded.get(
                "training_seed", recorded.get("train_seed")
            ),
            "condition": recorded.get("condition"),
            "d4_augment": recorded.get("d4_augment"),
            "policy_class": recorded.get("policy_class"),
            "ppo_class": recorded.get("ppo_class"),
            "value_head_lr_multiplier": recorded.get("value_head_lr_multiplier"),
            "final_timestep": recorded.get("final_timestep"),
            "training_git_commit": recorded.get("training_git_commit"),
        }
        expected = {
            "training_manifest_sha256": manifest_sha256,
            "training_model_sha256": model_sha256,
            "training_effective_config_sha256": effective_config_sha256,
            "training_native_extension_sha256": native_extension_sha256,
            "training_uv_lock_sha256": uv_lock_sha256,
            "model_sha256": model_sha256,
            "effective_config_sha256": effective_config_sha256,
            "native_extension_sha256": native_extension_sha256,
            "uv_lock_sha256": uv_lock_sha256,
            **{
                key: binding[key]
                for key in (
                    "training_seed",
                    "condition",
                    "d4_augment",
                    "policy_class",
                    "ppo_class",
                    "value_head_lr_multiplier",
                    "final_timestep",
                    "training_git_commit",
                )
            },
        }
        for field, expected_value in expected.items():
            if observed[field] is None and not require_recorded:
                continue
            observed_value = observed[field]
            if isinstance(expected_value, bool):
                valid_type = type(observed_value) is bool
            elif isinstance(expected_value, int):
                valid_type = type(observed_value) is int
            elif isinstance(expected_value, float):
                valid_type = (
                    type(observed_value) in (int, float)
                    and math.isfinite(float(observed_value))
                )
            elif isinstance(expected_value, str):
                valid_type = isinstance(observed_value, str)
            else:
                valid_type = True
            if not valid_type or observed_value != expected_value:
                raise ValueError(
                    f"benchmark provenance {field} does not match training manifest"
                )

    return binding
