from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


def _v3_definition():
    return {
        "name": "v3",
        "policy_class": (
            "twenty_forty_eight_ai.agent.policy.ValueNormalizedMaskablePolicy"
        ),
        "ppo_class": (
            "twenty_forty_eight_ai.agent.ppo.ValueHeadLRMaskablePPO"
        ),
        "value_head_lr_multiplier": 10.0,
    }


def _git_snapshot(commit: str, status: str) -> dict[str, str | bool]:
    dirty = bool(status)
    return {
        "git_commit": commit,
        "git_commit_at_start": commit,
        "git_status_porcelain": status,
        "git_status_at_start": status,
        "git_dirty": dirty,
        "git_dirty_at_start": dirty,
    }


def _make_v3_model():
    from stable_baselines3.common.vec_env import DummyVecEnv

    from twenty_forty_eight_ai.agent.architecture import CustomCNN
    from twenty_forty_eight_ai.agent.policy import ValueNormalizedMaskablePolicy
    from twenty_forty_eight_ai.agent.ppo import ValueHeadLRMaskablePPO
    from twenty_forty_eight_ai.env.environment import Game2048Env

    env = DummyVecEnv([lambda: Game2048Env()])
    model = ValueHeadLRMaskablePPO(
        ValueNormalizedMaskablePolicy,
        env,
        policy_kwargs={
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 8},
        },
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        device="cpu",
        verbose=0,
        value_head_lr_multiplier=10.0,
    )
    return model


def test_git_provenance_uses_porcelain_status(monkeypatch):
    from scripts import train as train_module

    calls = []

    def fake_check_output(command, **kwargs):
        calls.append(command)
        if command[1] == "rev-parse":
            return "a" * 40
        return " M changed.py\n"

    monkeypatch.setattr(train_module.subprocess, "check_output", fake_check_output)

    assert train_module.collect_git_provenance() == _git_snapshot(
        "a" * 40, " M changed.py\n",
    )
    assert ["git", "status", "--porcelain"] in calls


def test_untracked_unrelated_file_is_dirty_at_run_start(monkeypatch):
    from scripts import train as train_module

    def fake_check_output(command, **kwargs):
        if command[1] == "rev-parse":
            return "a" * 40
        return "?? unrelated.txt\n"

    monkeypatch.setattr(train_module.subprocess, "check_output", fake_check_output)

    snapshot = train_module.collect_git_provenance()

    assert snapshot["git_commit_at_start"] == "a" * 40
    assert snapshot["git_status_at_start"] == "?? unrelated.txt\n"
    assert snapshot["git_dirty_at_start"] is True


def test_training_manifest_records_complete_v3_provenance(tmp_path, monkeypatch):
    from scripts import train as train_module

    model_dir = Path(tmp_path)
    model_path = model_dir / "final_model.zip"
    native_path = model_dir / "native.so"
    native_path.write_bytes(b"native")
    effective_config = {
        "run_name": "hybrid_ppo_v3",
        "seed": 2,
        "training_seeds": [0, 1, 2, 3],
        "root_training_seed": 2,
        "total_timesteps": 200_000_000,
        "load_model": False,
        "checkpoint_path": None,
        "env_kwargs": {"d4_augment": True},
        "experiment_definition": _v3_definition(),
    }
    train_module.persist_effective_config(model_dir, effective_config)

    start_git_snapshot = _git_snapshot("b" * 40, "")
    monkeypatch.setattr(
        train_module,
        "collect_git_provenance",
        lambda: pytest.fail("manifest must use the pre-output Git snapshot"),
    )
    monkeypatch.setattr(
        train_module,
        "collect_runtime_provenance",
        lambda **_: {
            "effective_config_sha256": train_module.sha256_file(
                model_dir / "effective_config.json"
            ),
            "uv_lock_sha256": train_module.sha256_file(
                train_module.REPO_ROOT / "uv.lock"
            ),
            "native_extension_sha256": train_module.sha256_file(native_path),
            "model_sha256": train_module.sha256_file(model_path),
            "python_version": "3.12.0",
            "torch_version": "2.9.1",
            "sb3_version": "2.8.0",
            "cuda_runtime": "13.0",
            "gpu_name": "GPU",
            "compiler": "gcc",
        },
    )
    monkeypatch.setattr(
        train_module,
        "native_extension_identity",
        lambda: {
            "path": str(native_path),
            "sha256": train_module.sha256_file(native_path),
        },
    )
    monkeypatch.setattr(
        train_module,
        "relevant_package_versions",
        lambda: {
            "gymnasium": "1.2.0",
            "numpy": "1.26.4",
            "numba": "0.62.1",
            "sb3-contrib": "2.8.0",
            "pybind11": "3.0.1",
        },
    )

    model = _make_v3_model()
    try:
        model.num_timesteps = 200_000_000
        model.save(model_path)
        manifest_path = train_module.persist_training_manifest(
            model_dir,
            str(model_path),
            model,
            effective_config,
            git_provenance=start_git_snapshot,
        )
    finally:
        model.get_env().close()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["git_commit"] == "b" * 40
    assert manifest["git_commit_at_start"] == "b" * 40
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_at_start"] is False
    assert manifest["git_status_porcelain"] == ""
    assert manifest["git_status_at_start"] == ""
    assert manifest["effective_config"] == effective_config
    assert manifest["training_seed"] == 2
    assert manifest["d4_condition"] == "d4"
    assert manifest["model_path"] == str(model_path.resolve())
    assert manifest["model_sha256"] == train_module.sha256_file(model_path)
    assert manifest["effective_config_sha256"] == train_module.sha256_file(
        model_dir / "effective_config.json"
    )
    assert manifest["policy_class"].endswith("ValueNormalizedMaskablePolicy")
    assert manifest["ppo_class"].endswith("ValueHeadLRMaskablePPO")
    assert manifest["value_head_lr_multiplier"] == 10.0
    assert manifest["fresh_training"] is True
    assert manifest["load_model"] is False
    assert manifest["checkpoint_path"] is None
    assert manifest["paper_grade"] is True
    assert manifest["versions"]["gymnasium"] == "1.2.0"
    assert manifest["native_extension"] == {
        "path": str(native_path.resolve()),
        "sha256": train_module.sha256_file(native_path),
    }
    assert manifest["final_timestep"] == 200_000_000

    train_module.validate_training_manifest(manifest_path)
    missing_start_snapshot = dict(manifest)
    missing_start_snapshot.pop("git_commit_at_start")
    missing_start_snapshot.pop("git_status_at_start")
    missing_start_snapshot.pop("git_dirty_at_start")
    manifest_path.write_text(json.dumps(missing_start_snapshot))
    with pytest.raises(ValueError, match="git_commit_at_start"):
        train_module.validate_training_manifest(manifest_path)

    inconsistent_commit = dict(manifest)
    inconsistent_commit["git_commit_at_start"] = "c" * 40
    manifest_path.write_text(json.dumps(inconsistent_commit))
    with pytest.raises(ValueError, match="commit"):
        train_module.validate_training_manifest(manifest_path)

    invalid_timestep = dict(manifest)
    invalid_timestep["final_timestep"] = 0
    manifest_path.write_text(json.dumps(invalid_timestep))
    with pytest.raises(ValueError, match="final_timestep"):
        train_module.validate_training_manifest(manifest_path)

    other_dir = model_dir / "other"
    other_dir.mkdir()
    shutil.copy2(model_path, other_dir / "final_model.zip")
    shutil.copy2(model_dir / "effective_config.json", other_dir / "effective_config.json")
    cross_directory = dict(manifest)
    cross_directory["model_path"] = str((other_dir / "final_model.zip").resolve())
    cross_directory["effective_config_path"] = str(
        (other_dir / "effective_config.json").resolve()
    )
    manifest_path.write_text(json.dumps(cross_directory))
    with pytest.raises(ValueError, match="share a directory"):
        train_module.validate_training_manifest(manifest_path)


@pytest.mark.parametrize(
    ("changed_artifact", "error"),
    [
        ("model", "model_sha256"),
        ("effective_config", "effective_config_sha256"),
        ("native", "native_extension_sha256"),
    ],
)
def test_training_manifest_validator_rejects_changed_artifact(
    tmp_path, monkeypatch, changed_artifact, error
):
    from scripts import train as train_module

    model_dir = Path(tmp_path)
    model_path = model_dir / "final_model.zip"
    native_path = model_dir / "native.so"
    native_path.write_bytes(b"native")
    effective_config = {
        "run_name": "hybrid_ppo_v3_no_d4",
        "seed": 0,
        "training_seeds": [0, 1, 2, 3],
        "root_training_seed": 0,
        "total_timesteps": 200_000_000,
        "load_model": False,
        "checkpoint_path": None,
        "env_kwargs": {"d4_augment": False},
        "experiment_definition": _v3_definition(),
    }
    train_module.persist_effective_config(model_dir, effective_config)
    monkeypatch.setattr(
        train_module,
        "collect_runtime_provenance",
        lambda **_: {
            "effective_config_sha256": train_module.sha256_file(
                model_dir / "effective_config.json"
            ),
            "uv_lock_sha256": train_module.sha256_file(
                train_module.REPO_ROOT / "uv.lock"
            ),
            "native_extension_sha256": train_module.sha256_file(native_path),
            "model_sha256": train_module.sha256_file(model_path),
            "python_version": "3.12.0",
            "torch_version": "2.9.1",
            "sb3_version": "2.8.0",
            "cuda_runtime": "",
            "gpu_name": "",
            "compiler": "gcc",
        },
    )
    monkeypatch.setattr(
        train_module,
        "native_extension_identity",
        lambda: {
            "path": str(native_path),
            "sha256": train_module.sha256_file(native_path),
        },
    )
    monkeypatch.setattr(
        train_module,
        "relevant_package_versions",
        lambda: {
            "gymnasium": "1.2.0",
            "numpy": "1.26.4",
            "numba": "0.62.1",
            "sb3-contrib": "2.8.0",
            "pybind11": "3.0.1",
        },
    )
    model = _make_v3_model()
    try:
        model.save(model_path)
        manifest_path = train_module.persist_training_manifest(
            model_dir,
            str(model_path),
            model,
            effective_config,
            git_provenance=_git_snapshot("a" * 40, ""),
        )
    finally:
        model.get_env().close()

    if changed_artifact == "model":
        model_path.write_bytes(b"changed")
    elif changed_artifact == "effective_config":
        (model_dir / "effective_config.json").write_text("{}")
    else:
        native_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match=error):
        train_module.validate_training_manifest(manifest_path)


def test_training_manifests_record_each_seed(tmp_path, monkeypatch):
    from scripts import train as train_module

    native_path = Path(tmp_path) / "native.so"
    native_path.write_bytes(b"native")
    model = _make_v3_model()
    try:
        for seed in range(4):
            model_dir = Path(tmp_path) / f"seed{seed}"
            model_dir.mkdir()
            model_path = model_dir / "final_model.zip"
            effective_config = {
                "run_name": "hybrid_ppo_v3",
                "seed": seed,
                "training_seeds": [0, 1, 2, 3],
                "root_training_seed": seed,
                "total_timesteps": 200_000_000,
                "load_model": False,
                "checkpoint_path": None,
                "env_kwargs": {"d4_augment": True},
                "experiment_definition": _v3_definition(),
            }
            train_module.persist_effective_config(model_dir, effective_config)
            model.num_timesteps = 200_000_000
            model.save(model_path)
            monkeypatch.setattr(
                train_module,
                "collect_runtime_provenance",
                lambda *, model_path, effective_config: {
                    "effective_config_sha256": train_module.sha256_file(
                        effective_config
                    ),
                    "uv_lock_sha256": train_module.sha256_file(
                        train_module.REPO_ROOT / "uv.lock"
                    ),
                    "native_extension_sha256": train_module.sha256_file(native_path),
                    "model_sha256": train_module.sha256_file(model_path),
                    "python_version": "3.12.0",
                    "torch_version": "2.9.1",
                    "sb3_version": "2.8.0",
                    "cuda_runtime": "",
                    "gpu_name": "",
                    "compiler": "gcc",
                },
            )
            monkeypatch.setattr(
                train_module,
                "native_extension_identity",
                lambda: {
                    "path": str(native_path),
                    "sha256": train_module.sha256_file(native_path),
                },
            )
            monkeypatch.setattr(
                train_module,
                "relevant_package_versions",
                lambda: {
                    "gymnasium": "1.2.0",
                    "numpy": "1.26.4",
                    "numba": "0.62.1",
                    "sb3-contrib": "2.8.0",
                    "pybind11": "3.0.1",
                },
            )

            manifest_path = train_module.persist_training_manifest(
                model_dir,
                str(model_path),
                model,
                effective_config,
                git_provenance=_git_snapshot("c" * 40, ""),
            )
            manifest = json.loads(manifest_path.read_text())
            assert manifest["training_seed"] == seed
            assert manifest["root_training_seed"] == seed
    finally:
        model.get_env().close()


def test_dirty_training_manifest_is_marked_non_paper_grade(tmp_path, monkeypatch):
    from scripts import train as train_module

    model_dir = Path(tmp_path)
    model_path = model_dir / "final_model.zip"
    native_path = model_dir / "native.so"
    native_path.write_bytes(b"native")
    effective_config = {
        "run_name": "hybrid_ppo_v3",
        "seed": 1,
        "training_seeds": [0, 1, 2, 3],
        "root_training_seed": 1,
        "total_timesteps": 200_000_000,
        "load_model": False,
        "checkpoint_path": None,
        "env_kwargs": {"d4_augment": True},
        "experiment_definition": _v3_definition(),
    }
    train_module.persist_effective_config(model_dir, effective_config)
    start_git_snapshot = _git_snapshot("b" * 40, " M source.py\n")
    monkeypatch.setattr(
        train_module,
        "collect_git_provenance",
        lambda: pytest.fail("manifest must use the pre-output Git snapshot"),
    )
    monkeypatch.setattr(
        train_module,
        "collect_runtime_provenance",
        lambda **_: {
            "effective_config_sha256": train_module.sha256_file(
                model_dir / "effective_config.json"
            ),
            "uv_lock_sha256": train_module.sha256_file(
                train_module.REPO_ROOT / "uv.lock"
            ),
            "native_extension_sha256": train_module.sha256_file(native_path),
            "model_sha256": train_module.sha256_file(model_path),
            "python_version": "3.12.0",
            "torch_version": "2.9.1",
            "sb3_version": "2.8.0",
            "cuda_runtime": "",
            "gpu_name": "",
            "compiler": "gcc",
        },
    )
    monkeypatch.setattr(
        train_module,
        "native_extension_identity",
        lambda: {
            "path": str(native_path),
            "sha256": train_module.sha256_file(native_path),
        },
    )
    monkeypatch.setattr(
        train_module,
        "relevant_package_versions",
        lambda: {
            "gymnasium": "1.2.0",
            "numpy": "1.26.4",
            "numba": "0.62.1",
            "sb3-contrib": "2.8.0",
            "pybind11": "3.0.1",
        },
    )
    model = _make_v3_model()
    try:
        model.num_timesteps = 200_000_000
        model.save(model_path)
        manifest_path = train_module.persist_training_manifest(
            model_dir,
            str(model_path),
            model,
            effective_config,
            git_provenance=start_git_snapshot,
        )
    finally:
        model.get_env().close()

    manifest = train_module.validate_training_manifest(manifest_path)
    assert manifest["git_dirty"] is True
    assert manifest["git_dirty_at_start"] is True
    assert manifest["git_status_at_start"] == " M source.py\n"
    assert manifest["paper_grade"] is False

    manifest["git_status_porcelain"] = ""
    manifest["git_status_at_start"] = ""
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="git_dirty"):
        train_module.validate_training_manifest(manifest_path)

    manifest["git_status_porcelain"] = " M source.py\n"
    manifest["git_status_at_start"] = " M source.py\n"
    manifest.pop("final_timestep")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="final_timestep"):
        train_module.validate_training_manifest(manifest_path)

    manifest["final_timestep"] = 0
    manifest["root_training_seed"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="root training seed"):
        train_module.validate_training_manifest(manifest_path)

    manifest["root_training_seed"] = 1
    manifest["condition"] = "no_d4"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="condition"):
        train_module.validate_training_manifest(manifest_path)
