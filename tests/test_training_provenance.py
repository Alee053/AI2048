from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_git_provenance_uses_porcelain_status(monkeypatch):
    from scripts import train as train_module

    calls = []

    def fake_check_output(command, **kwargs):
        calls.append(command)
        if command[1] == "rev-parse":
            return "a" * 40
        return " M changed.py\n"

    monkeypatch.setattr(train_module.subprocess, "check_output", fake_check_output)

    assert train_module.collect_git_provenance() == {
        "git_commit": "a" * 40,
        "git_status_porcelain": " M changed.py\n",
        "git_dirty": True,
    }
    assert ["git", "status", "--porcelain"] in calls


def test_training_manifest_records_complete_v3_provenance(tmp_path, monkeypatch):
    from scripts import train as train_module

    model_dir = Path(tmp_path)
    model_path = model_dir / "final_model.zip"
    model_path.write_bytes(b"model")
    effective_config = {
        "run_name": "hybrid_ppo_v3",
        "seed": 2,
        "training_seeds": [0, 1, 2, 3],
        "root_training_seed": 2,
        "env_kwargs": {"d4_augment": True},
    }
    train_module.persist_effective_config(model_dir, effective_config)

    monkeypatch.setattr(
        train_module,
        "collect_git_provenance",
        lambda: {
            "git_commit": "b" * 40,
            "git_status_porcelain": "",
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(
        train_module,
        "collect_runtime_provenance",
        lambda **_: {
            "effective_config_sha256": "config-hash",
            "uv_lock_sha256": "lock-hash",
            "native_extension_sha256": "extension-hash",
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
            "path": "/repo/searcher.so",
            "sha256": "extension-hash",
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
        },
    )

    manifest_path = train_module.persist_training_manifest(
        model_dir,
        str(model_path),
        SimpleNamespace(num_timesteps=123),
        effective_config,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["git_commit"] == "b" * 40
    assert manifest["git_dirty"] is False
    assert manifest["git_status_porcelain"] == ""
    assert manifest["effective_config"] == effective_config
    assert manifest["training_seed"] == 2
    assert manifest["d4_condition"] == "d4"
    assert manifest["versions"]["gymnasium"] == "1.2.0"
    assert manifest["native_extension"] == {
        "path": "/repo/searcher.so",
        "sha256": "extension-hash",
    }
    assert manifest["final_timestep"] == 123
