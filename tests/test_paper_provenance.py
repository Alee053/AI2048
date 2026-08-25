from __future__ import annotations

import csv
import json
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


def _make_model():
    from stable_baselines3.common.vec_env import DummyVecEnv

    from twenty_forty_eight_ai.agent.architecture import CustomCNN
    from twenty_forty_eight_ai.agent.policy import ValueNormalizedMaskablePolicy
    from twenty_forty_eight_ai.agent.ppo import ValueHeadLRMaskablePPO
    from twenty_forty_eight_ai.env.environment import Game2048Env

    env = DummyVecEnv([lambda: Game2048Env()])
    return ValueHeadLRMaskablePPO(
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


@pytest.fixture
def valid_artifacts(tmp_path, monkeypatch):
    from scripts import train as train_module

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "final_model.zip"
    effective_config = {
        "run_name": "hybrid_ppo_v3",
        "seed": 0,
        "training_seeds": [0, 1, 2, 3],
        "root_training_seed": 0,
        "load_model": False,
        "checkpoint_path": None,
        "env_kwargs": {"d4_augment": True},
        "experiment_definition": _v3_definition(),
    }
    train_module.persist_effective_config(model_dir, effective_config)
    monkeypatch.setattr(
        train_module,
        "collect_git_provenance",
        lambda: {
            "git_commit": "a" * 40,
            "git_status_porcelain": "",
            "git_dirty": False,
        },
    )
    model = _make_model()
    try:
        model.num_timesteps = 123
        model.save(model_path)
        manifest_path = train_module.persist_training_manifest(
            model_dir, str(model_path), model, effective_config,
        )
    finally:
        model.get_env().close()
    return {
        "model_path": model_path,
        "manifest_path": manifest_path,
        "effective_config_path": model_dir / "effective_config.json",
    }


def test_valid_training_binding_recomputes_all_artifact_hashes(valid_artifacts):
    from scripts.paper_provenance import validate_benchmark_training_binding

    binding = validate_benchmark_training_binding(
        valid_artifacts["model_path"],
        valid_artifacts["manifest_path"],
        expected_train_seed=0,
        expected_effective_config_path=valid_artifacts["effective_config_path"],
    )

    assert binding["condition"] == "d4"
    assert binding["training_seed"] == 0
    assert binding["final_timestep"] == 123
    assert len(binding["training_manifest_sha256"]) == 64
    assert binding["training_model_sha256"]
    assert binding["training_native_extension_sha256"]
    assert binding["training_uv_lock_sha256"]


def test_paper_benchmark_derives_binding_from_manifest(valid_artifacts, monkeypatch):
    from scripts import benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(benchmark_module, "_git_dirty", lambda: False)
    args = benchmark_module.parse_args([
        str(valid_artifacts["model_path"]),
        "--paper-mode",
        "--train-seed", "0",
        "--base-eval-seed", "100",
        "--effective-config", str(valid_artifacts["effective_config_path"]),
        "--training-manifest", str(valid_artifacts["manifest_path"]),
    ])

    benchmark_module.validate_paper_mode(args)

    assert args._paper_provenance["training_seed"] == 0
    assert args._paper_provenance["condition"] == "d4"
    assert args._paper_provenance["training_git_commit"] == "a" * 40


def test_normal_benchmark_config_records_runtime_identity_without_manifest(
    valid_artifacts, monkeypatch,
):
    from scripts import benchmark as benchmark_module

    valid_artifacts["manifest_path"].unlink()
    monkeypatch.setattr(benchmark_module, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(benchmark_module, "_git_dirty", lambda: False)
    args = benchmark_module.parse_args([
        str(valid_artifacts["model_path"]),
        "--effective-config", str(valid_artifacts["effective_config_path"]),
        "--train-seed", "0", "--base-eval-seed", "100", "--sweep-name", "normal",
    ])

    config = benchmark_module.build_config(
        args, "run", 100, "deterministic-offset", "2026-01-01T00:00:00Z",
    )

    assert config["model_path"] == str(valid_artifacts["model_path"].resolve())
    assert config["effective_config_path"] == str(
        valid_artifacts["effective_config_path"].resolve()
    )
    assert config["condition"] == "d4"
    assert config["training_seed"] == 0
    assert config["base_eval_seed"] == 100


def _write_valid_paper_run(run_dir, valid_artifacts):
    from scripts.benchmark_io import EPISODE_COLUMNS, outcome_fingerprint
    from scripts.benchmark_provenance import sha256_file
    from scripts.benchmark_summary import compute_summary_from_rows
    from scripts.paper_provenance import validate_benchmark_training_binding

    run_dir.mkdir()
    manifest = json.loads(valid_artifacts["manifest_path"].read_text())
    binding = validate_benchmark_training_binding(
        valid_artifacts["model_path"], valid_artifacts["manifest_path"],
    )
    row = {column: 0 for column in EPISODE_COLUMNS}
    row.update({
        "schema_version": "2.1.0",
        "run_id": "run",
        "episode_idx": 0,
        "worker_id": 0,
        "train_seed": 0,
        "eval_seed": 100,
        "requested_depth": 0,
        "effective_depth": 0,
        "use_expectimax": False,
        "termination_reason": "board_full",
        "mean_chance_value": 0.0,
        "mean_empty_cells": 8.0,
        "mean_merge_score": 0.0,
        "mean_nps": 0.0,
        "mean_tt_hit_rate": 0.0,
        "mean_nodes_per_batch_call": 0.0,
        "win_1024": False,
        "win_2048": False,
        "win_4096": False,
        "win_8192": False,
    })
    config = {
        "benchmark_schema_version": "2.1.0",
        "sweep_name": "paper-v3",
        "run_name": "run",
        "run_id": "run",
        "model_path": str(valid_artifacts["model_path"]),
        "model_sha256": sha256_file(valid_artifacts["model_path"]),
        "effective_config_path": str(valid_artifacts["effective_config_path"]),
        "effective_config_sha256": sha256_file(valid_artifacts["effective_config_path"]),
        "training_manifest_path": str(valid_artifacts["manifest_path"]),
        "training_manifest_sha256": binding["training_manifest_sha256"],
        "training_model_sha256": binding["training_model_sha256"],
        "training_seed": 0,
        "train_seed": 0,
        "condition": "d4",
        "d4_augment": True,
        "policy_class": manifest["policy_class"],
        "ppo_class": manifest["ppo_class"],
        "value_head_lr_multiplier": 10.0,
        "training_effective_config_sha256": binding["training_effective_config_sha256"],
        "final_timestep": 123,
        "training_git_commit": manifest["git_commit"],
        "training_native_extension_sha256": binding["training_native_extension_sha256"],
        "training_uv_lock_sha256": binding["training_uv_lock_sha256"],
        "native_extension_sha256": binding["training_native_extension_sha256"],
        "uv_lock_sha256": binding["training_uv_lock_sha256"],
        "git_commit": "b" * 40,
        "git_dirty": False,
        "paper_grade": True,
        "paper_mode": True,
        "status": "completed",
        "interrupted": False,
        "n_runs": 1,
        "n_completed": 1,
        "base_eval_seed": 100,
        "eval_seed_strategy": "deterministic-offset",
        "depth": 0,
        "search_depth": 0,
        "use_expectimax": False,
        "search_canonicalization": "d4-min-packed",
        "search_transposition_table": True,
        "search_batch_size": 32768,
        "compiler": "gcc",
        "cuda_runtime": "",
        "gpu_name": "",
        "python_version": manifest["python_version"],
        "torch_version": manifest["torch_version"],
        "sb3_version": manifest["sb3_version"],
        "outcome_fingerprint": outcome_fingerprint([row]),
    }
    summary = compute_summary_from_rows([row], config, 1.0)
    summary.update({
        "status": "completed",
        "interrupted": False,
        "n_completed": 1,
        "n_runs_requested": 1,
    })
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "summary.json").write_text(json.dumps(summary))
    with (run_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_validate_paper_run_accepts_real_complete_artifact_and_rejects_substitution(
    tmp_path, valid_artifacts,
):
    from scripts.aggregate import validate_paper_run

    run_dir = tmp_path / "run"
    _write_valid_paper_run(run_dir, valid_artifacts)
    result = validate_paper_run(run_dir)
    assert result["config"]["condition"] == "d4"

    valid_artifacts["model_path"].write_bytes(b"substituted")
    with pytest.raises(ValueError, match="model_sha256|model artifact"):
        validate_paper_run(run_dir)


def test_validate_paper_run_rejects_recorded_hash_mismatch(tmp_path, valid_artifacts):
    from scripts.aggregate import validate_paper_run

    run_dir = tmp_path / "run"
    _write_valid_paper_run(run_dir, valid_artifacts)
    config = json.loads((run_dir / "config.json").read_text())
    config["model_sha256"] = "0" * 64
    (run_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="model_sha256|provenance"):
        validate_paper_run(run_dir)


def test_outcome_fingerprint_ignores_logged_twin_metadata():
    from scripts.benchmark_io import OUTCOME_FINGERPRINT_COLUMNS, outcome_fingerprint

    row = {column: 0 for column in OUTCOME_FINGERPRINT_COLUMNS}
    row.update({
        "termination_reason": "board_full",
        "win_1024": False,
        "win_2048": False,
        "win_4096": False,
        "win_8192": False,
    })
    twin = {
        **row,
        "run_id": "different",
        "worker_id": 99,
        "episode_time_s": 2.0,
        "total_think_ms": 4.0,
        "mean_nps": 123.0,
    }

    assert outcome_fingerprint([row]) == outcome_fingerprint([twin])
    changed = {**twin, "score": 1}
    assert outcome_fingerprint([row]) != outcome_fingerprint([changed])


def test_aggregator_ingests_manifest_run_with_arbitrary_folder_name(
    tmp_path, valid_artifacts,
):
    from scripts.aggregate import main

    run_dir = tmp_path / "not-a-depth-folder-name"
    _write_valid_paper_run(run_dir, valid_artifacts)

    assert main([str(tmp_path), "--sweep", "paper-v3"]) == 0
    assert (tmp_path / "summary.csv").exists()
    assert "d4" in (tmp_path / "summary.csv").read_text()


def test_default_aggregation_accepts_runtime_bound_run_without_training_manifest(
    tmp_path, valid_artifacts,
):
    from scripts.aggregate import main

    run_dir = tmp_path / "ordinary-run"
    _write_valid_paper_run(run_dir, valid_artifacts)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    for field in (
        "training_manifest_path", "training_manifest_sha256", "training_model_sha256",
        "training_effective_config_sha256", "training_native_extension_sha256",
        "training_uv_lock_sha256", "training_git_commit", "final_timestep",
        "policy_class", "ppo_class", "value_head_lr_multiplier",
    ):
        config.pop(field, None)
    config["paper_mode"] = False
    config["paper_grade"] = False
    config_path.write_text(json.dumps(config))

    assert main([str(tmp_path), "--sweep", "paper-v3"]) == 0


@pytest.mark.parametrize(
    "artifact", ["model_path", "effective_config_path", "manifest_path"],
)
def test_binding_rejects_missing_path(valid_artifacts, artifact):
    from scripts.paper_provenance import validate_benchmark_training_binding

    path = valid_artifacts[artifact]
    path.unlink()
    with pytest.raises(ValueError, match="missing|unreadable"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"], valid_artifacts["manifest_path"],
        )


def test_binding_rejects_relative_manifest_path(valid_artifacts):
    from scripts.paper_provenance import validate_benchmark_training_binding

    with pytest.raises(ValueError, match="absolute canonical"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"], valid_artifacts["manifest_path"].name,
        )


def test_binding_rejects_empty_model(valid_artifacts):
    from scripts.paper_provenance import validate_benchmark_training_binding

    valid_artifacts["model_path"].write_bytes(b"")
    with pytest.raises(ValueError, match="model|empty"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"], valid_artifacts["manifest_path"],
        )


def test_binding_rejects_empty_effective_config_or_native_extension(
    valid_artifacts, monkeypatch,
):
    from scripts.paper_provenance import validate_benchmark_training_binding
    from scripts.benchmark_provenance import sha256_file

    original_config = valid_artifacts["effective_config_path"].read_bytes()
    valid_artifacts["effective_config_path"].write_bytes(b"")
    with pytest.raises(ValueError, match="effective config|empty"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"], valid_artifacts["manifest_path"],
        )
    valid_artifacts["effective_config_path"].write_bytes(original_config)
    manifest = json.loads(valid_artifacts["manifest_path"].read_text())
    native_path = valid_artifacts["manifest_path"].parent / "test-native.so"
    native_path.write_bytes(b"native")
    native_hash = sha256_file(native_path)
    manifest["native_extension"] = {
        "path": str(native_path.resolve()), "sha256": native_hash,
    }
    manifest["native_extension_sha256"] = native_hash
    valid_artifacts["manifest_path"].write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "scripts.paper_provenance._native_extension_path", lambda: str(native_path),
    )
    native_path.write_bytes(b"")
    with pytest.raises(ValueError, match="native extension|empty"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"], valid_artifacts["manifest_path"],
        )


def test_binding_rejects_substituted_model(valid_artifacts, tmp_path):
    from scripts.paper_provenance import validate_benchmark_training_binding

    substituted = tmp_path / "substituted.zip"
    substituted.write_bytes(valid_artifacts["model_path"].read_bytes())
    with pytest.raises(ValueError, match="model_path"):
        validate_benchmark_training_binding(
            substituted, valid_artifacts["manifest_path"],
        )


def test_binding_rejects_manifest_hash_and_seed_condition_mismatch(valid_artifacts):
    from scripts.paper_provenance import validate_benchmark_training_binding

    with pytest.raises(ValueError, match="training manifest SHA-256 mismatch|provenance"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"],
            valid_artifacts["manifest_path"],
            recorded={"training_manifest_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="train_seed"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"],
            valid_artifacts["manifest_path"],
            expected_train_seed=1,
        )
    with pytest.raises(ValueError, match="condition"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"],
            valid_artifacts["manifest_path"],
            recorded={"condition": "no_d4"},
        )


def test_binding_rejects_training_manifest_mutation(valid_artifacts):
    from scripts.paper_provenance import validate_benchmark_training_binding

    manifest = json.loads(valid_artifacts["manifest_path"].read_text())
    manifest["training_seed"] = 1
    valid_artifacts["manifest_path"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="training seed|effective config"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"], valid_artifacts["manifest_path"],
        )


def test_paper_validation_rejects_dirty_training_manifest(tmp_path, valid_artifacts):
    from scripts.aggregate import validate_paper_run

    run_dir = tmp_path / "run"
    _write_valid_paper_run(run_dir, valid_artifacts)
    manifest = json.loads(valid_artifacts["manifest_path"].read_text())
    manifest["git_dirty"] = True
    manifest["git_status_porcelain"] = " M generated"
    manifest["paper_grade"] = False
    valid_artifacts["manifest_path"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="training manifest is not paper-grade"):
        validate_paper_run(run_dir)


def test_binding_rejects_fictitious_manifest_hash(valid_artifacts):
    from scripts.paper_provenance import validate_benchmark_training_binding

    manifest = json.loads(valid_artifacts["manifest_path"].read_text())
    manifest["model_sha256"] = "model-hash"
    valid_artifacts["manifest_path"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="model_sha256.*missing or invalid|hash is invalid"):
        validate_benchmark_training_binding(
            valid_artifacts["model_path"], valid_artifacts["manifest_path"],
        )


def _synthetic_run(tmp_path, condition, seed, depth, fingerprint):
    path = tmp_path / f"artifact-{condition}-{seed}-{depth}-{fingerprint}"
    config = {
        "condition": condition,
        "training_seed": seed,
        "depth": depth,
        "outcome_fingerprint": fingerprint,
    }
    return {"path": path, "config": config}


def test_manifest_ingestion_supports_full_d4_matrix_and_ignores_twins(
    tmp_path, monkeypatch,
):
    from scripts import aggregate

    fixtures = {}
    candidates = []
    for condition in ("d4", "no_d4"):
        for seed in range(4):
            for depth in range(4):
                fixture = _synthetic_run(tmp_path, condition, seed, depth, "a" * 64)
                fixtures[str(fixture["path"])] = {
                    **fixture,
                    "config": fixture["config"],
                    "episodes": [],
                    "summary": {},
                }
                candidates.append(fixture)

    twin = _synthetic_run(tmp_path, "d4", 0, 0, "a" * 64)
    candidates.append(twin)
    fixtures[str(twin["path"])] = fixtures[str(candidates[0]["path"])]
    monkeypatch.setattr(
        aggregate,
        "validate_paper_run",
        lambda path, **kwargs: fixtures[str(path)],
    )

    validated, twins = aggregate.validate_manifest_runs(
        candidates, require_paper_grade=True,
    )

    assert len(validated) == 32
    assert len(twins) == 1
    assert {run["key"] for run in validated} == {
        (condition, seed, depth)
        for condition in ("d4", "no_d4")
        for seed in range(4)
        for depth in range(4)
    }


def test_normal_manifest_ingestion_accepts_non_paper_condition_and_seed(
    tmp_path, monkeypatch,
):
    from scripts import aggregate

    candidate = _synthetic_run(tmp_path, "baseline", 99, 0, "a" * 64)
    fixture = {**candidate, "episodes": [], "summary": {}}
    monkeypatch.setattr(
        aggregate,
        "validate_benchmark_run",
        lambda path: fixture,
    )

    validated, twins = aggregate.validate_manifest_runs(
        [candidate], require_paper_grade=False,
    )

    assert twins == []
    assert validated[0]["key"] == ("baseline", 99, 0)


def test_manifest_ingestion_rejects_duplicate_identity_with_different_result(
    tmp_path, monkeypatch,
):
    from scripts import aggregate

    first = _synthetic_run(tmp_path, "d4", 0, 3, "a" * 64)
    second = _synthetic_run(tmp_path, "d4", 0, 3, "b" * 64)
    fixtures = {
        str(first["path"]): {**first, "episodes": [], "summary": {}},
        str(second["path"]): {**second, "episodes": [], "summary": {}},
    }
    monkeypatch.setattr(
        aggregate,
        "validate_paper_run",
        lambda path, **kwargs: fixtures[str(path)],
    )

    with pytest.raises(ValueError, match="duplicate experimental identity"):
        aggregate.validate_manifest_runs([first, second], require_paper_grade=True)


def test_manifest_ingestion_rejects_same_outcome_with_different_artifact(
    tmp_path, monkeypatch,
):
    from scripts import aggregate

    first = _synthetic_run(tmp_path, "d4", 0, 3, "a" * 64)
    second = _synthetic_run(tmp_path, "d4", 0, 3, "a" * 64)
    second["path"] = tmp_path / "different-artifact"
    first["config"]["model_sha256"] = "model-a"
    second["config"]["model_sha256"] = "model-b"
    fixtures = {
        str(first["path"]): {**first, "episodes": [], "summary": {}},
        str(second["path"]): {**second, "episodes": [], "summary": {}},
    }
    monkeypatch.setattr(
        aggregate,
        "validate_paper_run",
        lambda path, **kwargs: fixtures[str(path)],
    )

    with pytest.raises(ValueError, match="different provenance"):
        aggregate.validate_manifest_runs([first, second], require_paper_grade=True)


def test_paper_manifest_ingestion_rejects_unpaired_condition(tmp_path, monkeypatch):
    from scripts import aggregate

    candidate = _synthetic_run(tmp_path, "d4", 0, 0, "a" * 64)
    fixture = {
        **candidate,
        "episodes": [{"eval_seed": 100}],
        "summary": {},
    }
    monkeypatch.setattr(
        aggregate,
        "validate_paper_run",
        lambda path, **kwargs: fixture,
    )

    with pytest.raises(ValueError, match="requires d4/no_d4 pairs"):
        aggregate.validate_manifest_runs([candidate], require_paper_grade=True)


def test_paper_manifest_ingestion_rejects_mismatched_eval_seed_pair(
    tmp_path, monkeypatch,
):
    from scripts import aggregate

    left = _synthetic_run(tmp_path, "d4", 0, 0, "a" * 64)
    right = _synthetic_run(tmp_path, "no_d4", 0, 0, "a" * 64)
    fixtures = {
        str(left["path"]): {**left, "episodes": [{"eval_seed": 100}], "summary": {}},
        str(right["path"]): {**right, "episodes": [{"eval_seed": 101}], "summary": {}},
    }
    monkeypatch.setattr(
        aggregate,
        "validate_paper_run",
        lambda path, **kwargs: fixtures[str(path)],
    )

    with pytest.raises(ValueError, match="mismatched eval seed"):
        aggregate.validate_manifest_runs([left, right], require_paper_grade=True)


def test_manifest_discovery_uses_metadata_not_directory_name(tmp_path):
    from scripts.aggregate import discover_manifest_runs

    run_dir = tmp_path / "arbitrary-copy-name"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "sweep_name": "paper-v3",
        "benchmark_schema_version": "2.1.0",
    }))

    runs, violations = discover_manifest_runs(tmp_path, "paper-v3")

    assert violations == []
    assert [run["path"] for run in runs] == [run_dir]


def test_manifest_discovery_reports_incompatible_schema_from_metadata(tmp_path):
    from scripts.aggregate import discover_manifest_runs

    run_dir = tmp_path / "does-not-match-depth-name"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "sweep_name": "paper-v3",
        "benchmark_schema_version": "1.0.0",
    }))

    runs, violations = discover_manifest_runs(tmp_path, "paper-v3")

    assert runs == []
    assert violations == [(str(run_dir), "1.0.0")]


@pytest.mark.parametrize("contents", ["not-json", "[]"])
def test_manifest_discovery_reports_malformed_config_json(tmp_path, contents):
    from scripts.aggregate import discover_manifest_runs

    run_dir = tmp_path / "malformed"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(contents)

    runs, violations = discover_manifest_runs(tmp_path, "paper-v3")

    assert runs == []
    assert violations and violations[0][0] == str(run_dir)
