"""Paper-grade benchmark provenance and artifact-integrity tests."""
from __future__ import annotations

import csv
import json
from argparse import Namespace

import pytest


def _args(**overrides):
    values = {
        "model_path": "model.zip",
        "model_version": None,
        "train_seed": 7,
        "n_runs": 2,
        "workers": 1,
        "device": "cpu",
        "depth": 3,
        "log_moves": False,
        "base_eval_seed": 100,
        "paper_mode": True,
        "allow_dirty_paper_run": False,
        "effective_config": None,
        "search_batch_size": 32768,
        "disable_tt": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _paper_args(tmp_path, **overrides):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")
    (tmp_path / "effective_config.json").write_text(
        json.dumps({"root_training_seed": 7})
    )
    (tmp_path / "training_manifest.json").write_text("{}")
    return _args(model_path=str(model_path), **overrides)


@pytest.fixture(autouse=True)
def _stub_training_manifest_binding(monkeypatch):
    """Keep structural integrity tests independent from model loading."""
    monkeypatch.setattr(
        "scripts.aggregate.validate_benchmark_training_binding",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "scripts.benchmark.validate_benchmark_training_binding",
        lambda *args, **kwargs: {
            "training_manifest_path": str(args[1]),
            "training_manifest_sha256": "t" * 64,
            "training_model_sha256": "m" * 64,
            "training_seed": kwargs.get("expected_train_seed", 7),
            "condition": "d4",
            "d4_augment": True,
            "policy_class": "policy",
            "ppo_class": "ppo",
            "value_head_lr_multiplier": 10.0,
            "training_effective_config_sha256": "e" * 64,
            "final_timestep": 100,
            "training_git_commit": "c" * 40,
            "training_native_extension_sha256": "n" * 64,
            "training_uv_lock_sha256": "u" * 64,
        },
    )


def _provenance(**overrides):
    values = {
        "model_sha256": "model", "effective_config_sha256": "config",
        "uv_lock_sha256": "lock", "native_extension_sha256": "extension",
        "python_version": "3.12", "torch_version": "2.9", "sb3_version": "2.8",
        "cuda_runtime": "", "gpu_name": "", "compiler": "gcc",
    }
    values.update(overrides)
    return values


def _write_run(
    tmp_path, *, rows=None, config_overrides=None, summary_overrides=None,
    episode_fieldnames=None,
):
    from scripts.benchmark_io import EPISODE_COLUMNS
    from scripts.benchmark_summary import compute_summary_from_rows

    rows = rows or [_episode(0, 100), _episode(1, 101)]
    (tmp_path / "model.zip").write_bytes(b"model")
    (tmp_path / "effective_config.json").write_text("{}")
    (tmp_path / "training_manifest.json").write_text("{}")
    config = {
        "benchmark_schema_version": "2.1.0",
        "run_name": tmp_path.name,
        "sweep_name": tmp_path.name,
        "run_id": "run",
        "model_path": str(tmp_path / "model.zip"),
        "effective_config_path": str(tmp_path / "effective_config.json"),
        "training_manifest_path": str(tmp_path / "training_manifest.json"),
        "model_sha256": "m" * 64,
        "effective_config_sha256": "e" * 64,
        "uv_lock_sha256": "u" * 64,
        "native_extension_sha256": "n" * 64,
        "training_manifest_sha256": "t" * 64,
        "training_model_sha256": "m" * 64,
        "n_runs": 2,
        "status": "completed",
        "interrupted": False,
        "n_completed": 2,
        "paper_mode": True,
        "paper_grade": True,
        "git_dirty": False,
        "git_commit": "a" * 40,
        "base_eval_seed": 100,
        "eval_seed_strategy": "deterministic-offset",
        "train_seed": 7,
        "training_seed": 7,
        "condition": "d4",
        "d4_augment": True,
        "policy_class": "policy",
        "ppo_class": "ppo",
        "value_head_lr_multiplier": 10.0,
        "training_effective_config_sha256": "e" * 64,
        "final_timestep": 100,
        "training_git_commit": "c" * 40,
        "training_native_extension_sha256": "n" * 64,
        "training_uv_lock_sha256": "u" * 64,
        "model_sha256": "model-hash",
        "effective_config_sha256": "config-hash",
        "uv_lock_sha256": "lock-hash",
        "native_extension_sha256": "extension-hash",
        "python_version": "3.12.0",
        "torch_version": "2.9.1",
        "sb3_version": "2.8.0",
        "cuda_runtime": "",
        "gpu_name": "",
        "compiler": "gcc",
        "search_depth": 3,
        "search_canonicalization": "d4-min-packed",
        "search_transposition_table": True,
        "search_batch_size": 32768,
        "depth": 3,
        "use_expectimax": True,
    }
    config.update(config_overrides or {})
    from scripts.benchmark_io import outcome_fingerprint
    config["outcome_fingerprint"] = outcome_fingerprint(rows)
    summary = compute_summary_from_rows(rows, config, 1.0)
    summary.update({
        "status": config["status"],
        "interrupted": config["interrupted"],
        "n_completed": len(rows),
        "n_runs_requested": config["n_runs"],
    })
    if summary_overrides:
        summary.update(summary_overrides)

    with (tmp_path / "config.json").open("w") as stream:
        json.dump(config, stream)
    with (tmp_path / "summary.json").open("w") as stream:
        json.dump(summary, stream)
    with (tmp_path / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=episode_fieldnames or EPISODE_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return config


def _episode(index, eval_seed, **overrides):
    row = {
        "schema_version": "2.1.0", "run_id": "run", "episode_idx": index,
        "worker_id": 0, "train_seed": 7, "eval_seed": eval_seed,
        "requested_depth": 3, "effective_depth": 3, "use_expectimax": True,
        "score": 1000 + index, "max_tile": 128, "max_log_tile": 7,
        "steps": 50, "episode_time_s": 1.0, "mean_move_time_ms": 1.0,
        "median_move_time_ms": 1.0, "p95_move_time_ms": 1.0,
        "max_move_time_ms": 1.0, "termination_reason": "board_full",
        "win_1024": False, "win_2048": False, "win_4096": False,
        "win_8192": False, "total_think_ms": 2.0, "total_nodes": 3,
        "total_batches": 1, "total_tt_lookups": 1, "total_tt_hits": 0,
        "total_tt_collisions": 0, "total_tt_same_key_overwrites": 0,
        "total_moves_resolved": 1, "total_moves_unresolved": 0,
        "total_cap_hits": 0,
        "total_chance_nodes": 1, "total_max_nodes": 1,
        "mean_chance_value": 1.0, "mean_empty_cells": 8.0,
        "min_empty_cells": 3, "mean_merge_score": 4.0, "mean_nps": 5.0,
        "mean_tt_hit_rate": 0.0, "mean_nodes_per_batch_call": 3.0,
    }
    row.update(overrides)
    return row


def test_paper_mode_rejects_dirty_git_without_override(monkeypatch):
    from scripts.benchmark import validate_paper_mode

    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: True)

    with pytest.raises(ValueError, match="clean git worktree"):
        validate_paper_mode(_args())


def test_dirty_paper_override_is_explicitly_non_paper_grade(monkeypatch, tmp_path):
    from scripts.benchmark import build_config

    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: True)
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")
    (tmp_path / "effective_config.json").write_text(
        json.dumps({"root_training_seed": 7})
    )
    (tmp_path / "training_manifest.json").write_text("{}")
    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark.collect_runtime_provenance", lambda **_: _provenance())

    config = build_config(
        _args(model_path=str(model_path), allow_dirty_paper_run=True), "run", 100,
        "deterministic-offset", "2026-01-01T00:00:00Z",
    )

    assert config["paper_mode"] is True
    assert config["paper_grade"] is False


def test_build_config_records_provenance_fields(monkeypatch):
    from scripts.benchmark import build_config

    provenance = {
        "model_sha256": "model", "effective_config_sha256": "config",
        "uv_lock_sha256": "lock", "native_extension_sha256": "extension",
        "python_version": "3.12", "torch_version": "2.9", "sb3_version": "2.8",
        "cuda_runtime": "13.0", "gpu_name": "GPU", "compiler": "gcc",
    }
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)
    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "commit")
    monkeypatch.setattr("scripts.benchmark.collect_runtime_provenance", lambda **_: provenance)

    config = build_config(
        _args(paper_mode=False), "run", 100, "deterministic-offset", "2026-01-01T00:00:00Z",
    )

    assert config["git_commit"] == "commit"
    assert config["model_sha256"] == "model"
    assert config["effective_config_sha256"] == "config"
    assert config["uv_lock_sha256"] == "lock"
    assert config["native_extension_sha256"] == "extension"
    assert config["search_depth"] == 3
    assert config["search_canonicalization"] == "d4-min-packed"
    assert config["search_transposition_table"] is True
    assert config["search_batch_size"] == 32768


def test_paper_mode_search_failure_is_detected_before_aggregation():
    from scripts.benchmark_runner import paper_search_failure_reason

    assert paper_search_failure_reason(_episode(0, 100)) is None
    assert "cap hits" in paper_search_failure_reason(
        _episode(0, 100, total_cap_hits=1)
    )
    assert "unresolved" in paper_search_failure_reason(
        _episode(0, 100, total_moves_unresolved=1)
    )


def test_aggregate_cli_exposes_paper_mode():
    from scripts.aggregate import parse_args

    args = parse_args(["benchmarks", "--sweep", "paper", "--paper-mode"])

    assert args.paper_mode is True


def test_summary_csv_fieldnames_include_columns_only_present_in_later_rows():
    from scripts.aggregate import summary_fieldnames

    fieldnames = summary_fieldnames([
        {"depth": 0, "avg_score": 1.0},
        {"depth": 1, "avg_score": 2.0, "avg_think_ms": 3.0},
    ])

    assert fieldnames == ["depth", "avg_score", "avg_think_ms"]


@pytest.mark.parametrize("git_commit, git_dirty", [(None, False), ("unknown", False), ("a" * 40, None)])
def test_paper_mode_rejects_unknown_git_identity_or_status(monkeypatch, git_commit, git_dirty):
    from scripts.benchmark import validate_paper_mode

    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: git_commit)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: git_dirty)

    with pytest.raises(ValueError, match="git identity or status"):
        validate_paper_mode(_args())


def test_paper_build_config_fails_closed_when_git_status_cannot_be_collected(monkeypatch):
    from scripts.benchmark import build_config

    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: None)

    with pytest.raises(ValueError, match="git identity or status"):
        build_config(_args(), "run", 100, "deterministic-offset", "2026-01-01T00:00:00Z")


def test_paper_build_config_fails_closed_when_effective_config_disappears(monkeypatch):
    from scripts.benchmark import build_config

    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)

    with pytest.raises(ValueError, match="effective config"):
        build_config(_args(), "run", 100, "deterministic-offset", "2026-01-01T00:00:00Z")


def test_paper_mode_rejects_missing_train_seed_before_workers(monkeypatch, tmp_path):
    from scripts.benchmark import validate_paper_mode

    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)

    with pytest.raises(ValueError, match="--train-seed"):
        validate_paper_mode(_paper_args(tmp_path, train_seed=None))


def test_paper_mode_rejects_missing_eval_seed_before_workers(monkeypatch, tmp_path):
    from scripts.benchmark import validate_paper_mode

    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)

    with pytest.raises(ValueError, match="--base-eval-seed"):
        validate_paper_mode(_paper_args(tmp_path, base_eval_seed=None))


@pytest.mark.parametrize(
    "field",
    [
        "model_sha256", "effective_config_sha256", "uv_lock_sha256",
        "native_extension_sha256", "python_version", "torch_version",
        "sb3_version", "compiler",
    ],
)
def test_paper_mode_rejects_unavailable_required_provenance(monkeypatch, tmp_path, field):
    from scripts.benchmark import validate_paper_mode

    provenance = _provenance(**{field: ""})
    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)
    monkeypatch.setattr(
        "scripts.benchmark.collect_runtime_provenance", lambda **_: provenance,
    )

    with pytest.raises(ValueError, match=field):
        validate_paper_mode(_paper_args(tmp_path))


def test_paper_mode_uses_model_adjacent_effective_config(monkeypatch, tmp_path):
    from scripts.benchmark import validate_paper_mode

    model_path = tmp_path / "final_model.zip"
    model_path.write_bytes(b"model")
    effective_config = tmp_path / "effective_config.json"
    effective_config.write_text(json.dumps({"root_training_seed": 7}))
    (tmp_path / "training_manifest.json").write_text("{}")
    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)
    args = _args(model_path=str(model_path))

    validate_paper_mode(args)

    assert args.effective_config == str(effective_config)


def test_paper_mode_rejects_missing_effective_config(monkeypatch, tmp_path):
    from scripts.benchmark import validate_paper_mode

    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)

    with pytest.raises(ValueError, match="effective config"):
        validate_paper_mode(_args(model_path=str(tmp_path / "model.zip")))


@pytest.mark.parametrize(
    ("effective_config", "error"),
    [
        ({}, "root_training_seed"),
        ({"root_training_seed": 8}, "root_training_seed"),
    ],
)
def test_paper_mode_rejects_effective_config_with_invalid_training_seed(
    monkeypatch, tmp_path, effective_config, error,
):
    from scripts.benchmark import validate_paper_mode

    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)
    args = _paper_args(tmp_path)
    (tmp_path / "effective_config.json").write_text(json.dumps(effective_config))

    with pytest.raises(ValueError, match=error):
        validate_paper_mode(args)


def test_paper_mode_accepts_existing_explicit_effective_config(monkeypatch, tmp_path):
    from scripts.benchmark import validate_paper_mode

    supplied = tmp_path / "supplied-effective-config.json"
    supplied.write_text(json.dumps({"root_training_seed": 7}))
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")
    (tmp_path / "training_manifest.json").write_text("{}")
    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "a" * 40)
    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)
    args = _args(
        model_path=str(model_path), effective_config=str(supplied),
    )

    validate_paper_mode(args)

    assert args.effective_config == str(supplied)


@pytest.mark.parametrize(
    ("rows", "config_overrides", "summary_overrides", "error"),
    [
        (None, {"status": "failed"}, None, "status is not completed"),
        ([_episode(0, 100)], None, None, "row count"),
        ([_episode(0, 100), _episode(0, 101)], None, None, "episode_idx"),
        ([_episode(0, 100), _episode(1, 100)], None, None, "eval_seed"),
        ([_episode(0, 100), _episode(1, 101, total_cap_hits=1)], None, None, "cap hits"),
        ([_episode(0, 100), _episode(1, 101, total_moves_unresolved=1)], None, None, "unresolved"),
        (None, {"git_dirty": True, "paper_grade": False}, None, "paper-grade"),
        (None, None, {"metrics": {"avg_score": -1}}, "summary"),
    ],
)
def test_strict_validation_rejects_invalid_paper_artifacts(
    tmp_path, rows, config_overrides, summary_overrides, error,
):
    from scripts.aggregate import validate_paper_run

    _write_run(
        tmp_path, rows=rows, config_overrides=config_overrides,
        summary_overrides=summary_overrides,
    )

    with pytest.raises(ValueError, match=error):
        validate_paper_run(tmp_path)


def test_strict_validation_accepts_complete_consistent_artifact(tmp_path):
    from scripts.aggregate import validate_paper_run

    _write_run(tmp_path)

    result = validate_paper_run(tmp_path)

    assert result["config"]["model_sha256"] == "model-hash"
    assert [row["eval_seed"] for row in result["episodes"]] == [100, 101]


@pytest.mark.parametrize("n_runs", [0, -1, True])
def test_strict_validation_rejects_nonpositive_n_runs(tmp_path, n_runs):
    from scripts.aggregate import validate_paper_run

    _write_run(tmp_path, config_overrides={"n_runs": n_runs})

    with pytest.raises(ValueError, match="n_runs"):
        validate_paper_run(tmp_path)


@pytest.mark.parametrize(
    ("config_overrides", "summary_overrides", "episode_fieldnames", "rows", "error"),
    [
        ({"benchmark_schema_version": "1.0.0"}, None, None, None, "schema"),
        ({"benchmark_schema_version": "2.0.0"}, None, None, None, "schema"),
        (None, {"benchmark_schema_version": "1.0.0"}, None, None, "summary schema"),
        (None, None, None, [_episode(0, 100, schema_version="1.0.0"), _episode(1, 101)], "episode schema"),
        (None, None, ["episode_idx", "eval_seed", "score"], None, "episode CSV columns"),
        (None, None, None, [_episode(0, 100, requested_depth=2), _episode(1, 101)], "requested_depth"),
        (None, None, None, [_episode(0, 100, effective_depth=2), _episode(1, 101)], "effective_depth"),
        (None, None, None, [_episode(0, 100, use_expectimax=False), _episode(1, 101)], "use_expectimax"),
        ({"run_id": None}, None, None, None, "run_id"),
        ({"run_id": "other"}, None, None, None, "run_id"),
        (None, None, None, [_episode(0, 100, run_id="other"), _episode(1, 101)], "run_id"),
    ],
)
def test_strict_validation_rejects_schema_missing_columns_and_search_mismatches(
    tmp_path, config_overrides, summary_overrides, episode_fieldnames, rows, error,
):
    from scripts.aggregate import validate_paper_run

    _write_run(
        tmp_path, rows=rows, config_overrides=config_overrides,
        summary_overrides=summary_overrides, episode_fieldnames=episode_fieldnames,
    )

    with pytest.raises(ValueError, match=error):
        validate_paper_run(tmp_path)


def test_strict_aggregate_checks_schema_for_direct_single_run(tmp_path):
    from scripts.aggregate import main

    _write_run(tmp_path, config_overrides={"benchmark_schema_version": "1.0.0"})

    assert main([str(tmp_path), "--sweep", tmp_path.name, "--paper-mode"]) == 2


def test_training_persists_resolved_config_beside_models(tmp_path):
    from scripts.train import persist_effective_config

    path = persist_effective_config(tmp_path, {"env_kwargs": {"d4_augment": True}})

    assert path == tmp_path / "effective_config.json"
    assert json.loads(path.read_text()) == {"env_kwargs": {"d4_augment": True}}


def test_paired_validation_rejects_provenance_or_seed_set_mismatch(tmp_path):
    from scripts.aggregate import validate_paired_paper_runs

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _write_run(left)
    _write_run(
        right,
        config_overrides={
            "native_extension_sha256": "other",
            "condition": "no_d4",
            "d4_augment": False,
        },
    )

    with pytest.raises(ValueError, match="native_extension_sha256"):
        validate_paired_paper_runs([left, right])

    _write_run(
        right, rows=[_episode(0, 101), _episode(1, 102)],
        config_overrides={
            "base_eval_seed": 101,
            "condition": "no_d4",
            "d4_augment": False,
        },
    )

    with pytest.raises(ValueError, match="eval seed set"):
        validate_paired_paper_runs([left, right])
