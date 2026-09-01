"""
Post-processing aggregator for AI2048 benchmark results.

Produces summary.csv and paper_figures/ with violin plots, bar charts, and heatmaps
across seeds and depths for a given sweep.

The v3 target examples below apply only after the complete PRE-FREEZE matrix
has been produced; the diagnostic pilot is not an input to paper aggregation.

Usage:
    python scripts/aggregate.py data/benchmarks/ --sweep hybrid_ppo_v3_pre_freeze
    python scripts/aggregate.py data/benchmarks/ --sweep hybrid_ppo_v3_pre_freeze --win-threshold 4096

Output: summary.csv + paper_figures/ in the benchmark root directory.
"""

import argparse
import os
import re
import sys
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

from scripts.benchmark_summary import compute_summary_from_rows
from scripts.benchmark_io import (
    EPISODE_COLUMNS,
    EPISODE_SCHEMA_VERSION,
    outcome_fingerprint,
)
from scripts.benchmark_provenance import (
    REPO_ROOT,
    _native_extension_path,
    validate_artifact_sha256,
)
from scripts.paper_provenance import validate_benchmark_training_binding
from scripts.benchmark_statistics import (
    EXPECTED_TRAINING_SEEDS,
    compute_model_metrics,
    cross_depth_effects,
    model_level_confidence_intervals,
    paired_d4_no_d4_effects,
    sample_sd,
    student_t_critical_95,
    validate_complete_design,
)

_REQUIRED_PAPER_PROVENANCE = (
    "git_commit", "sweep_name", "model_path", "model_sha256", "effective_config_path",
    "effective_config_sha256", "uv_lock_sha256",
    "native_extension_sha256", "train_seed", "base_eval_seed", "python_version",
    "torch_version", "sb3_version", "cuda_runtime", "gpu_name", "compiler",
    "search_depth", "search_canonicalization", "search_transposition_table",
    "search_batch_size", "depth", "use_expectimax",
    "training_manifest_path", "training_manifest_sha256", "training_model_sha256",
    "training_seed",
    "condition", "d4_augment", "policy_class", "ppo_class",
    "value_head_lr_multiplier", "training_effective_config_sha256",
    "final_timestep", "training_git_commit",
    "training_native_extension_sha256", "training_uv_lock_sha256",
    "outcome_fingerprint",
)
_REQUIRED_RUNTIME_PROVENANCE = (
    "git_commit", "model_path", "model_sha256", "effective_config_path",
    "effective_config_sha256", "uv_lock_sha256", "native_extension_sha256",
    "python_version", "torch_version", "sb3_version", "compiler",
    "search_depth", "search_canonicalization", "search_transposition_table",
    "search_batch_size", "depth", "use_expectimax", "outcome_fingerprint",
)
_PAIRED_PROVENANCE = (
    "git_commit", "uv_lock_sha256", "native_extension_sha256", "device",
    "python_version", "torch_version",
    "sb3_version", "cuda_runtime", "gpu_name", "compiler",
    "search_canonicalization", "search_transposition_table", "search_batch_size",
)
_TWIN_IDENTITY_FIELDS = (
    "git_commit", "model_sha256", "effective_config_sha256", "uv_lock_sha256",
    "native_extension_sha256", "device", "python_version", "torch_version",
    "sb3_version", "cuda_runtime", "gpu_name", "compiler", "base_eval_seed",
    "search_depth", "search_canonicalization", "search_transposition_table",
    "search_batch_size", "training_manifest_sha256", "training_model_sha256",
    "train_seed", "training_seed", "condition", "d4_augment", "policy_class", "ppo_class",
    "value_head_lr_multiplier", "training_effective_config_sha256",
    "final_timestep", "training_git_commit", "training_native_extension_sha256",
    "training_uv_lock_sha256", "paper_grade", "git_dirty",
)
_INT_EPISODE_FIELDS = {
    "episode_idx", "worker_id", "train_seed", "eval_seed", "requested_depth",
    "effective_depth", "score", "max_tile", "max_log_tile", "steps", "total_nodes",
    "total_batches", "total_tt_lookups", "total_tt_hits", "total_tt_collisions",
    "total_tt_same_key_overwrites", "total_moves_resolved", "total_moves_unresolved",
    "total_cap_hits", "total_chance_nodes", "total_max_nodes",
    "min_empty_cells",
}
_FLOAT_EPISODE_FIELDS = {
    "episode_time_s", "mean_move_time_ms", "median_move_time_ms", "p95_move_time_ms",
    "max_move_time_ms", "total_think_ms", "mean_chance_value", "mean_empty_cells",
    "mean_merge_score", "mean_nps", "mean_tt_hit_rate", "mean_nodes_per_batch_call",
}
_REQUIRED_EPISODE_METRICS = {
    *_INT_EPISODE_FIELDS - {"train_seed"},
    *_FLOAT_EPISODE_FIELDS,
    "termination_reason",
}
_BOOL_EPISODE_FIELDS = ("use_expectimax", "win_1024", "win_2048", "win_4096", "win_8192")


def validate_paper_run(
    run_dir: str | Path, *, require_paper_grade: bool = True,
) -> dict:
    """Validate one complete manifest-bound run before it can be aggregated."""
    return _validate_run(
        run_dir, require_paper_grade=require_paper_grade,
        require_training_binding=True,
    )


def validate_benchmark_run(run_dir: str | Path) -> dict:
    """Validate one complete runtime-provenance-bound run without training metadata."""
    return _validate_run(
        run_dir, require_paper_grade=False,
        require_training_binding=False,
    )


def _validate_run(
    run_dir: str | Path, *, require_paper_grade: bool,
    require_training_binding: bool,
) -> dict:
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.json")
    summary = _load_json(run_dir / "summary.json")
    episodes = _load_episode_rows(run_dir / "episodes.csv")

    if config.get("benchmark_schema_version") != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"{run_dir}: unsupported config schema version")
    if (
        summary.get("benchmark_schema_version") != EPISODE_SCHEMA_VERSION
        or summary.get("benchmark_schema_version") != config.get("benchmark_schema_version")
    ):
        raise ValueError(f"{run_dir}: unsupported or mismatched summary schema version")

    if config.get("status") != "completed":
        raise ValueError(f"{run_dir}: status is not completed")
    if type(config.get("interrupted")) is not bool or config["interrupted"]:
        raise ValueError(f"{run_dir}: interrupted must be false for completed runs")
    if require_paper_grade and (not config.get("paper_grade") or config.get("git_dirty")):
        raise ValueError(f"{run_dir}: run is not paper-grade")
    required_provenance = (
        _REQUIRED_PAPER_PROVENANCE if require_training_binding
        else _REQUIRED_RUNTIME_PROVENANCE
    )
    missing = [key for key in required_provenance if key not in config or config[key] is None]
    nonempty = (
        "git_commit", "sweep_name", "model_path", "model_sha256", "effective_config_path",
        "effective_config_sha256", "uv_lock_sha256", "native_extension_sha256",
        "compiler", "search_canonicalization", "training_manifest_path",
        "training_manifest_sha256", "training_model_sha256", "condition",
        "policy_class", "ppo_class",
        "training_git_commit", "training_effective_config_sha256",
        "training_native_extension_sha256", "training_uv_lock_sha256",
        "outcome_fingerprint",
    ) if require_training_binding else (
        "git_commit", "model_path", "model_sha256", "effective_config_path",
        "effective_config_sha256", "uv_lock_sha256", "native_extension_sha256",
        "python_version", "torch_version", "sb3_version", "compiler",
        "search_canonicalization", "outcome_fingerprint",
    )
    missing.extend(key for key in nonempty if not config.get(key) and key not in missing)
    if missing:
        raise ValueError(f"{run_dir}: missing paper provenance: {', '.join(missing)}")

    if (
        not isinstance(config.get("git_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", config["git_commit"]) is None
    ):
        raise ValueError(f"{run_dir}: git_commit must be a 40-character SHA")
    for field in ("git_dirty", "paper_grade"):
        if type(config.get(field)) is not bool:
            raise ValueError(f"{run_dir}: {field} must be a boolean")
    path_fields = ["model_path", "effective_config_path"]
    if require_training_binding:
        path_fields.append("training_manifest_path")
    for field in path_fields:
        value = config.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{run_dir}: {field} must be a string path")
        path = Path(value)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError(f"{run_dir}: {field} must be an absolute canonical path")
    string_fields = [
        "python_version", "torch_version", "sb3_version", "compiler",
        "search_canonicalization", "cuda_runtime", "gpu_name", "device",
    ]
    if require_training_binding:
        string_fields.extend([
            "sweep_name", "condition", "policy_class", "ppo_class",
            "training_git_commit",
        ])
    for field in string_fields:
        if field in config and not isinstance(config[field], str):
            raise ValueError(f"{run_dir}: {field} must be a string")
    for field in ("use_expectimax", "search_transposition_table", "git_dirty", "paper_grade"):
        if type(config.get(field)) is not bool:
            raise ValueError(f"{run_dir}: {field} must be a boolean")
    integer_fields = ("train_seed", "training_seed", "search_depth", "search_batch_size")
    if require_training_binding:
        integer_fields += ("final_timestep",)
    for field in integer_fields:
        if field in config and config[field] is not None and (
            isinstance(config[field], bool) or not isinstance(config[field], int)
        ):
            raise ValueError(f"{run_dir}: {field} must be an integer")
    if require_training_binding:
        if re.fullmatch(r"[0-9a-f]{40}", config["training_git_commit"]) is None:
            raise ValueError(f"{run_dir}: training_git_commit must be a 40-character SHA")
        if type(config["d4_augment"]) is not bool:
            raise ValueError(f"{run_dir}: d4_augment must be a boolean")
        multiplier = config["value_head_lr_multiplier"]
        if type(multiplier) not in (int, float) or not math.isfinite(float(multiplier)):
            raise ValueError(f"{run_dir}: value_head_lr_multiplier must be finite")

    requested_value = config.get("n_runs")
    if (
        isinstance(requested_value, bool)
        or not isinstance(requested_value, int)
        or requested_value <= 0
    ):
        raise ValueError(f"{run_dir}: n_runs must be a positive integer")
    requested = requested_value
    if require_training_binding:
        validate_benchmark_training_binding(
            config["model_path"],
            config["training_manifest_path"],
            expected_train_seed=config["train_seed"],
            expected_effective_config_path=config["effective_config_path"],
            recorded=config,
            require_recorded=True,
            require_paper_grade=require_paper_grade,
        )
        if config["train_seed"] != config["training_seed"]:
            raise ValueError(f"{run_dir}: train_seed does not match training_seed")
    else:
        _validate_runtime_artifacts(config)
        if (
            config.get("train_seed") is not None
            and config.get("training_seed") is not None
            and config["train_seed"] != config["training_seed"]
        ):
            raise ValueError(f"{run_dir}: train_seed does not match training_seed")

    base_eval_seed = config.get("base_eval_seed")
    if isinstance(base_eval_seed, bool) or not isinstance(base_eval_seed, int):
        raise ValueError(f"{run_dir}: base_eval_seed must be an integer")
    depth_value = config.get("depth")
    if (
        isinstance(depth_value, bool)
        or not isinstance(depth_value, int)
        or depth_value not in range(4)
    ):
        raise ValueError(f"{run_dir}: depth must be one of 0, 1, 2, 3")
    if (
        type(config.get("n_completed")) is not int
        or config["n_completed"] != requested
    ):
        raise ValueError(f"{run_dir}: episode row count does not match requested runs")
    if (
        summary.get("status") != "completed"
        or type(summary.get("interrupted")) is not bool
        or summary["interrupted"]
        or type(summary.get("n_completed")) is not int
        or type(summary.get("n_runs_requested")) is not int
        or summary["n_completed"] != requested
        or summary["n_runs_requested"] != requested
    ):
        raise ValueError(f"{run_dir}: summary row count does not match requested runs")
    if len(episodes) != requested:
        raise ValueError(f"{run_dir}: episode row count does not match requested runs")
    if any(row["schema_version"] != EPISODE_SCHEMA_VERSION for row in episodes):
        raise ValueError(f"{run_dir}: unsupported episode schema version")

    run_id = config.get("run_id")
    if run_id in (None, "") or any(row.get("run_id") != run_id for row in episodes):
        raise ValueError(f"{run_dir}: episode run_id does not match config")
    indices = [row["episode_idx"] for row in episodes]
    if len(indices) != len(set(indices)) or set(indices) != set(range(requested)):
        raise ValueError(f"{run_dir}: duplicate or missing episode_idx values")
    seeds = [row["eval_seed"] for row in episodes]
    expected_seeds = {base_eval_seed + index for index in range(requested)}
    if len(seeds) != len(set(seeds)) or set(seeds) != expected_seeds:
        raise ValueError(f"{run_dir}: duplicate or missing eval_seed values")
    train_seed = config.get("train_seed")
    if train_seed is None:
        if any(row.get("train_seed") not in (None, "") for row in episodes):
            raise ValueError(f"{run_dir}: episode training seed does not match config")
    elif any(row.get("train_seed") != train_seed for row in episodes):
        raise ValueError(f"{run_dir}: episode training seed does not match config")
    expected_depth = depth_value
    if type(config.get("use_expectimax")) is not bool:
        raise ValueError(f"{run_dir}: use_expectimax must be a boolean")
    expected_search = config["use_expectimax"]
    if any(row["requested_depth"] != expected_depth for row in episodes):
        raise ValueError(f"{run_dir}: episode requested_depth does not match config")
    if any(row["effective_depth"] != expected_depth for row in episodes):
        raise ValueError(f"{run_dir}: episode effective_depth does not match config")
    if any(row["use_expectimax"] != expected_search for row in episodes):
        raise ValueError(f"{run_dir}: episode use_expectimax does not match config")
    if (
        not isinstance(config["outcome_fingerprint"], str)
        or re.fullmatch(r"[0-9a-f]{64}", config["outcome_fingerprint"]) is None
    ):
        raise ValueError(f"{run_dir}: outcome_fingerprint is invalid")
    if outcome_fingerprint(episodes) != config["outcome_fingerprint"]:
        raise ValueError(f"{run_dir}: outcome_fingerprint does not match episodes")
    if any(row["total_cap_hits"] for row in episodes):
        raise ValueError(f"{run_dir}: episodes contain search cap hits")
    if any(row["total_moves_unresolved"] for row in episodes):
        raise ValueError(f"{run_dir}: episodes contain unresolved search moves")

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{run_dir}: summary metrics must be an object")
    try:
        total_time_s = float(metrics.get("total_time_s", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{run_dir}: summary total_time_s is invalid") from exc
    if not math.isfinite(total_time_s):
        raise ValueError(f"{run_dir}: summary total_time_s must be finite")
    expected_summary = compute_summary_from_rows(episodes, config, total_time_s)
    expected_summary.update({
        "status": "completed", "interrupted": False,
        "n_completed": requested, "n_runs_requested": requested,
    })
    for key in expected_summary:
        if key not in summary or not _equivalent(expected_summary[key], summary[key]):
            raise ValueError(f"{run_dir}: summary.json does not match episodes.csv ({key})")
    return {"config": config, "summary": summary, "episodes": episodes}


def _validate_runtime_artifacts(config: dict) -> None:
    validate_artifact_sha256(config["model_path"], config["model_sha256"], label="model")
    validate_artifact_sha256(
        config["effective_config_path"], config["effective_config_sha256"],
        label="effective config",
    )
    validate_artifact_sha256(
        _native_extension_path(), config["native_extension_sha256"],
        label="native extension",
    )
    validate_artifact_sha256(
        REPO_ROOT / "uv.lock", config["uv_lock_sha256"], label="uv.lock",
    )


def validate_paired_paper_runs(run_dirs) -> list[dict]:
    """Require paired conditions to share eval seeds and execution provenance."""
    runs = [validate_paper_run(run_dir) for run_dir in run_dirs]
    _validate_paired_runs(runs)
    _require_condition_pair(runs)
    return runs


def _validate_paired_runs(runs: list[dict]) -> None:
    """Require already validated paired runs to share seeds and provenance."""
    if len(runs) < 2:
        return
    baseline = runs[0]
    baseline_seeds = {row["eval_seed"] for row in baseline["episodes"]}
    for run in runs[1:]:
        if {row["eval_seed"] for row in run["episodes"]} != baseline_seeds:
            raise ValueError("paired runs have mismatched eval seed sets")
        for key in _PAIRED_PROVENANCE:
            if run["config"].get(key) != baseline["config"].get(key):
                raise ValueError(f"paired runs differ in {key}")


def _same_twin_identity(left: dict, right: dict) -> bool:
    return all(left.get(field) == right.get(field) for field in _TWIN_IDENTITY_FIELDS)


def _require_condition_pair(runs: list[dict]) -> None:
    if len(runs) >= 2 and {
        run["config"].get("condition") for run in runs
    } != {"d4", "no_d4"}:
        raise ValueError("paper pairing requires d4/no_d4 conditions")


def _load_json(path: Path) -> dict:
    try:
        with path.open() as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _load_episode_rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            if fieldnames != EPISODE_COLUMNS:
                missing = [column for column in EPISODE_COLUMNS if column not in fieldnames]
                unexpected = sorted(set(fieldnames) - set(EPISODE_COLUMNS))
                raise ValueError(
                    f"{path}: episode CSV columns are not exact; "
                    f"missing={', '.join(missing)} unexpected={', '.join(unexpected)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"{path}: unreadable episodes CSV") from exc
    for row_index, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: malformed episode CSV row at row {row_index}")
        missing_metrics = [
            key for key in _REQUIRED_EPISODE_METRICS
            if row.get(key) in (None, "")
        ]
        if missing_metrics:
            raise ValueError(
                f"{path}: missing episode metric values at row {row_index}: "
                f"{', '.join(sorted(missing_metrics))}"
            )
        for key in _INT_EPISODE_FIELDS:
            if key in row and row[key] not in (None, ""):
                row[key] = int(row[key])
        for key in _FLOAT_EPISODE_FIELDS:
            if key in row and row[key] not in (None, ""):
                value = float(row[key])
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}: episode metric {key} must be finite "
                        f"at row {row_index}"
                    )
                row[key] = value
        for key in _BOOL_EPISODE_FIELDS:
            row[key] = _parse_bool(row[key], path)
    return rows


def _parse_bool(value, path: Path) -> bool:
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"{path}: invalid boolean use_expectimax value")


def _equivalent(expected, actual) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _equivalent(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _equivalent(left, right) for left, right in zip(expected, actual)
        )
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return expected == actual


def summary_fieldnames(rows: list[dict]) -> list[str]:
    """Return the ordered union of all summary row fields."""
    return list(dict.fromkeys(key for row in rows for key in row))


def _check_schema_versions(benchmark_dir, sweep_name):
    """Return incompatible schema versions from manifest-selected runs."""
    return discover_manifest_runs(benchmark_dir, sweep_name)[1]


def discover_depth_folders(benchmark_dir: str, sweep_name: str, legacy: bool = False) -> dict:
    """Find legacy name-based JSON runs; never use this for paper ingestion."""
    if not legacy:
        return {}
    pattern = re.compile(rf"{re.escape(sweep_name)}_depth(\d+)")
    depth_folders = {}

    for entry in os.scandir(benchmark_dir):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        depth = int(m.group(1))
        legacy_files = sorted([
            f.path for f in os.scandir(entry.path)
            if re.match(r"results_seed_\d+\.json", f.name)
        ])
        if legacy_files:
            depth_folders[depth] = {
                "path": entry.path,
                "results": legacy_files,
            }
            continue
    return depth_folders


def discover_manifest_runs(
    benchmark_dir: str | Path, sweep_name: str,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Discover v2 runs from config metadata, never from directory names."""
    root = Path(benchmark_dir)
    runs = []
    violations = []
    for config_path in sorted(root.rglob("config.json")):
        try:
            config = _load_json(config_path)
        except ValueError as exc:
            violations.append((str(config_path.parent), str(exc)))
            continue
        if config.get("sweep_name") != sweep_name:
            continue
        version = config.get("benchmark_schema_version", "")
        if version != EPISODE_SCHEMA_VERSION:
            violations.append((str(config_path.parent), str(version)))
            continue
        runs.append({"path": config_path.parent, "config": config})
    return runs, violations


def validate_manifest_runs(
    candidates: list[dict], *, require_paper_grade: bool,
) -> tuple[list[dict], list[dict]]:
    """Validate, key, and deduplicate manifest runs by experimental identity."""
    validated = []
    twins = []
    by_key: dict[tuple[str, int, int], dict] = {}
    for candidate in candidates:
        validator = validate_paper_run if require_paper_grade else validate_benchmark_run
        run = validator(candidate["path"])
        config = run["config"]
        depth = config.get("depth")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth not in range(4):
            raise ValueError(
                f"{candidate['path']}: depth must be one of 0, 1, 2, 3"
            )
        condition = config.get("condition")
        seed = config.get("training_seed")
        if require_paper_grade:
            valid_condition = (
                isinstance(condition, str) and condition in {"d4", "no_d4"}
            )
            valid_seed = (
                not isinstance(seed, bool)
                and isinstance(seed, int)
                and seed in range(4)
            )
        else:
            valid_condition = isinstance(condition, str) and bool(condition)
            valid_seed = not isinstance(seed, bool) and isinstance(seed, int)
        if not valid_condition:
            raise ValueError(f"{candidate['path']}: condition is invalid")
        if not valid_seed:
            raise ValueError(f"{candidate['path']}: training_seed is invalid")
        key = (condition, seed, depth)
        existing = by_key.get(key)
        item = {**run, "path": Path(candidate["path"]), "key": key}
        if existing is None:
            by_key[key] = item
            validated.append(item)
            continue
        if existing["config"]["outcome_fingerprint"] == config["outcome_fingerprint"]:
            if not _same_twin_identity(existing["config"], config):
                raise ValueError(
                    "duplicate experimental identity has the same outcome but "
                    f"different provenance: {key} ({existing['path']} vs {candidate['path']})"
                )
            item["twin_of"] = str(existing["path"])
            twins.append(item)
            continue
        raise ValueError(
            "duplicate experimental identity with different outcome: "
            f"{key} ({existing['path']} vs {candidate['path']})"
        )
    if require_paper_grade:
        paired: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for run in validated:
            paired[(run["config"]["training_seed"], run["config"]["depth"])].append(run)
        for identity, pair in paired.items():
            if {run["config"].get("condition") for run in pair} != {"d4", "no_d4"}:
                raise ValueError(
                    "paper aggregation requires d4/no_d4 pairs for "
                    f"training_seed={identity[0]}, depth={identity[1]}"
                )
            _validate_paired_runs(pair)
    return validated, twins


def load_result_file(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_win_rates(raw_data: list, thresholds: list) -> dict:
    """Compute win rate (fraction >= threshold) for each threshold."""
    n = len(raw_data)
    result = {}
    for t in thresholds:
        result[f"win_rate_{t}"] = (sum(1 for r in raw_data if r["max_tile"] >= t) / n) if n else 0.0
    return result


def compute_max_tile_eq_pct(raw_data: list, thresholds: list) -> dict:
    """Compute fraction of episodes where max_tile == threshold."""
    n = len(raw_data)
    result = {}
    for t in thresholds:
        result[f"max_tile_eq_{t}_pct"] = (sum(1 for r in raw_data if r["max_tile"] == t) / n) if n else 0.0
    return result


def aggregate_depth(depth_results: list) -> dict:
    """Aggregate legacy model results across training seeds."""
    avg_scores = [r["metrics"]["avg_score"] for r in depth_results]
    std_scores = [r["metrics"]["std_score"] for r in depth_results]
    min_scores = [r["metrics"]["min_score"] for r in depth_results]
    max_scores = [r["metrics"]["max_score"] for r in depth_results]
    avg_steps = [r["metrics"]["avg_steps"] for r in depth_results]

    n_seeds = len(depth_results)
    n_episodes = depth_results[0]["config"]["n_runs"]

    result = {
        "mean_score": np.mean(avg_scores),
        "std_score": sample_sd(avg_scores),
        "min_score": np.mean(min_scores),
        "max_score": np.mean(max_scores),
        "mean_steps": np.mean(avg_steps),
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
    }

    # Aggregate search metrics if present
    search_keys = ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]
    for key in search_keys:
        vals = [r["metrics"][key] for r in depth_results if key in r["metrics"]]
        if vals:
            result[key] = np.mean(vals)

    return result


def compute_ci(mean: float, std: float, n: int) -> tuple:
    """Compute a model-level 95% Student-t CI for legacy results."""
    if n < 2:
        return float("nan"), float("nan")
    margin = student_t_critical_95(n - 1) * std / np.sqrt(n)
    return mean - margin, mean + margin


def _pstdev(values):
    """Population standard deviation. Used for per-seed std in summary rows."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def load_episodes_csv(run_dir):
    """Load and strictly validate episodes.csv from a run folder."""
    csv_path = Path(run_dir) / "episodes.csv"
    if not csv_path.exists():
        return []
    return _load_episode_rows(csv_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=str, help="Root folder containing manifest-bound benchmark runs")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep name to aggregate")
    parser.add_argument("--win-threshold", type=int, default=None, help="Report single win threshold (default: all observed)")
    parser.add_argument("--output", type=str, default=None, help="Override output directory (default: benchmark_dir)")
    parser.add_argument("--legacy", action="store_true",
                        help="Read legacy results_seed_N.json files instead of the new CSV layout.")
    parser.add_argument("--paper-mode", "--strict", dest="paper_mode", action="store_true",
                        help="Reject incomplete, non-paper-grade, or unpaired result artifacts.")
    return parser.parse_args(argv)


_MODEL_METRIC_FIELDS = [
    "analysis_unit", "sweep_name", "condition", "training_seed", "depth", "model_path",
    "model_sha256", "n_episodes", "mean_score", "mean_steps", "mean_max_tile",
    "win_rate_1024", "win_rate_2048", "win_rate_4096", "win_rate_8192",
    "mean_total_think_ms", "mean_total_nodes",
    "mean_total_batches", "mean_tt_hit_rate", "mean_nps", "episode_uncertainty_unit",
    "episode_score_sd_descriptive", "episode_score_ci95_low_descriptive",
    "episode_score_ci95_high_descriptive",
    "episode_win_rate_2048_ci95_low_descriptive",
    "episode_win_rate_2048_ci95_high_descriptive",
    "episode_win_rate_4096_ci95_low_descriptive",
    "episode_win_rate_4096_ci95_high_descriptive",
    "episode_win_rate_1024_ci95_low_descriptive",
    "episode_win_rate_1024_ci95_high_descriptive",
    "episode_win_rate_8192_ci95_low_descriptive",
    "episode_win_rate_8192_ci95_high_descriptive",
]
_EFFECT_FIELDS = [
    "analysis", "comparison", "condition", "depth", "depth_a", "depth_b",
    "metric", "n_models", "df", "mean_delta", "sd_delta", "ci95_low",
    "ci95_high", "unit", "ci_method",
]
_DELTA_FIELDS = [
    "analysis", "comparison", "condition", "depth", "depth_a", "depth_b",
    "training_seed", "metric", "delta", "unit",
]
_MODEL_CI_FIELDS = [
    "analysis", "condition", "depth", "metric", "n_models", "df", "mean", "sd",
    "ci95_low", "ci95_high", "unit", "ci_method",
]
_TWIN_FIELDS = [
    "path", "twin_of", "condition", "training_seed", "depth", "outcome_fingerprint",
]


def _write_table(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _with_ci_metadata(rows: list[dict]) -> list[dict]:
    return [{**row, "ci_method": "student_t_95"} for row in rows]


def _write_manifest_figures(model_metrics: list[dict], output_dir: Path) -> None:
    """Write descriptive figures from model-level estimates, never pooled episodes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir = output_dir / "paper_figures"
    figures_dir.mkdir(exist_ok=True)
    depths = sorted({model["depth"] for model in model_metrics})
    for depth in depths:
        models = sorted(
            [model for model in model_metrics if model["depth"] == depth],
            key=lambda model: (model["condition"], model["training_seed"]),
        )
        labels = [f"{model['condition']}/s{model['training_seed']}" for model in models]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, [model["mean_score"] for model in models], color="steelblue")
        ax.set_xlabel("Condition / training seed (model unit)")
        ax.set_ylabel("Mean score per model")
        ax.set_title(f"Model Mean Scores - Depth {depth}")
        fig.autofmt_xdate()
        fig.savefig(figures_dir / f"model_score_depth{depth}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        values = [model["win_rate_2048"] for model in models]
        ax.bar(labels, values, color="steelblue", alpha=0.7)
        ax.axhline(float(math.fsum(values) / len(values)), color="orange", linestyle="--")
        ax.set_xlabel("Condition / training seed (model unit)")
        ax.set_ylabel("Win rate 2048+ per model")
        ax.set_title(f"Model Win Rates - Depth {depth}")
        fig.autofmt_xdate()
        fig.savefig(figures_dir / f"bar_winrate_depth{depth}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    heatmap = np.array([
        [
            float(math.fsum(
                model["mean_max_tile"]
                for model in model_metrics
                if model["condition"] == condition and model["depth"] == depth
            ) / sum(
                1 for model in model_metrics
                if model["condition"] == condition and model["depth"] == depth
            ))
            for depth in depths
        ]
        for condition in ("d4", "no_d4")
    ])
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(depths)), labels=[f"Depth {depth}" for depth in depths])
    ax.set_yticks(range(2), labels=["d4", "no_d4"])
    ax.set_xlabel("Search depth")
    ax.set_ylabel("Condition")
    ax.set_title("Mean Max Tile by Training Model")
    fig.colorbar(image, ax=ax, label="Mean max tile per model")
    fig.savefig(figures_dir / "heatmap_max_tile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _aggregate_manifest_statistics(
    manifest_runs: list[dict],
    twins: list[dict],
    output_dir: Path,
    sweep_name: str,
) -> int:
    """Write model-level and paired seed-level statistics for CSV runs."""
    model_metrics = [
        compute_model_metrics(run["config"], run["episodes"])
        for run in manifest_runs
    ]
    validate_complete_design(
        model_metrics, expected_seeds=EXPECTED_TRAINING_SEEDS,
    )
    d4_effects, d4_deltas = paired_d4_no_d4_effects(
        model_metrics, expected_seeds=EXPECTED_TRAINING_SEEDS,
    )
    model_cis = model_level_confidence_intervals(
        model_metrics, expected_seeds=EXPECTED_TRAINING_SEEDS,
    )
    depth_effects, depth_deltas = cross_depth_effects(
        model_metrics, expected_seeds=EXPECTED_TRAINING_SEEDS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_table(output_dir / "per_model_metrics.csv", model_metrics, _MODEL_METRIC_FIELDS)
    # Keep summary.csv as a model-level table; it must not imply pooled episodes.
    _write_table(output_dir / "summary.csv", model_metrics, _MODEL_METRIC_FIELDS)
    _write_table(
        output_dir / "paired_d4_no_d4_seed_deltas.csv", d4_deltas, _DELTA_FIELDS,
    )
    _write_table(
        output_dir / "cross_depth_paired_seed_deltas.csv", depth_deltas, _DELTA_FIELDS,
    )
    d4_effects = _with_ci_metadata(d4_effects)
    depth_effects = _with_ci_metadata(depth_effects)
    _write_table(
        output_dir / "paired_d4_no_d4_effects.csv", d4_effects, _EFFECT_FIELDS,
    )
    _write_table(
        output_dir / "model_level_confidence_intervals.csv", model_cis, _MODEL_CI_FIELDS,
    )
    _write_table(
        output_dir / "cross_depth_paired_effects.csv", depth_effects, _EFFECT_FIELDS,
    )
    _write_table(
        output_dir / "cross_depth_ci_table.csv", depth_effects, _EFFECT_FIELDS,
    )
    _write_table(
        output_dir / "confidence_intervals.csv",
        d4_effects + depth_effects,
        _EFFECT_FIELDS,
    )
    _write_table(
        output_dir / "excluded_twins.csv",
        [
            {
                "path": str(twin["path"]),
                "twin_of": twin.get("twin_of", ""),
                "condition": twin["config"].get("condition", ""),
                "training_seed": twin["config"].get("training_seed", ""),
                "depth": twin["config"].get("depth", ""),
                "outcome_fingerprint": twin["config"].get("outcome_fingerprint", ""),
            }
            for twin in twins
        ],
        _TWIN_FIELDS,
    )
    _write_manifest_figures(model_metrics, output_dir)
    print(f"Wrote {len(model_metrics)} model-level metric rows for sweep '{sweep_name}'.")
    print(f"Wrote {len(d4_effects)} D4/No-D4 paired effects.")
    print(f"Wrote {len(depth_effects)} cross-depth paired effects.")
    print(f"Excluded {len(twins)} provenance twin artifact(s).")
    return 0


def main(argv=None):
    args = parse_args(argv)

    benchmark_dir = args.benchmark_dir
    sweep_name = args.sweep
    output_dir = Path(args.output or benchmark_dir)

    if args.paper_mode and args.legacy:
        print("Error: --paper-mode cannot ingest legacy name-driven artifacts.")
        return 2

    # New CSV runs are discovered from their metadata, not directory names.
    manifest_runs = []
    if args.legacy:
        depth_folders = discover_depth_folders(
            benchmark_dir, sweep_name, legacy=True,
        )
    else:
        candidates, schema_violations = discover_manifest_runs(
            benchmark_dir, sweep_name,
        )
        if schema_violations:
            print("Error: invalid benchmark manifest metadata in some run folders:")
            for path, version in schema_violations:
                print(f"  {path}: {version}")
            print(f"aggregate.py only accepts schema {EPISODE_SCHEMA_VERSION}.")
            return 2
        try:
            manifest_runs, twins = validate_manifest_runs(
                candidates, require_paper_grade=args.paper_mode,
            )
        except ValueError as exc:
            print(f"Error: manifest-driven aggregation rejected: {exc}")
            return 2
        if twins:
            print(f"Ignored {len(twins)} deterministic twin artifact(s).")
        depth_folders = {}
        for run in manifest_runs:
            depth = run["config"]["depth"]
            result_path = str(run["path"] / "episodes.csv")
            info = depth_folders.setdefault(
                depth,
                {"path": str(run["path"]), "results": [], "run_by_result": {}},
            )
            info["results"].append(result_path)
            info["run_by_result"][result_path] = run

    if not depth_folders:
        print(f"Error: no manifest runs found for sweep '{sweep_name}' in {benchmark_dir}")
        print("  Expected completed config.json artifacts with sweep_name metadata.")
        print("  Unmanifested or incompatible artifacts remain legacy/non-paper-grade and were not ingested.")
        return 1

    if not args.legacy:
        try:
            return _aggregate_manifest_statistics(
                manifest_runs, twins, output_dir, sweep_name,
            )
        except ValueError as exc:
            print(f"Error: statistical aggregation rejected: {exc}")
            return 2

    print(f"Found depths: {sorted(depth_folders.keys())}")
    for depth, info in sorted(depth_folders.items()):
        print(f"  depth {depth}: {len(info['results'])} seed result(s)")

    THRESHOLDS = [1024, 2048, 4096, 8192]
    REPORT_THRESHOLDS = THRESHOLDS if args.win_threshold is None else [args.win_threshold]

    # Collect all rows for summary.csv
    summary_rows = []
    cross_depth_rows = []

    for depth, info in sorted(depth_folders.items()):
        depth_results = []
        for result_path in info["results"]:
            manifest_run = info.get("run_by_result", {}).get(result_path)
            basename = os.path.basename(result_path)
            if manifest_run is not None:
                seed_n = manifest_run["config"]["training_seed"]
                condition = manifest_run["config"]["condition"]
            else:
                seed_m = re.search(r"results_seed_(\d+)\.json", basename)
                if seed_m is not None:
                    seed_n = int(seed_m.group(1))
                elif basename == "episodes.csv":
                    seed_n = 0
                else:
                    continue
                condition = "legacy"
            if basename.endswith(".json"):
                result = load_result_file(result_path)
                raw = result.get("raw_data", [])
            else:
                result = {}
                raw = []

            win_rates = compute_win_rates(raw, REPORT_THRESHOLDS)
            tile_eq_pcts = compute_max_tile_eq_pct(raw, THRESHOLDS)

            row = {
                "sweep_name": sweep_name,
                "depth": depth,
                "seed": seed_n,
                "condition": condition,
            }
            run_dir = os.path.dirname(result_path)
            episodes = (
                manifest_run["episodes"]
                if manifest_run is not None
                else load_episodes_csv(run_dir) if not args.legacy else []
            )

            if episodes:
                # New CSV layout
                scores = [int(r["score"]) for r in episodes]
                steps_ = [int(r["steps"]) for r in episodes]
                max_tiles = [int(r["max_tile"]) for r in episodes]
                row.update({
                    "avg_score": float(sum(scores) / len(scores)),
                    "std_score": float(_pstdev(scores)),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "avg_steps": float(sum(steps_) / len(steps_)),
                })
                # Include search-mode metrics from episodes.csv when present.
                for key, col in [
                    ("avg_think_ms", "total_think_ms"),
                    ("avg_nodes_visited", "total_nodes"),
                    ("avg_batches_eval", "total_batches"),
                    ("avg_tt_collisions", "total_tt_collisions"),
                    ("avg_tt_same_key_overwrites", "total_tt_same_key_overwrites"),
                    ("avg_moves_resolved", "total_moves_resolved"),
                    ("avg_moves_unresolved", "total_moves_unresolved"),
                    ("avg_cap_hits", "total_cap_hits"),
                    ("avg_chance_nodes", "total_chance_nodes"),
                    ("avg_max_nodes", "total_max_nodes"),
                    ("avg_nodes_per_sec", "mean_nps"),
                    ("avg_tt_hit_rate", "mean_tt_hit_rate"),
                    ("avg_chance_value", "mean_chance_value"),
                ]:
                    vals = [r[col] for r in episodes if r.get(col) is not None and not (isinstance(r[col], float) and (r[col] != r[col]))]
                    if vals:
                        row[key] = float(sum(vals) / len(vals))
                # Reuse the existing win-rate/tile-eq computation by passing
                # synthetic raw_data-like list.
                win_rates = compute_win_rates(
                    [{"score": s, "max_tile": mt} for s, mt in zip(scores, max_tiles)],
                    REPORT_THRESHOLDS,
                )
                tile_eq_pcts = compute_max_tile_eq_pct(
                    [{"max_tile": mt} for mt in max_tiles],
                    THRESHOLDS,
                )
            else:
                # Legacy JSON layout
                row.update({
                    "avg_score": result["metrics"]["avg_score"],
                    "std_score": result["metrics"]["std_score"],
                    "min_score": result["metrics"]["min_score"],
                    "max_score": result["metrics"]["max_score"],
                    "avg_steps": result["metrics"]["avg_steps"],
                })
                for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
                    if key in result["metrics"]:
                        row[key] = result["metrics"][key]

            row.update(win_rates)
            row.update(tile_eq_pcts)
            summary_rows.append(row)
            depth_results.append(result)

        # Cross-seed aggregate row for this depth (legacy only; new flat
        # layout contributes a single per-seed row that already encodes
        # the metrics).
        legacy_depth_results = [r for r in depth_results if r.get("metrics")]
        if legacy_depth_results:
            agg = aggregate_depth(legacy_depth_results)
            ci_lower, ci_upper = compute_ci(agg["mean_score"], agg["std_score"], agg["n_seeds"])
            agg_row = {
                "sweep_name": sweep_name,
                "depth": depth,
                "seed": "aggregate",
                "avg_score": agg["mean_score"],
                "std_score": agg["std_score"],
                "min_score": agg["min_score"],
                "max_score": agg["max_score"],
                "avg_steps": agg["mean_steps"],
            }
            # Aggregate search metrics across seeds
            for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
                vals = [r["metrics"][key] for r in legacy_depth_results if key in r["metrics"]]
                if vals:
                    agg_row[key] = round(np.mean(vals), 3 if key == "avg_think_ms" else 2 if key == "avg_tt_hit_rate" else 1)
            # Keep legacy summaries model-weighted: episodes are not independent
            # treatment replicates.
            model_win_rates = [
                compute_win_rates(res.get("raw_data", []), REPORT_THRESHOLDS)
                for res in legacy_depth_results
            ]
            model_tile_eq = [
                compute_max_tile_eq_pct(res.get("raw_data", []), THRESHOLDS)
                for res in legacy_depth_results
            ]
            agg_win_rates = {
                key: float(np.mean([rates[key] for rates in model_win_rates]))
                for key in model_win_rates[0]
            }
            agg_tile_eq = {
                key: float(np.mean([rates[key] for rates in model_tile_eq]))
                for key in model_tile_eq[0]
            }
            agg_row.update(agg_win_rates)
            agg_row.update(agg_tile_eq)
            summary_rows.append(agg_row)

            # Cross-depth CI table
            ci_row = {
                "depth": depth,
                "mean_score": agg["mean_score"],
                "std_score": agg["std_score"],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_seeds": agg["n_seeds"],
                "n_episodes": agg["n_seeds"] * agg["n_episodes"],
            }
            # Include search metrics in CI table
            for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
                if key in agg:
                    ci_row[key] = agg[key]
            cross_depth_rows.append(ci_row)

    # Write summary.csv
    csv_path = output_dir / "summary.csv"
    if summary_rows:
        fieldnames = summary_fieldnames(summary_rows)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary written to {csv_path}")

    # Write cross-depth CI table
    ci_csv_path = output_dir / "cross_depth_ci_table.csv"
    if cross_depth_rows:
        # Dynamically build fieldnames from all keys present in rows
        ci_fields = ["depth", "mean_score", "std_score", "ci_lower", "ci_upper", "n_seeds", "n_episodes"]
        for key in ["avg_think_ms", "avg_nodes_visited", "avg_batches_eval", "avg_nodes_per_sec", "avg_tt_hit_rate"]:
            if any(key in row for row in cross_depth_rows):
                ci_fields.append(key)
        with open(ci_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ci_fields)
            writer.writeheader()
            writer.writerows(cross_depth_rows)
        print(f"Cross-depth CI table written to {ci_csv_path}")

    # Generate paper figures
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        print("Warning: matplotlib not found. Skipping paper figures.")

    if MATPLOTLIB_AVAILABLE:
        figures_dir = output_dir / "paper_figures"
        figures_dir.mkdir(exist_ok=True)

        def _load_scores_for_path(result_path, manifest_run=None):
            """Return a list of {seed, scores, raw} entries for one path."""
            basename = os.path.basename(result_path)
            if basename.endswith(".json"):
                result = load_result_file(result_path)
                seed_m = re.search(r"results_seed_(\d+)\.json", basename)
                seed = seed_m.group(1) if seed_m else "0"
                raw = result.get("raw_data", [])
                scores = [r["score"] for r in raw]
                return [{
                    "seed": seed,
                    "scores": scores,
                    "mean_score": result.get("metrics", {}).get("avg_score", 0.0),
                    "raw": raw,
                }]
            if basename == "episodes.csv":
                run_dir = os.path.dirname(result_path)
                episodes = load_episodes_csv(run_dir)
                raw = [{"score": int(r["score"]), "max_tile": int(r["max_tile"])} for r in episodes]
                scores = [int(r["score"]) for r in episodes]
                if manifest_run is None:
                    seed = "0"
                    condition = "legacy"
                else:
                    seed = str(manifest_run["config"]["training_seed"])
                    condition = manifest_run["config"]["condition"]
                return [{
                    "seed": seed,
                    "condition": condition,
                    "scores": scores,
                    "mean_score": float(sum(scores) / len(scores)) if scores else 0.0,
                    "raw": raw,
                }]
            return []

        for depth, info in sorted(depth_folders.items()):
            seed_data = []
            for result_path in info["results"]:
                entries = _load_scores_for_path(
                    result_path, info.get("run_by_result", {}).get(result_path),
                )
                seed_data.extend(entries)

            if not seed_data:
                continue

            # 1. Model-level mean score plot; episodes remain descriptive only.
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(seed_data)), [d["mean_score"] for d in seed_data], color="steelblue")
            ax.set_xticks(range(len(seed_data)))
            ax.set_xticklabels([
                f"{d.get('condition', 'legacy')}/s{d['seed']}"
                for d in seed_data
            ])
            ax.set_xlabel("Condition / training seed (model unit)")
            ax.set_ylabel("Mean score per model")
            ax.set_title(f"Model Mean Scores — Depth {depth}")
            plt.savefig(figures_dir / f"model_score_depth{depth}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 2. Bar chart of win rates (per seed + aggregate)
            win_rates_per_seed = []
            for d in seed_data:
                raw = d["raw"]
                wr = sum(1 for r in raw if r["max_tile"] >= 2048) / len(raw) if raw else 0
                win_rates_per_seed.append(wr)

            agg_wr = float(np.mean(win_rates_per_seed)) if win_rates_per_seed else 0.0

            fig, ax = plt.subplots(figsize=(10, 6))
            x = list(range(len(win_rates_per_seed)))
            bars = ax.bar(x, win_rates_per_seed, color="steelblue", alpha=0.7)
            ax.axhline(agg_wr, color="orange", linestyle="--", linewidth=2, label=f"Aggregate: {agg_wr:.2%}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Seed {d['seed']}" for d in seed_data])
            ax.set_ylabel("Win Rate (>=2048)")
            ax.set_title(f"Win Rate per Seed — Depth {depth}")
            ax.legend()
            plt.savefig(figures_dir / f"bar_winrate_depth{depth}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 3. Heatmap of max tile frequency across all seeds and depths
        all_depths = sorted(depth_folders.keys())
        tile_values = [1024, 2048, 4096, 8192, 16384]
        heatmap_data = np.zeros((len(all_depths), len(tile_values)))

        for di, depth in enumerate(all_depths):
            for result_path in depth_folders[depth]["results"]:
                entries = _load_scores_for_path(result_path)
                for e in entries:
                    raw = e["raw"]
                    n = len(raw)
                    for ti, tile in enumerate(tile_values):
                        count = sum(1 for r in raw if r["max_tile"] == tile)
                        heatmap_data[di, ti] += count / n if n else 0
            if depth_folders[depth]["results"]:
                heatmap_data[di] /= len(depth_folders[depth]["results"])

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(tile_values)))
        ax.set_xticklabels(tile_values)
        ax.set_yticks(range(len(all_depths)))
        ax.set_yticklabels([f"Depth {d}" for d in all_depths])
        ax.set_xlabel("Max Tile")
        ax.set_ylabel("Configuration")
        plt.colorbar(im, ax=ax, label="Fraction of episodes")
        ax.set_title(f"Max Tile Frequency Heatmap — {sweep_name}")
        plt.savefig(figures_dir / "heatmap_max_tile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Paper figures written to {figures_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
