"""Schema constants, dataclasses, and CSV writers for the benchmark harness.

This module is the single source of truth for the on-disk schema. Both
benchmark.py (the producer) and aggregate.py (the consumer) read column
lists from here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --- Schema versioning (semver) ---------------------------------------------

# v2 removes the non-implemented alpha-beta diagnostic; v2.1 adds manifest binding.
EPISODE_SCHEMA_VERSION = "2.1.0"

# Informational major version; the current aggregate loader requires exact 2.1.0.
SUPPORTED_SCHEMA_MAJOR = 2


# --- Column lists (exact order matches on-disk CSV) ------------------------

EPISODE_COLUMNS: list[str] = [
    "schema_version",
    "run_id",
    "episode_idx",
    "worker_id",
    "train_seed",
    "eval_seed",
    "requested_depth",
    "effective_depth",
    "use_expectimax",
    "score",
    "max_tile",
    "max_log_tile",
    "steps",
    "episode_time_s",
    "mean_move_time_ms",
    "median_move_time_ms",
    "p95_move_time_ms",
    "max_move_time_ms",
    "termination_reason",
    "win_1024",
    "win_2048",
    "win_4096",
    "win_8192",
    "total_think_ms",
    "total_nodes",
    "total_batches",
    "total_tt_lookups",
    "total_tt_hits",
    "total_tt_collisions",
    "total_tt_same_key_overwrites",
    "total_moves_resolved",
    "total_moves_unresolved",
    "total_cap_hits",
    "total_chance_nodes",
    "total_max_nodes",
    "mean_chance_value",
    "mean_empty_cells",
    "min_empty_cells",
    "mean_merge_score",
    "mean_nps",
    "mean_tt_hit_rate",
    "mean_nodes_per_batch_call",
]


MOVE_COLUMNS: list[str] = [
    "schema_version",
    "run_id",
    "episode_idx",
    "move_idx",
    "worker_id",
    "action",
    "action_name",
    "canonical_board_hash",
    "board_state",
    "empty_cells_before",
    "max_tile_before",
    "max_log_tile_before",
    "merge_score",
    "move_time_ms",
    "think_ms",
    "nodes_visited",
    "batches_eval",
    "nodes_per_batch_call",
    "tt_hit_rate",
    "tt_lookups",
    "tt_hits",
    "n_legal_actions",
    "score_up",
    "score_right",
    "score_down",
    "score_left",
    "is_unresolved",
    "is_cap_hit",
    "best_move",
]

OUTCOME_FINGERPRINT_COLUMNS: tuple[str, ...] = (
    "episode_idx",
    "eval_seed",
    "score",
    "max_tile",
    "max_log_tile",
    "steps",
    "termination_reason",
    "win_1024",
    "win_2048",
    "win_4096",
    "win_8192",
    "total_nodes",
    "total_batches",
    "total_tt_lookups",
    "total_tt_hits",
    "total_tt_collisions",
    "total_tt_same_key_overwrites",
    "total_moves_resolved",
    "total_moves_unresolved",
    "total_cap_hits",
    "total_chance_nodes",
    "total_max_nodes",
    "mean_chance_value",
    "mean_empty_cells",
    "min_empty_cells",
    "mean_merge_score",
    "mean_tt_hit_rate",
    "mean_nodes_per_batch_call",
)


# --- Action names ----------------------------------------------------------

ACTION_NAMES: dict[int, str] = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


# --- Row converters and CSVWriter will be added in Tasks 2 and 3 ----------


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Result of one episode of 2048 gameplay.

    All counts are per-episode aggregates. Per-move details live in
    `move_records` (empty unless `log_moves=True`).
    """

    schema_version: str
    run_id: str
    episode_idx: int
    worker_id: int
    train_seed: int | None
    eval_seed: int
    requested_depth: int
    effective_depth: int
    use_expectimax: bool
    score: int
    max_tile: int
    max_log_tile: int
    steps: int
    episode_time_s: float
    mean_move_time_ms: float
    median_move_time_ms: float
    p95_move_time_ms: float
    max_move_time_ms: float
    termination_reason: str
    terminated: bool
    truncated: bool
    win_1024: bool
    win_2048: bool
    win_4096: bool
    win_8192: bool
    total_think_ms: float
    total_nodes: int
    total_batches: int
    total_tt_lookups: int
    total_tt_hits: int
    total_tt_collisions: int
    total_tt_same_key_overwrites: int
    total_moves_resolved: int
    total_moves_unresolved: int
    total_cap_hits: int
    total_chance_nodes: int
    total_max_nodes: int
    mean_chance_value: float
    mean_empty_cells: float
    min_empty_cells: int
    mean_merge_score: float
    mean_nps: float
    mean_tt_hit_rate: float
    mean_nodes_per_batch_call: float
    move_records: list["MoveRecord"] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """One player move in one episode.

    All board-snapshot fields (board_state, canonical_board_hash,
    empty_cells_before, max_tile_before, max_log_tile_before,
    n_legal_actions, score_*) are captured from the same pre-action
    board snapshot.
    """

    schema_version: str
    run_id: str
    episode_idx: int
    move_idx: int
    worker_id: int
    action: int
    action_name: str
    canonical_board_hash: str
    board_state: str
    empty_cells_before: int
    max_tile_before: int
    max_log_tile_before: int
    merge_score: int
    move_time_ms: float
    think_ms: float
    nodes_visited: int
    batches_eval: int
    nodes_per_batch_call: float
    tt_hit_rate: float
    tt_lookups: int
    tt_hits: int
    n_legal_actions: int
    score_up: float
    score_right: float
    score_down: float
    score_left: float
    is_unresolved: bool
    is_cap_hit: bool
    best_move: int


def _validate_termination_flags(terminated: Any, truncated: Any) -> None:
    """Validate the mutually exclusive Gymnasium episode outcome flags."""
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("terminated and truncated must be bool")
    if terminated and truncated:
        raise ValueError("terminated and truncated cannot both be True")


def episode_to_row(result: EpisodeResult) -> dict[str, Any]:
    """Convert an EpisodeResult to a dict matching EPISODE_COLUMNS exactly.

    `move_records` is intentionally excluded — it lives in MOVE_COLUMNS, not
    EPISODE_COLUMNS.
    """
    assert "move_records" not in EPISODE_COLUMNS, (
        "move_records is a list of MoveRecord objects; it must not be "
        "serialized into the per-episode CSV row."
    )
    try:
        terminated = result.terminated
        truncated = result.truncated
    except AttributeError as exc:
        raise ValueError(
            "episode result must include terminated and truncated"
        ) from exc
    _validate_termination_flags(terminated, truncated)
    return {k: getattr(result, k) for k in EPISODE_COLUMNS}


def move_to_row(move: MoveRecord) -> dict[str, Any]:
    """Convert a MoveRecord to a dict matching MOVE_COLUMNS exactly."""
    return {k: getattr(move, k) for k in MOVE_COLUMNS}


def outcome_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Fingerprint deterministic episode outcomes, excluding run metadata/timing."""
    canonical_rows = [
        {column: row[column] for column in OUTCOME_FINGERPRINT_COLUMNS}
        for row in sorted(rows, key=lambda row: row["episode_idx"])
    ]
    payload = json.dumps(
        canonical_rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CSVWriter:
    """Writes config.json, episodes.csv, moves.csv, summary.json to a directory.

    Thread-safe via a single internal lock. The master process owns one
    instance; workers never write directly.
    """

    def __init__(self, output_dir: Path | str, log_moves: bool) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_moves = log_moves
        self._lock = threading.Lock()

        self._episodes_path = self.output_dir / "episodes.csv"
        self._moves_path = self.output_dir / "moves.csv"
        self._config_path = self.output_dir / "config.json"
        self._summary_path = self.output_dir / "summary.json"

        self._episodes_file = open(self._episodes_path, "w", newline="")
        self._episodes_writer = csv.DictWriter(
            self._episodes_file, fieldnames=EPISODE_COLUMNS
        )
        self._episodes_writer.writeheader()
        self._episodes_file.flush()

        self._moves_file = None
        self._moves_writer = None
        if log_moves:
            try:
                self._moves_file = open(self._moves_path, "w", newline="")
                self._moves_writer = csv.DictWriter(
                    self._moves_file, fieldnames=MOVE_COLUMNS
                )
                self._moves_writer.writeheader()
                self._moves_file.flush()
            except Exception:
                self._episodes_file.close()
                raise

    def write_config(self, config: dict) -> None:
        with self._lock:
            with open(self._config_path, "w") as f:
                json.dump(config, f, sort_keys=True)

    def writerow_episode(self, row: dict) -> None:
        with self._lock:
            missing = [column for column in EPISODE_COLUMNS if column not in row]
            if missing:
                raise ValueError(
                    "episode row is missing required columns: "
                    + ", ".join(missing)
                )
            self._episodes_writer.writerow(row)
            self._episodes_file.flush()

    def writerow_moves(self, rows: list) -> None:
        if not self.log_moves or self._moves_writer is None:
            return
        with self._lock:
            for row in rows:
                self._moves_writer.writerow(row)
            self._moves_file.flush()

    def write_summary(self, summary: dict) -> None:
        with self._lock:
            with open(self._summary_path, "w") as f:
                json.dump(summary, f, sort_keys=True)

    def close(self) -> None:
        with self._lock:
            if self._episodes_file and not self._episodes_file.closed:
                self._episodes_file.close()
            if self._moves_file and not self._moves_file.closed:
                self._moves_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
