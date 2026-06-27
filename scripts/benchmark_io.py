"""Schema constants, dataclasses, and CSV writers for the benchmark harness.

This module is the single source of truth for the on-disk schema. Both
benchmark.py (the producer) and aggregate.py (the consumer) read column
lists from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --- Schema versioning (semver) ---------------------------------------------

EPISODE_SCHEMA_VERSION = "1.0.0"

# Any major version change requires an aggregate.py migration.
SUPPORTED_SCHEMA_MAJOR = 1


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
    "total_alpha_beta_cuts",
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
    total_alpha_beta_cuts: int
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


def episode_to_row(result: EpisodeResult) -> dict[str, Any]:
    """Convert an EpisodeResult to a dict matching EPISODE_COLUMNS exactly.

    `move_records` is intentionally excluded — it lives in MOVE_COLUMNS, not
    EPISODE_COLUMNS.
    """
    assert "move_records" not in EPISODE_COLUMNS, (
        "move_records is a list of MoveRecord objects; it must not be "
        "serialized into the per-episode CSV row."
    )
    return {k: getattr(result, k) for k in EPISODE_COLUMNS}


def move_to_row(move: MoveRecord) -> dict[str, Any]:
    """Convert a MoveRecord to a dict matching MOVE_COLUMNS exactly."""
    return {k: getattr(move, k) for k in MOVE_COLUMNS}