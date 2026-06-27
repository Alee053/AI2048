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