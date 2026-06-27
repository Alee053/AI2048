"""Tests for scripts.benchmark_io — schema, dataclasses, writers."""
from __future__ import annotations

import pickle

import pytest


def test_schema_version_is_semver():
    from scripts.benchmark_io import EPISODE_SCHEMA_VERSION

    parts = EPISODE_SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_episode_columns_is_nonempty_list_of_str():
    from scripts.benchmark_io import EPISODE_COLUMNS

    assert isinstance(EPISODE_COLUMNS, list)
    assert len(EPISODE_COLUMNS) > 10
    assert all(isinstance(c, str) for c in EPISODE_COLUMNS)
    assert "schema_version" in EPISODE_COLUMNS
    assert "score" in EPISODE_COLUMNS
    assert "max_tile" in EPISODE_COLUMNS


def test_move_columns_is_nonempty_list_of_str():
    from scripts.benchmark_io import MOVE_COLUMNS

    assert isinstance(MOVE_COLUMNS, list)
    assert len(MOVE_COLUMNS) > 10
    assert all(isinstance(c, str) for c in MOVE_COLUMNS)
    assert "score_up" in MOVE_COLUMNS
    assert "score_left" in MOVE_COLUMNS


def test_episode_columns_no_duplicates():
    from scripts.benchmark_io import EPISODE_COLUMNS

    assert len(EPISODE_COLUMNS) == len(set(EPISODE_COLUMNS))


def test_move_columns_no_duplicates():
    from scripts.benchmark_io import MOVE_COLUMNS

    assert len(MOVE_COLUMNS) == len(set(MOVE_COLUMNS))


def test_episode_result_is_picklable():
    from scripts.benchmark_io import EpisodeResult, MOVE_COLUMNS

    result = EpisodeResult(
        schema_version="1.0.0",
        run_id="test-run",
        episode_idx=0,
        worker_id=0,
        train_seed=None,
        eval_seed=42,
        requested_depth=3,
        effective_depth=3,
        use_expectimax=True,
        score=1000,
        max_tile=128,
        max_log_tile=7,
        steps=50,
        episode_time_s=5.0,
        mean_move_time_ms=100.0,
        median_move_time_ms=95.0,
        p95_move_time_ms=150.0,
        max_move_time_ms=200.0,
        termination_reason="board_full",
        win_1024=False,
        win_2048=False,
        win_4096=False,
        win_8192=False,
        total_think_ms=1900.0,
        total_nodes=10000,
        total_batches=10,
        total_tt_lookups=5000,
        total_tt_hits=200,
        total_tt_collisions=1,
        total_tt_same_key_overwrites=0,
        total_moves_resolved=50,
        total_moves_unresolved=0,
        total_cap_hits=0,
        total_alpha_beta_cuts=5,
        total_chance_nodes=200,
        total_max_nodes=100,
        mean_chance_value=0.1,
        mean_empty_cells=8.0,
        min_empty_cells=3,
        mean_merge_score=20.0,
        mean_nps=5263.0,
        mean_tt_hit_rate=0.04,
        mean_nodes_per_batch_call=1000.0,
        move_records=[],
    )
    blob = pickle.dumps(result)
    restored = pickle.loads(blob)
    assert restored.score == 1000
    assert restored.episode_idx == 0


def test_move_record_is_picklable():
    from scripts.benchmark_io import MoveRecord

    rec = MoveRecord(
        schema_version="1.0.0",
        run_id="test-run",
        episode_idx=0,
        move_idx=0,
        worker_id=0,
        action=0,
        action_name="UP",
        canonical_board_hash="12345",
        board_state="0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0",
        empty_cells_before=14,
        max_tile_before=2,
        max_log_tile_before=1,
        merge_score=0,
        move_time_ms=10.0,
        think_ms=8.0,
        nodes_visited=100,
        batches_eval=1,
        nodes_per_batch_call=100.0,
        tt_hit_rate=0.0,
        tt_lookups=10,
        tt_hits=0,
        n_legal_actions=3,
        score_up=0.5,
        score_right=0.3,
        score_down=0.2,
        score_left=0.1,
        is_unresolved=False,
        is_cap_hit=False,
        best_move=0,
    )
    blob = pickle.dumps(rec)
    restored = pickle.loads(blob)
    assert restored.action == 0
    assert restored.score_up == 0.5


def test_episode_to_row_keys_match_episode_columns():
    from scripts.benchmark_io import (
        EpisodeResult, episode_to_row, EPISODE_COLUMNS,
    )

    result = EpisodeResult(
        schema_version="1.0.0",
        run_id="r", episode_idx=0, worker_id=0, train_seed=None,
        eval_seed=42, requested_depth=3, effective_depth=3,
        use_expectimax=True, score=1000, max_tile=128, max_log_tile=7,
        steps=50, episode_time_s=5.0, mean_move_time_ms=100.0,
        median_move_time_ms=95.0, p95_move_time_ms=150.0,
        max_move_time_ms=200.0, termination_reason="board_full",
        win_1024=False, win_2048=False, win_4096=False, win_8192=False,
        total_think_ms=1900.0, total_nodes=10000, total_batches=10,
        total_tt_lookups=5000, total_tt_hits=200, total_tt_collisions=1,
        total_tt_same_key_overwrites=0, total_moves_resolved=50,
        total_moves_unresolved=0, total_cap_hits=0,
        total_alpha_beta_cuts=5, total_chance_nodes=200,
        total_max_nodes=100, mean_chance_value=0.1,
        mean_empty_cells=8.0, min_empty_cells=3, mean_merge_score=20.0,
        mean_nps=5263.0, mean_tt_hit_rate=0.04,
        mean_nodes_per_batch_call=1000.0, move_records=[],
    )
    row = episode_to_row(result)
    assert set(row.keys()) == set(EPISODE_COLUMNS)
    assert row["score"] == 1000
    assert row["max_tile"] == 128
    assert row["schema_version"] == "1.0.0"


def test_move_to_row_keys_match_move_columns():
    from scripts.benchmark_io import (
        MoveRecord, move_to_row, MOVE_COLUMNS,
    )

    rec = MoveRecord(
        schema_version="1.0.0", run_id="r", episode_idx=0, move_idx=0,
        worker_id=0, action=0, action_name="UP",
        canonical_board_hash="12345",
        board_state="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
        empty_cells_before=16, max_tile_before=1, max_log_tile_before=0,
        merge_score=0, move_time_ms=10.0, think_ms=0.0,
        nodes_visited=0, batches_eval=0, nodes_per_batch_call=0.0,
        tt_hit_rate=0.0, tt_lookups=0, tt_hits=0,
        n_legal_actions=4, score_up=float("nan"), score_right=float("nan"),
        score_down=float("nan"), score_left=float("nan"),
        is_unresolved=False, is_cap_hit=False, best_move=0,
    )
    row = move_to_row(rec)
    assert set(row.keys()) == set(MOVE_COLUMNS)
    assert row["action"] == 0
    assert row["score_up"] != row["score_up"]  # NaN check