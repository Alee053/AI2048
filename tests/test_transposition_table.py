"""Tests for the cross-search aging behavior of the TranspositionTable.

The cross-search eviction regression is verified end-to-end by
tests/stress_depth4_real.py --cumulative 30, which uses a real CNN and
actually stresses the TT. A unit-test version with fake_batch_eval cannot
reliably fire the eviction path: with 10 random boards the bucket fill
rate is ~0.05/bucket, so the store() collision path is never reached
(tt_collisions == 0 across the whole test run) and tt_size grows
monotonically whether the fix is present or not. The test below is a
baseline check that clear_tt() resets the working set to cold-state
behavior.
"""
import numpy as np
import pytest

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher


def fake_batch_eval(boards):
    """Deterministic fake evaluator for reproducibility."""
    return [float(sum(sum(row) for row in b)) for b in boards]


def make_diverse_boards(n=10, seed=42):
    """Generate n unrelated 4x4 boards (log2 values)."""
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        # Random tile layout: each cell is 0..11 (representing 2^0..2^11)
        board = rng.integers(0, 12, size=(4, 4), dtype=np.int32)
        # Ensure at least one zero so the search has something to explore
        if not (board == 0).any():
            board[0, 0] = 0
        boards.append(board)
    return boards


class TestCrossSearchTTBounding:
    def test_clear_tt_behaves_like_cold_searcher(self):
        """clear_tt() must reset the working set so the next search behaves
        like one on a fresh searcher. This is a baseline check; the
        cross-search eviction regression is caught by the cumulative
        stress test (see tests/stress_depth4_real.py)."""
        searcher = ExpectimaxSearcher(target_batch_size=32768)
        board = make_diverse_boards(n=1)[0]
        searcher.find_best_move(board, 4, fake_batch_eval)
        assert searcher.find_best_move(board, 4, fake_batch_eval)['tt_size'] > 0

        searcher.clear_tt()
        # Re-run a search; the new search's tt_size reflects only its own writes.
        post_clear_size = searcher.find_best_move(board, 4, fake_batch_eval)['tt_size']
        # Post-clear searcher should behave like a fresh searcher.
        fresh = ExpectimaxSearcher(target_batch_size=32768)
        fresh_size = fresh.find_best_move(board, 4, fake_batch_eval)['tt_size']
        assert post_clear_size == fresh_size, (
            f"After clear_tt, size={post_clear_size:,} but cold TT size={fresh_size:,}. "
            f"clear_tt() did not actually clear the table."
        )


def test_depth_two_result_does_not_reuse_depth_three_tt_values():
    board = np.array([
        [1, 2, 0, 0],
        [0, 1, 2, 0],
        [0, 0, 1, 2],
        [2, 0, 0, 1],
    ], dtype=np.int32)

    fresh = ExpectimaxSearcher(target_batch_size=32768)
    fresh_result = fresh.find_best_move(board, 2, fake_batch_eval)

    warmed = ExpectimaxSearcher(target_batch_size=32768)
    warmed.find_best_move(board, 3, fake_batch_eval)
    warmed_result = warmed.find_best_move(board, 2, fake_batch_eval)

    assert warmed_result["best_move"] == fresh_result["best_move"]
    assert warmed_result["move_scores"] == fresh_result["move_scores"]


def test_tt_disabled_search_matches_enabled_search():
    board = np.array([
        [1, 2, 0, 0],
        [0, 1, 2, 0],
        [0, 0, 1, 2],
        [2, 0, 0, 1],
    ], dtype=np.int32)

    enabled = ExpectimaxSearcher(target_batch_size=32768)
    disabled = ExpectimaxSearcher(target_batch_size=32768, use_transposition_table=False)
    disabled.clear_tt()
    enabled_result = enabled.find_best_move(board, 2, fake_batch_eval)
    disabled_result = disabled.find_best_move(board, 2, fake_batch_eval)

    assert disabled_result["best_move"] == enabled_result["best_move"]
    assert disabled_result["move_scores"] == enabled_result["move_scores"]
    assert disabled_result["tt_lookups"] == 0
    assert disabled_result["tt_hits"] == 0
    assert disabled_result["tt_size"] == 0


def test_tt_disabled_search_uses_canonical_leaf_boards():
    board = np.array([
        [1, 2, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    def orientation_sensitive_eval(boards):
        return [
            float(sum((row * 4 + column + 1) * value
                      for row, values in enumerate(board)
                      for column, value in enumerate(values)))
            for board in boards
        ]

    enabled = ExpectimaxSearcher(target_batch_size=32768)
    disabled = ExpectimaxSearcher(target_batch_size=32768, use_transposition_table=False)
    enabled_result = enabled.find_best_move(board, 1, orientation_sensitive_eval)
    disabled_result = disabled.find_best_move(board, 1, orientation_sensitive_eval)

    assert disabled_result["best_move"] == enabled_result["best_move"]
    assert disabled_result["move_scores"] == enabled_result["move_scores"]
    assert disabled_result["tt_lookups"] == 0
    assert disabled_result["tt_hits"] == 0
    assert disabled_result["tt_size"] == 0


def test_exact_depth_entries_coexist_after_depth_three_depth_two_depth_three_searches():
    board = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 0],
    ], dtype=np.int32)
    searcher = ExpectimaxSearcher(target_batch_size=32768)

    first_depth_three = searcher.find_best_move(board, 3, fake_batch_eval)
    depth_two = searcher.find_best_move(board, 2, fake_batch_eval)
    second_depth_two = searcher.find_best_move(board, 2, fake_batch_eval)
    second_depth_three = searcher.find_best_move(board, 3, fake_batch_eval)

    assert depth_two["tt_size"] > first_depth_three["tt_size"]
    assert second_depth_two["nodes_visited"] == depth_two["moves_resolved"]
    assert second_depth_two["tt_hits"] == depth_two["moves_resolved"]
    assert second_depth_three["nodes_visited"] == first_depth_three["moves_resolved"]
    assert second_depth_three["tt_hits"] == first_depth_three["moves_resolved"]
