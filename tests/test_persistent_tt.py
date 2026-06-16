"""Extended tests for the persistent transposition table architecture."""
import numpy as np
import pytest

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher


def fake_batch_eval(boards):
    """Deterministic fake evaluator for reproducibility."""
    return [float(sum(sum(row) for row in b)) for b in boards]


def make_board():
    """Return a simple non-trivial board."""
    return np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [1, 0, 0, 2],
    ], dtype=np.int32)


class TestPersistentTT:
    def test_searcher_returns_dict(self):
        searcher = ExpectimaxSearcher()
        board = make_board()
        result = searcher.find_best_move(board, 1, fake_batch_eval)
        assert isinstance(result, dict)
        assert 'best_move' in result
        assert 'tt_size' in result
        assert 'tt_lookups' in result
        assert 'tt_hits' in result

    def test_tt_persists_across_moves(self):
        """The TT should not be cleared between find_best_move calls."""
        searcher = ExpectimaxSearcher()
        board = make_board()

        # First call — cold TT
        r1 = searcher.find_best_move(board, 2, fake_batch_eval)
        tt_size_1 = r1['tt_size']
        hits_1 = r1['tt_hits']

        # Second call on the same board — warm TT
        r2 = searcher.find_best_move(board, 2, fake_batch_eval)
        tt_size_2 = r2['tt_size']
        hits_2 = r2['tt_hits']

        # TT occupancy should stay the same (same board states explored)
        assert tt_size_2 == tt_size_1
        # Hits should be non-zero on the second call because the TT is warm
        assert hits_2 > 0

    def test_clear_tt_works(self):
        """clear_tt should reset the table."""
        searcher = ExpectimaxSearcher()
        board = make_board()

        # Warm the TT
        r1 = searcher.find_best_move(board, 2, fake_batch_eval)
        hits_warm = r1['tt_hits']

        # Clear TT
        searcher.clear_tt()

        # Cold run after clear — should have fewer or equal hits than the warm run
        r2 = searcher.find_best_move(board, 2, fake_batch_eval)
        hits_cold = r2['tt_hits']

        # The warm run benefited from entries cached during the search AND
        # entries cached from previous turns. After clearing, only intra-search
        # transpositions can hit, so hits should be <= the warm run.
        assert hits_cold <= hits_warm

    def test_same_result_with_and_without_tt_warmth(self):
        """Results must be identical whether the TT is cold or warm."""
        searcher = ExpectimaxSearcher()
        board = make_board()

        r1 = searcher.find_best_move(board, 2, fake_batch_eval)
        r2 = searcher.find_best_move(board, 2, fake_batch_eval)

        assert r1['best_move'] == r2['best_move']
        np.testing.assert_allclose(r1['move_scores'], r2['move_scores'], rtol=1e-5)

    def test_different_depths_same_board(self):
        """Deeper search should be allowed to overwrite shallower cached entries."""
        searcher = ExpectimaxSearcher()
        board = make_board()

        # Shallow search first
        r1 = searcher.find_best_move(board, 1, fake_batch_eval)
        size_1 = r1['tt_size']

        # Deeper search — should populate more entries / overwrite some
        r2 = searcher.find_best_move(board, 2, fake_batch_eval)
        size_2 = r2['tt_size']

        # Deeper search explores (at least) as many unique internal nodes,
        # so occupancy should be >= previous.
        assert size_2 >= size_1

    def test_chance_node_tt_caching(self):
        """Chance nodes are also cached; repeated calls should hit."""
        searcher = ExpectimaxSearcher()
        board = make_board()

        # Depth 2 means we hit chance nodes
        r1 = searcher.find_best_move(board, 2, fake_batch_eval)
        lookups_1 = r1['tt_lookups']

        r2 = searcher.find_best_move(board, 2, fake_batch_eval)
        hits_2 = r2['tt_hits']

        # Some of the lookups on the warm run should hit
        assert hits_2 > 0

    def test_tt_size_bounded(self):
        """The fixed-size TT must not grow unboundedly."""
        searcher = ExpectimaxSearcher()
        board = make_board()

        # Run many times on the same board
        for _ in range(10):
            searcher.find_best_move(board, 3, fake_batch_eval)

        final = searcher.find_best_move(board, 3, fake_batch_eval)
        # 2^24 entries is the hard cap, but in practice we should be far below it
        assert final['tt_size'] <= 16_777_216
