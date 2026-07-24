"""Test that depth-4 search converges without hitting MAX_PASSES."""
import numpy as np
import pytest

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher


def fake_batch_eval(boards):
    """Deterministic fake evaluator for reproducibility."""
    return [float(sum(sum(row) for row in b)) for b in boards]


def make_complex_board():
    """Return a board state that previously triggered MAX_PASSES."""
    return np.array([
        [2, 4, 2, 0],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ], dtype=np.int32)


class TestDepth4Convergence:
    def test_depth_4_converges(self):
        """Depth-4 search should resolve all moves without MAX_PASSES warning."""
        searcher = ExpectimaxSearcher(target_batch_size=32768)
        board = make_complex_board()

        stats = searcher.find_best_move(board, 4, fake_batch_eval)

        assert stats['batches_eval'] > 0
        assert stats['batches_eval'] <= 20, (
            f"Depth-4 search used {stats['batches_eval']} batches, "
            "expected ≤ 20 with per-move batching"
        )
        assert stats['cap_hits'] == 0
        assert stats['moves_unresolved'] == 0
