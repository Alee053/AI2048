"""Tests for the searcher Python wrapper."""
import numpy as np

def test_expectimax_searcher_returns_dict():
    """find_best_move should return a dict, not a pybind11 object."""
    from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher

    searcher = ExpectimaxSearcher()
    board = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 2]])

    def fake_batch_eval(boards):
        return [0.0] * len(boards)

    result = searcher.find_best_move(board, 1, fake_batch_eval)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'best_move' in result
    assert 'think_ms' in result
    assert 'nodes_visited' in result
    assert 'batches_eval' in result
    assert 'move_scores' in result
    assert 'tt_size' in result
    assert 'tt_lookups' in result
    assert 'tt_hits' in result