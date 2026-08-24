"""Tests for the searcher Python wrapper."""
import numpy as np
import pytest

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher

def test_expectimax_searcher_returns_dict():
    """find_best_move should return a dict, not a pybind11 object."""
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
    assert 'alpha_beta_cuts' not in result
    assert result['has_legal_move'] is True
    assert result['search_complete'] is True
    assert result['failure_reason'] is None


def test_no_legal_move_is_reported_as_a_terminal_search_outcome():
    searcher = ExpectimaxSearcher()
    board = np.array([
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ])

    result = searcher.find_best_move(board, 1, lambda boards: [0.0] * len(boards))

    assert result['best_move'] == -1
    assert result['has_legal_move'] is False
    assert result['search_complete'] is True
    assert result['failure_reason'] == 'no_legal_move'
    assert all(np.isneginf(score) for score in result['move_scores'])


def test_incomplete_root_search_has_no_action_outcome():
    searcher = ExpectimaxSearcher(target_batch_size=1)
    board = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    result = searcher.find_best_move(board, 2, lambda boards: [0.0] * len(boards))

    assert result['has_legal_move'] is True
    assert result['search_complete'] is False
    assert result['failure_reason'] == 'search_incomplete'
    assert result['best_move'] == -1
    assert result['cap_hits'] > 0


def test_depth_zero_directs_callers_to_raw_ppo_path():
    searcher = ExpectimaxSearcher()
    board = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 2]])

    with pytest.raises(ValueError, match="raw PPO"):
        searcher.find_best_move(board, 0, lambda boards: [0.0] * len(boards))


@pytest.mark.parametrize("depth", [256, 257])
def test_depth_above_tt_range_is_rejected(depth):
    searcher = ExpectimaxSearcher()
    board = np.array([
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ])

    with pytest.raises(ValueError, match="255"):
        searcher.find_best_move(board, depth, lambda boards: [0.0] * len(boards))


def test_search_rejects_out_of_range_input_before_simulation():
    searcher = ExpectimaxSearcher()
    board = np.array([[16, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    with pytest.raises(ValueError, match="search input.*0.*15"):
        searcher.find_best_move(board, 1, lambda boards: [0.0] * len(boards))


def test_search_ignores_root_moves_that_would_overflow_exponent_contract():
    searcher = ExpectimaxSearcher()
    board = np.array([[15, 15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    result = searcher.find_best_move(board, 1, lambda boards: [0.0] * len(boards))

    assert result["best_move"] == 2
    assert np.isneginf(result["move_scores"][0])
    assert np.isneginf(result["move_scores"][1])
    assert np.isneginf(result["move_scores"][3])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_root_ordering_callback_rejects_non_finite_values(value):
    searcher = ExpectimaxSearcher()
    board = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    with pytest.raises(ValueError, match="finite"):
        searcher.find_best_move(board, 1, lambda boards: [value] * len(boards))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_leaf_callback_rejects_non_finite_values(value):
    searcher = ExpectimaxSearcher()
    board = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    callback_calls = 0

    def non_finite_leaf_callback(boards):
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            return [0.0] * len(boards)
        return [value] * len(boards)

    with pytest.raises(ValueError, match="finite"):
        searcher.find_best_move(board, 1, non_finite_leaf_callback)


def test_target_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="target_batch_size.*positive"):
        ExpectimaxSearcher(0)


def test_root_ordering_callback_must_return_one_value_per_board():
    searcher = ExpectimaxSearcher()
    board = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    with pytest.raises(ValueError, match="expected 2.*got 1"):
        searcher.find_best_move(board, 1, lambda boards: [0.0])


def test_leaf_callback_must_return_one_value_per_board():
    searcher = ExpectimaxSearcher()
    board = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    callback_calls = 0

    def short_leaf_callback(boards):
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            return [0.0] * len(boards)
        return [0.0] * (len(boards) - 1)

    with pytest.raises(ValueError, match="expected .* got"):
        searcher.find_best_move(board, 1, short_leaf_callback)


def test_search_rejects_non_finite_arithmetic_results():
    searcher = ExpectimaxSearcher(use_transposition_table=False)
    board = np.array([
        [1, 1, 2, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    max_float32 = float(np.finfo(np.float32).max)

    with pytest.raises(ValueError, match="non-finite"):
        searcher.find_best_move(
            board,
            2,
            lambda boards: [max_float32] * len(boards),
        )


@pytest.mark.parametrize("use_transposition_table", [True, False])
def test_tied_root_scores_choose_lower_action_index(use_transposition_table):
    searcher = ExpectimaxSearcher(use_transposition_table=use_transposition_table)
    board = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    result = searcher.find_best_move(board, 1, lambda boards: [0.0] * len(boards))

    assert result["best_move"] == 1
