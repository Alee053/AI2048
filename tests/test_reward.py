import numpy as np
import pytest

from twenty_forty_eight_ai.env.d4_transforms import apply_d4, NUM_TRANSFORMS
from twenty_forty_eight_ai.env.reward import (
    D4_GRADIENTS,
    FREE_CELLS_COEF,
    GRADIENT_COEF,
    MERGE_REWARD_COEF,
    _calculate_gradient_reward,
    calculate_reward,
)


def _expected_gradient(board: np.ndarray) -> float:
    board_float = board.astype(np.float32)
    return max(float(np.sum(board_float * gradient)) for gradient in D4_GRADIENTS)


def test_reward_precomputes_eight_d4_gradient_templates():
    assert D4_GRADIENTS.shape == (NUM_TRANSFORMS, 4, 4)
    assert len({tuple(gradient.ravel()) for gradient in D4_GRADIENTS}) == NUM_TRANSFORMS


@pytest.mark.parametrize(
    ("exponent", "expected"),
    [
        (0, 0.0),
        (1, 15.0),
        (15, 225.0),
    ],
)
def test_gradient_uses_board_exponents_directly(exponent, expected):
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, 0] = exponent

    assert _calculate_gradient_reward(board) == pytest.approx(expected)


def test_gradient_does_not_apply_a_second_log2():
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, 0] = 4

    assert _calculate_gradient_reward(board) == pytest.approx(60.0)


def test_gradient_is_invariant_under_all_d4_transforms():
    board = np.array(
        [
            [1, 2, 0, 3],
            [4, 0, 5, 6],
            [0, 7, 8, 0],
            [9, 10, 0, 11],
        ],
        dtype=np.int32,
    )
    expected = _expected_gradient(board)

    for transform in range(NUM_TRANSFORMS):
        transformed = apply_d4(board, transform)
        assert _calculate_gradient_reward(transformed) == pytest.approx(expected)


@pytest.mark.parametrize("merge_score", [0, 4, 8, 12])
def test_merge_reward_remains_log_scaled(merge_score):
    board = np.zeros((4, 4), dtype=np.int32)
    expected_merge = np.log2(merge_score) if merge_score > 0 else 0.0
    expected = (
        MERGE_REWARD_COEF * expected_merge
        + FREE_CELLS_COEF * 16
    )

    assert calculate_reward(board, merge_score, True) == pytest.approx(expected)


def test_free_cell_bonus_remains_count_based():
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, 0] = 1
    expected = (
        FREE_CELLS_COEF * np.sum(board == 0)
        + GRADIENT_COEF * _expected_gradient(board)
    )

    assert calculate_reward(board, 0, True) == pytest.approx(expected)


def test_invalid_move_reward_remains_minus_one():
    board = np.full((4, 4), 15, dtype=np.int32)

    assert calculate_reward(board, 999, False) == -1.0


def test_complete_reward_is_d4_invariant():
    board = np.array(
        [
            [1, 2, 0, 3],
            [4, 0, 5, 6],
            [0, 7, 8, 0],
            [9, 10, 0, 11],
        ],
        dtype=np.int32,
    )
    expected = calculate_reward(board, 12, True)

    for transform in range(NUM_TRANSFORMS):
        transformed = apply_d4(board, transform)
        assert calculate_reward(transformed, 12, True) == pytest.approx(expected)
