import numpy as np
from numba import njit

@njit
def _calculate_potential(board: np.ndarray) -> float:
    factor = 64.0

    top_left_exponent = board[0, 0]

    if top_left_exponent == 0:
        return 0.0

    top_left_value = 2.0 ** top_left_exponent

    return factor * top_left_value


@njit
def calculate_reward(board: np.ndarray, prev_board: np.ndarray, merge_score: int) -> float:

    merge_reward = float(merge_score)

    potential_new = _calculate_potential(board)
    potential_old = _calculate_potential(prev_board)
    potential_reward = potential_new - potential_old

    return merge_reward + potential_reward