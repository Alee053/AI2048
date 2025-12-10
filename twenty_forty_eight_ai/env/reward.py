import numpy as np
from numba import njit

# Reward coefficients
MERGE_REWARD_COEF = 1.0
FREE_CELLS_COEF = 0.1
GRADIENT_COEF = 1e-4

# Heuristic gradients
ROW_GRADIENT = np.arange(16, dtype=np.float32).reshape(4, 4)
COL_GRADIENT = ROW_GRADIENT.T

@njit
def _njit_log2_where_zero(board: np.ndarray) -> np.ndarray:
    """Numba-optimized log2 for non-zero elements."""
    result = np.zeros_like(board, dtype=np.float32)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            val = board[i, j]
            if val != 0:
                result[i, j] = np.log2(val)
    return result


@njit
def calculate_reward(board: np.ndarray, merge_score: int, moved: bool) -> float:
    """Calculate reward signal (JIT)."""
    if not moved:
        return -1.0

    # Merge reward (log scaled)
    merge_reward = np.log2(merge_score) if merge_score > 0 else 0.0

    # Empty cell bonus
    free_cells_reward = np.sum(board == 0)

    # Gradient heuristic
    log_board = _njit_log2_where_zero(board.astype(np.float32))
    s1 = np.sum(log_board * ROW_GRADIENT)
    s2 = np.sum(log_board * COL_GRADIENT)
    gradient_reward = np.maximum(s1, s2)

    # Total reward
    total_reward = (
            (MERGE_REWARD_COEF * merge_reward) +
            (FREE_CELLS_COEF * free_cells_reward) +
            (GRADIENT_COEF * gradient_reward)
    )
    return total_reward