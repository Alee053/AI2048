import numpy as np
from numba import njit

# --- Constants for Readability ---
# These define the weights for each component of the reward signal.
MERGE_REWARD_COEF = 1.0
FREE_CELLS_COEF = 0.1
GRADIENT_COEF = 1e-4

# --- Gradients for Heuristics ---
# These define the "desirability" of each cell on the 4x4 board.
ROW_GRADIENT = np.arange(16, dtype=np.float32).reshape(4, 4)
COL_GRADIENT = ROW_GRADIENT.T

@njit
def _njit_log2_where_zero(board: np.ndarray) -> np.ndarray:
    """
    A Numba-compatible version of np.log2(board, where=board!=0).
    It calculates the log2 for non-zero elements and leaves zero for zero elements.
    """
    result = np.zeros_like(board, dtype=np.float32)
    # Numba will heavily optimize this loop into fast machine code.
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            val = board[i, j]
            if val != 0:
                result[i, j] = np.log2(val)
    return result


@njit
def calculate_reward(board: np.ndarray, merge_score: int, moved: bool) -> float:
    """
    Calculates a reward signal for a given board state and move outcome.

    This function is JIT-compiled with Numba for maximum performance.

    Args:
        board (np.ndarray): The board state *after* the move.
        merge_score (int): The sum of values of tiles merged in the move.
        moved (bool): A flag indicating if the move changed the board state.

    Returns:
        float: The calculated reward.
    """
    if not moved:
        return -1.0

    # 1. Reward for merging (log-scaled to keep values smooth)
    merge_reward = np.log2(merge_score) if merge_score > 0 else 0.0

    # 2. "Common sense" bonus for keeping the board clear
    free_cells_reward = np.sum(board == 0)

    # 3. Heuristic reward for having high-value tiles on preferred edges
    log_board = _njit_log2_where_zero(board.astype(np.float32))
    s1 = np.sum(log_board * ROW_GRADIENT)
    s2 = np.sum(log_board * COL_GRADIENT)
    gradient_reward = np.maximum(s1, s2)

    # Combine the components using the defined coefficients
    total_reward = (
            (MERGE_REWARD_COEF * merge_reward) +
            (FREE_CELLS_COEF * free_cells_reward) +
            (GRADIENT_COEF * gradient_reward)
    )
    return total_reward