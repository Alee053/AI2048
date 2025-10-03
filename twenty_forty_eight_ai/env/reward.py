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
    """Calculates log2 for non-zero elements in a Numba-compatible way.

    This function serves as a Numba-jitted replacement for `np.log2(board, where=board!=0)`,
    which is not directly supported by Numba. It iterates through the board and
    computes the base-2 logarithm for any tile that is not zero.

    Args:
        board (np.ndarray): The game board, represented as a 2D numpy array.

    Returns:
        np.ndarray: A new 2D numpy array of the same shape as the input board,
        containing the log2 of each non-zero element.
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
    """Calculates a comprehensive reward for a given game state.

    This function computes a reward signal based on several factors: the score
    from merging tiles, the number of free cells, and a heuristic that
    encourages a monotonic gradient of tile values. The function is JIT-compiled
    with Numba for performance. A penalty is applied for invalid moves.

    Args:
        board (np.ndarray): The game board state *after* the move.
        merge_score (int): The score obtained from merging tiles in the move.
        moved (bool): A flag indicating whether the move resulted in a change
            to the board state.

    Returns:
        float: The calculated reward value. Returns -1.0 for an invalid move.
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