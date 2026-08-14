import numpy as np
from numba import njit

from .d4_transforms import NUM_TRANSFORMS, apply_d4

# Reward coefficients
MERGE_REWARD_COEF = 1.0
FREE_CELLS_COEF = 0.1
GRADIENT_COEF = 1e-4

# Heuristic gradients
ROW_GRADIENT = np.arange(16, dtype=np.float32).reshape(4, 4)
COL_GRADIENT = ROW_GRADIENT.T
D4_GRADIENTS = np.stack(
    [apply_d4(ROW_GRADIENT, transform) for transform in range(NUM_TRANSFORMS)]
).astype(np.float32)


@njit
def _calculate_gradient_reward(board: np.ndarray) -> float:
    """Calculate the best D4-oriented gradient over exponent values."""
    board_float = board.astype(np.float32)
    gradient_reward = 0.0
    for transform in range(D4_GRADIENTS.shape[0]):
        weighted_sum = 0.0
        for row in range(4):
            for col in range(4):
                weighted_sum += board_float[row, col] * D4_GRADIENTS[
                    transform, row, col
                ]
        if weighted_sum > gradient_reward:
            gradient_reward = weighted_sum
    return gradient_reward


@njit
def calculate_reward(board: np.ndarray, merge_score: int, moved: bool) -> float:
    """Calculate reward signal (JIT)."""
    if not moved:
        return -1.0

    # Merge reward (log scaled)
    merge_reward = np.log2(merge_score) if merge_score > 0 else 0.0

    # Empty cell bonus
    free_cells_reward = np.sum(board == 0)

    # Gradient heuristic over the board's exponent values
    gradient_reward = _calculate_gradient_reward(board)

    # Total reward
    total_reward = (
            (MERGE_REWARD_COEF * merge_reward) +
            (FREE_CELLS_COEF * free_cells_reward) +
            (GRADIENT_COEF * gradient_reward)
    )
    return total_reward
