import numpy as np

CORNER_GRADIENT = np.array([
    [15, 14, 13, 12],
    [14, 10, 9, 8],
    [13, 9, 4, 3],
    [12, 8, 3, 0]
], dtype=np.float32)


def _calculate_potential(board: np.ndarray) -> float:
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))
    return np.sum(log_board * CORNER_GRADIENT)


def calculate_reward(board: np.ndarray, prev_board: np.ndarray, merge_score: int, moved: bool) -> float:
    if not moved:
        return -1.0

    merge_reward = np.log2(merge_score) if merge_score > 0 else 0.0

    potential_new = _calculate_potential(board)
    potential_old = _calculate_potential(prev_board)
    potential_reward = potential_new - potential_old

    free_cells_reward = np.sum(board == 0) * 0.1

    return merge_reward + potential_reward + free_cells_reward