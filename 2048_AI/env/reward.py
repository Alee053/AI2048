import numpy as np
ROW_GRADIENT = np.arange(16, dtype=np.float32).reshape(4, 4)
COL_GRADIENT = ROW_GRADIENT.T
def calculate_reward(board, merge_score):
    merge_reward = np.log2(merge_score) if merge_score > 0 else 0.0

    free_cells = np.sum(board == 0)
    free_cells_reward = free_cells

    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))

    s1 = np.sum(log_board * ROW_GRADIENT)
    s2 = np.sum(log_board * COL_GRADIENT)

    gradient_reward = np.maximum(s1, s2)

    reward = merge_reward * 1 + \
              free_cells_reward * 0.1 + \
              gradient_reward * 1e-4

    return reward