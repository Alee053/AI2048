import numpy as np

def board_to_tensor(board):
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))

    int_log_board = log_board.astype(np.int64)

    if int_log_board.ndim == 2:
        return np.expand_dims(int_log_board, axis=0)
    else:
        return np.expand_dims(int_log_board, axis=1)