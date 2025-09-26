import numpy as np
from numba import njit

@njit
def _njit_log2_where_zero(board: np.ndarray) -> np.ndarray:
    """
    A Numba-compatible version of np.log2(board, where=board!=0).
    It calculates the log2 for non-zero elements and leaves zero for zero elements.
    """
    result = np.zeros_like(board, dtype=np.float32)
    # This loop is compiled to fast machine code by Numba
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            val = board[i, j]
            if val != 0:
                result[i, j] = np.log2(val)
    return result

def board_to_tensor(board: np.ndarray) -> np.ndarray:
    """
    Converts a 2048 game board into the tensor format required by the neural network.

    This involves:
    1. Taking the log base 2 of each tile value.
    2. Casting the result to integers for the embedding layer.
    3. Adding a "channel" dimension for the CNN.

    Args:
        board (np.ndarray): The input game board. Can be a single board (4x4)
                           or a batch of boards from a VecEnv (N, 4, 4).

    Returns:
        np.ndarray: The processed observation tensor. A single board is returned
                    as (1, 4, 4), a batch is returned as (N, 1, 4, 4).
    """
    # 1. Use the fast, JIT-compiled function to get log2 values.
    log_board = _njit_log2_where_zero(board.astype(np.float32))

    # 2. Cast to integer indices for the network's embedding layer.
    int_log_board = log_board.astype(np.int64)

    # 3. Add the channel dimension required by the CNN.
    if int_log_board.ndim == 2:  # Case for a single (4, 4) board
        return np.expand_dims(int_log_board, axis=0)
    else:  # Case for a batch of boards (N, 4, 4)
        return np.expand_dims(int_log_board, axis=1)