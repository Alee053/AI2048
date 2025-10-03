import numpy as np
from numba import njit


@njit
def _njit_log2_where_zero(board: np.ndarray) -> np.ndarray:
    """Calculates log2 of board values, compatible with Numba and batches.

    This function serves as a Numba-jitted replacement for `np.log2(board, where=board!=0)`,
    which is not directly supported by Numba. It handles both single game boards
    (2D arrays) and batches of boards (3D arrays).

    Args:
        board (np.ndarray): A 2D or 3D numpy array representing the game board(s).
            Values are expected to be powers of 2.

    Returns:
        np.ndarray: A new numpy array of the same shape as the input, containing
        the base-2 logarithm of each non-zero element, with a float32 data type.
    """
    result = np.zeros_like(board, dtype=np.float32)

    if board.ndim == 2:  # Case for a single (4, 4) board
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                val = board[r, c]
                if val != 0:
                    result[r, c] = np.log2(val)
    elif board.ndim == 3:  # Case for a batch of (N, 4, 4) boards
        for i in range(board.shape[0]):
            for r in range(board.shape[1]):
                for c in range(board.shape[2]):
                    val = board[i, r, c]
                    if val != 0:
                        result[i, r, c] = np.log2(val)

    return result


def board_to_tensor(board: np.ndarray) -> np.ndarray:
    """Converts a 2048 game board into the tensor format for the neural network.

    This function preprocesses the raw game board into a format suitable for
    the policy network. The transformation involves three main steps:
    1.  Taking the base-2 logarithm of each tile's value.
    2.  Casting the result to integers, which will serve as indices for an
        embedding layer in the network.
    3.  Adding a "channel" dimension to make it compatible with CNN layers.

    Args:
        board (np.ndarray): The input game board. This can be a single board
            with shape (4, 4) or a batch of boards from a vectorized
            environment with shape (N, 4, 4).

    Returns:
        np.ndarray: The processed observation tensor. A single board is returned
            with shape (1, 4, 4), and a batch is returned with shape (N, 1, 4, 4).
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