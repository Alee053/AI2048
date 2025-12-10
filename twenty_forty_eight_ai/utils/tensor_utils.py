import numpy as np
from numba import njit


@njit
def _njit_log2_where_zero(board: np.ndarray) -> np.ndarray:
    """Batch-aware, Numba-optimized log2 for non-zero elements."""
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
    """Convert board(s) to tensor format."""
    # Compute log2
    log_board = _njit_log2_where_zero(board.astype(np.float32))

    # Cast to int64
    int_log_board = log_board.astype(np.int64)

    # Add channel dim
    if int_log_board.ndim == 2:  # Single board (4, 4)
        return np.expand_dims(int_log_board, axis=0)
    else:  # Batch (N, 4, 4)
        return np.expand_dims(int_log_board, axis=1)