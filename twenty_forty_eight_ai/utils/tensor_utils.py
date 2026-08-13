import numpy as np

from ..constants import MAX_EXPONENT


def board_to_tensor(board: np.ndarray) -> np.ndarray:
    """Convert exponent-encoded board(s) to model tensor format."""
    board_array = np.asarray(board)
    if board_array.ndim == 2:
        valid_shape = board_array.shape == (4, 4)
    elif board_array.ndim == 3:
        valid_shape = board_array.shape[1:] == (4, 4)
    else:
        valid_shape = False

    if not valid_shape:
        raise ValueError("Board must have shape (4, 4) or (N, 4, 4).")

    try:
        valid_exponents = (
            np.all(np.isfinite(board_array))
            and np.all(board_array == np.floor(board_array))
            and np.all(board_array >= 0)
            and np.all(board_array <= MAX_EXPONENT)
        )
    except (TypeError, ValueError):
        valid_exponents = False

    if not valid_exponents:
        raise ValueError(
            f"Board exponents must be integer values in the range 0..{MAX_EXPONENT}."
        )

    int_board = board_array.astype(np.int64)

    if int_board.ndim == 2:  # Single board (4, 4)
        return np.expand_dims(int_board, axis=0)
    else:  # Batch (N, 4, 4)
        return np.expand_dims(int_board, axis=1)
