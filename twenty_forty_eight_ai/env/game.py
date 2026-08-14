import numpy as np
from numba import njit

from ..constants import MAX_EXPONENT

# Constants
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

class Fast2048:
    """Optimized LUT-based 2048 game logic."""
    move_row_LUT: list = []
    move_reward_LUT: list = []
    move_valid_LUT: list = []

    def __init__(self, seed: int | np.random.SeedSequence | None = None):
        """Initialize game and LUTs."""
        if not Fast2048.move_row_LUT:
            self.init_LUT()

        self._rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.max_tile: int = 0
        self.score: int = 0
        self.done: bool = False

        self.reset()

    def init_LUT(self):
        """Pre-compute row move LUTs."""
        for i in range(65536):
            original_row = np.array([(i >> (j * 4)) & 0xF for j in range(4)], dtype=np.int32)
            row = original_row.copy()

            row = stack_row(row)
            row, reward = merge_row(row)
            row = stack_row(row)

            Fast2048.move_row_LUT.append(row)
            Fast2048.move_reward_LUT.append(reward)
            Fast2048.move_valid_LUT.append(not np.array_equal(original_row, row))

    def reset(self, seed: int | np.random.SeedSequence | None = None):
        """Reset game state, optionally restarting the tile RNG stream."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.board.fill(0)
        self.max_tile = 0
        self.score = 0
        self.done = False
        self.generate_random()
        self.generate_random()
        self.update_max_tile()

    def update_max_tile(self) -> bool:
        """Update max_tile."""
        self.max_tile = np.max(self.board)

    def is_move_valid(self, action):
        """Check move validity."""
        _raise_if_invalid_board(self.board)
        if board_would_overflow(self.board, action):
            return False

        if action == LEFT:
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[i])]: return True
        elif action == RIGHT:
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[i, ::-1])]: return True
        elif action == UP:
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[:, i])]: return True
        elif action == DOWN:
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[::-1, i])]: return True
        return False

    def generate_random(self):
        """Spawn random tile."""
        num = 1 if self._rng.random() < 0.9 else 2
        empty_cells = np.argwhere(self.board == 0)

        if empty_cells.size == 0:
            return

        chosen_position = empty_cells[self._rng.choice(len(empty_cells))]

        self.board[chosen_position[0], chosen_position[1]] = num

    def check_done(self) -> bool:
        """Check if game over."""
        for i in range(4):
            if self.is_move_valid(i):
                return False
        return True


    def move(self, direction) -> tuple[int, bool, bool]:
        """Execute move. Returns (score, done, moved)."""
        _raise_if_invalid_board(self.board)
        if board_would_overflow(self.board, direction):
            raise ValueError(
                f"Move would exceed max exponent {MAX_EXPONENT} by producing "
                f"{MAX_EXPONENT + 1}."
            )

        board_before_move = self.board.copy()
        merge_score = 0

        if direction == LEFT:
            for i in range(4):
                index = row_to_number(self.board[i])
                merge_score += self.move_reward_LUT[index]
                self.board[i] = self.move_row_LUT[index]
        elif direction == RIGHT:
            for i in range(4):
                index = row_to_number(self.board[i, ::-1])
                merge_score += self.move_reward_LUT[index]
                self.board[i, ::-1] = self.move_row_LUT[index]
        elif direction == UP:
            for i in range(4):
                index = row_to_number(self.board[:, i])
                merge_score += self.move_reward_LUT[index]
                self.board[:, i] = self.move_row_LUT[index]
        elif direction == DOWN:
            for i in range(4):
                index = row_to_number(self.board[::-1, i])
                merge_score += self.move_reward_LUT[index]
                self.board[::-1, i] = self.move_row_LUT[index]

        self.score += merge_score
        moved = not np.array_equal(board_before_move, self.board)
        if moved:
            self.generate_random()

        self.update_max_tile()
        self.done = self.check_done()

        return merge_score, self.done, moved

@njit
def row_to_number(row):
    """Row to 16-bit integer."""
    return row[0] | (row[1] << 4) | (row[2] << 8) | (row[3] << 12)


def _raise_if_invalid_board(board):
    if board_has_invalid_exponent(board):
        raise ValueError(
            f"Board exponents must be in the range 0..{MAX_EXPONENT}."
        )


@njit
def board_has_invalid_exponent(board):
    for r in range(4):
        for c in range(4):
            if board[r, c] < 0 or board[r, c] > MAX_EXPONENT:
                return True
    return False


@njit
def row_would_overflow(row):
    """Return whether stacking this row would merge two maximum tiles."""
    stacked = row.copy()
    stack_row(stacked)
    for i in range(1, 4):
        if stacked[i - 1] == MAX_EXPONENT and stacked[i] == MAX_EXPONENT:
            return True
    return False


@njit
def board_would_overflow(board, direction):
    """Return whether a move would create an exponent outside the contract."""
    if direction == LEFT:
        for i in range(4):
            if row_would_overflow(board[i]):
                return True
    elif direction == RIGHT:
        for i in range(4):
            if row_would_overflow(board[i, ::-1]):
                return True
    elif direction == UP:
        for i in range(4):
            if row_would_overflow(board[:, i]):
                return True
    elif direction == DOWN:
        for i in range(4):
            if row_would_overflow(board[::-1, i]):
                return True
    return False

@njit
def stack_row(row):
    """Stack row tiles."""
    for k in range(4):
        for i in range(1, 4):
            if row[i] != 0 and row[i - 1] == 0:
                row[i - 1] = row[i]
                row[i] = 0
    return row

@njit
def merge_row(row):
    """Merge row tiles."""
    reward = 0
    for i in range(1, 4):
        if row[i-1] == row[i] and row[i] != 0:
            row[i-1] += 1
            row[i] = 0
            reward += 2**row[i-1]
    return (row, reward)
