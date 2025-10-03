import numpy as np
from numba import njit

# --- Constants for improved readability ---
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

class Fast2048:
    """A highly optimized, Look-Up Table (LUT) based implementation of the 2048 game logic.

    This class manages the game state, including the board, score, and game-over status.
    It uses pre-computed look-up tables for move calculations to achieve high performance.

    Attributes:
        move_row_LUT (list): A look-up table for row movements.
        move_reward_LUT (list): A look-up table for rewards based on row movements.
        move_valid_LUT (list): A look-up table to check if a move is valid for a row.
        board (np.ndarray): A 4x4 numpy array representing the game board.
        max_tile (int): The value of the highest tile on the board.
        score (int): The current score of the game.
        done (bool): A flag indicating if the game is over.
    """
    move_row_LUT: list = []
    move_reward_LUT: list = []
    move_valid_LUT: list = []

    def __init__(self):
        """Initializes the game instance, creating the LUTs if they don't exist."""
        if not Fast2048.move_row_LUT:
            self.init_LUT()

        self.board = np.zeros((4, 4), dtype=np.int32)
        self.max_tile: int = 0
        self.score: int = 0
        self.done: bool = False

        self.reset()

    def init_LUT(self):
        """Pre-computes the results of all 65,536 possible row moves.

        This method generates look-up tables for all possible outcomes of a move
        on a single row, which significantly speeds up the game calculations.
        """
        for i in range(65536):
            original_row = np.array([(i >> (j * 4)) & 0xF for j in range(4)], dtype=np.int32)
            row = original_row.copy()

            row = stack_row(row)
            row, reward = merge_row(row)
            row = stack_row(row)

            Fast2048.move_row_LUT.append(row)
            Fast2048.move_reward_LUT.append(reward)
            Fast2048.move_valid_LUT.append(not np.array_equal(original_row, row))

    def reset(self):
        """Resets the game to a starting state with two random tiles.

        This method clears the board, resets the score and max_tile, and places
        two new random tiles to begin a new game.
        """
        self.board.fill(0)
        self.max_tile = 0
        self.score = 0
        self.done = False
        self.generate_random()
        self.generate_random()
        self.update_max_tile()

    def update_max_tile(self) -> bool:
        """Updates the max_tile attribute with the highest value on the board."""
        self.max_tile = np.max(self.board)

    def is_move_valid(self, action: int) -> bool:
        """Checks if a move is possible in the given direction.

        Args:
            action (int): The direction of the move (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT).

        Returns:
            bool: True if the move is valid, False otherwise.
        """
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
        """Generates a new tile on the board.

        A new tile is placed in a random empty cell. The tile has a 90% chance of
        being a 2 (represented as 1) and a 10% chance of being a 4 (represented as 2).
        """
        num = 1 if np.random.random() < 0.9 else 2
        empty_cells = np.argwhere(self.board == 0)

        if empty_cells.size == 0:
            return

        chosen_position = empty_cells[np.random.choice(len(empty_cells))]

        self.board[chosen_position[0], chosen_position[1]] = num

    def check_done(self) -> bool:
        """Checks if there are any valid moves left on the board.

        Returns:
            bool: True if no more moves are possible, False otherwise.
        """
        for i in range(4):
            if self.is_move_valid(i):
                return False
        return True


    def move(self, direction: int) -> tuple[int, bool, bool]:
        """Performs a move in a given direction and updates the game state.

        Args:
            direction (int): The direction of the move (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT).

        Returns:
            tuple[int, bool, bool]: A tuple containing the merge score, a flag
            indicating if the game is done, and a flag indicating if the move
            resulted in a change to the board.
        """
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
def row_to_number(row: np.ndarray) -> int:
    """Converts a 4-element row into a 16-bit integer for LUT indexing.

    Args:
        row (np.ndarray): A 1D numpy array of 4 integers representing a row.

    Returns:
        int: The 16-bit integer representation of the row.
    """
    return row[0] | (row[1] << 4) | (row[2] << 8) | (row[3] << 12)

@njit
def stack_row(row: np.ndarray) -> np.ndarray:
    """Stacks tiles to the left side of a row.

    This function is used during the LUT creation process. It shifts all non-zero
    tiles to the left, filling the empty spaces on the right with zeros.

    Args:
        row (np.ndarray): A 1D numpy array of 4 integers representing a row.

    Returns:
        np.ndarray: The row with all tiles stacked to the left.
    """
    for k in range(4):
        for i in range(1, 4):
            if row[i] != 0 and row[i - 1] == 0:
                row[i - 1] = row[i]
                row[i] = 0
    return row

@njit
def merge_row(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Merges adjacent identical tiles in a row and calculates the reward.

    This function is used during the LUT creation process. It merges adjacent
    tiles of the same value, adds the merged value to the score, and shifts
    the remaining tiles accordingly.

    Args:
        row (np.ndarray): A 1D numpy array of 4 integers representing a row.

    Returns:
        tuple[np.ndarray, int]: A tuple containing the merged row and the
        reward obtained from the merge.
    """
    reward = 0
    for i in range(1, 4):
        if row[i-1] == row[i] and row[i] != 0:
            row[i-1] += 1
            row[i] = 0
            reward += 2**row[i-1]
    return (row, reward)