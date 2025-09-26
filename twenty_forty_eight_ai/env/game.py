import numpy as np
from numba import njit

# --- Constants for improved readability ---
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

class Fast2048:
    """
        A highly optimized, Look-Up Table (LUT) based implementation of the 2048 game logic.
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
        """Pre-computes the results of all 65,536 possible row moves."""
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
        """Resets the game to a starting state with two random tiles."""
        self.board.fill(0)
        self.max_tile = 0
        self.score = 0
        self.done = False
        self.generate_random()
        self.generate_random()
        self.update_max_tile()

    def update_max_tile(self) -> bool:
        """Updates the max_tile attribute."""
        self.max_tile = np.max(self.board)

    def is_move_valid(self, action):
        """Checks if a move is possible in the given direction."""
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
        """Generates a new tile on the board, potentially using the helpful spawner."""
        num = 1 if np.random.random() < 0.9 else 2
        empty_cells = np.argwhere(self.board == 0)

        if empty_cells.size == 0:
            return

        chosen_position = empty_cells[np.random.choice(len(empty_cells))]

        self.board[chosen_position[0], chosen_position[1]] = num

    def check_done(self) -> bool:
        """Checks if there are any valid moves left on the board."""
        for i in range(4):
            if self.is_move_valid(i):
                return False
        return True


    def move(self, direction) -> tuple[int, bool, bool]:
        """
            Performs a move in a given direction and returns the results.
            Returns: (merge_score, is_done, was_moved)
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
def row_to_number(row):
    """Converts a 4-element row into a 16-bit integer for LUT indexing."""
    return row[0] | (row[1] << 4) | (row[2] << 8) | (row[3] << 12)

@njit
def stack_row(row):
    """Stacks tiles to one side of a row (part of LUT creation)."""
    for k in range(4):
        for i in range(1, 4):
            if row[i] != 0 and row[i - 1] == 0:
                row[i - 1] = row[i]
                row[i] = 0
    return row

@njit
def merge_row(row):
    """Merges adjacent identical tiles in a row (part of LUT creation)."""
    reward = 0
    for i in range(1, 4):
        if row[i-1] == row[i] and row[i] != 0:
            row[i-1] += 1
            row[i] = 0
            reward += 2**row[i-1]
    return (row, reward)