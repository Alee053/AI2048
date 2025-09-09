import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

# Custom Wandb Callback
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]

                if 'episode' in info:
                    wandb.log({
                        "Episode Reward Mean": info['episode']['r'],
                        "Episode Length Mean": info['episode']['l'],
                        "Score": info.get('score', 0),
                        "Max Tile": info.get('max_tile', 0),
                    })
        return True

# Game2048Env functions
def board_to_tensor(board):
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))
    if log_board.ndim == 2: # Single board
        return np.expand_dims(log_board, axis=0)
    else: # Batch of boards
        return np.expand_dims(log_board, axis=1) # Add channel dimension

MASTER_SNAKE_PATTERN = np.array([
    [15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]
], dtype=np.float32)
SNAKE_PATTERNS = []
def generate_snake_patterns():
    if SNAKE_PATTERNS: return
    board = MASTER_SNAKE_PATTERN
    for _ in range(4):
        SNAKE_PATTERNS.append(board)
        SNAKE_PATTERNS.append(np.fliplr(board))
        board = np.rot90(board)

def _find_longest_snake(board, start_r, start_c):
    stack = [([(start_r, start_c)], board[start_r, start_c])]
    max_score = 0

    while stack:
        path, current_score = stack.pop()

        if current_score > max_score:
            max_score = current_score

        r, c = path[-1]
        current_val = board[r, c]

        # Explore neighbors (Up, Down, Left, Right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # Check bounds and if the neighbor has been visited in this path
            if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in path:
                neighbor_val = board[nr, nc]

                # The core logic: the next tile must be exactly one less than the current one.
                if neighbor_val == current_val - 1 and neighbor_val > 0:
                    new_path = path + [(nr, nc)]
                    # The score is the sum of the log2 values in the snake
                    new_score = current_score + neighbor_val
                    stack.append((new_path, new_score))

    return max_score

ROW_GRADIENT = np.arange(16, dtype=np.float32).reshape(4, 4)
COL_GRADIENT = ROW_GRADIENT.T # Transposed version
def calculate_reward(board, merge_score, moved):
    if not moved:
        return -1.0

    reward = np.log2(merge_score) if merge_score > 0 else 0.0

    free_cells = np.sum(board == 0)
    reward += free_cells * 0.1

    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))

    s1 = np.sum(log_board * ROW_GRADIENT)
    s2 = np.sum(log_board * COL_GRADIENT)

    reward += np.maximum(s1, s2) * 1e-4

    return reward
# Fast 2048 functions
def row_to_number(row):
    return row[0] | row[1]<<4 | row[2]<<8 | row[3]<<12
def stack_row(row):
    for k in range(4):
        for i in range(1, 4):
            if row[i]!=0 and row[i - 1]==0:
                row[i-1]=row[i]
                row[i]=0
    return row
def merge_row(row):
    reward=0
    for i in range(1,4):
        if row[i-1]==row[i] and row[i]!=0:
            row[i-1]+=1
            row[i]=0
            reward+=2**row[i-1]
    return [row,reward]

