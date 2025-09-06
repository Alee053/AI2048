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

    return np.expand_dims(log_board, axis=0)

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


def calculate_reward(board, merge_score, moved):
    if not moved:
        return -1

    event_reward = 0
    if merge_score > 0:
        event_reward = np.log2(merge_score)

    log_board = np.log2 (board, out=np.zeros_like(board, dtype=float), where=(board != 0))

    mono_score = 0
    for i in range(4):
        row = log_board[i, :]
        col = log_board[:, i]
        row_filtered, col_filtered = row[row > 0], col[col > 0]
        if len(row_filtered) > 1:
            mono_score += max(np.sum(np.diff(row_filtered) <= 0), np.sum(np.diff(row_filtered) >= 0))
        if len(col_filtered) > 1:
            mono_score += max(np.sum(np.diff(col_filtered) <= 0), np.sum(np.diff(col_filtered) >= 0))

    corner_bonus = 0
    max_tile_val = np.max(log_board)
    if max_tile_val > 0 and log_board[0, 0] == max_tile_val:
        corner_bonus = max_tile_val

    empty_score = np.sum(board == 0)

    final_reward = (
            event_reward * 1.0 +
            mono_score * 0.15 +
            empty_score * 0.1 +
            corner_bonus * 0.05
    )

    return final_reward

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