import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb


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
        # Filter out zeros before checking for order
        row_filtered, col_filtered = row[row > 0], col[col > 0]
        # Check for decreasing or increasing order and take the best score
        if len(row_filtered) > 1:
            mono_score += max(np.sum(np.diff(row_filtered) <= 0), np.sum(np.diff(row_filtered) >= 0))
        if len(col_filtered) > 1:
            mono_score += max(np.sum(np.diff(col_filtered) <= 0), np.sum(np.diff(col_filtered) >= 0))

    final_reward = (
        event_reward * 1.0 +         # The most important signal
        mono_score * 0.1         # A strong nudge for order
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