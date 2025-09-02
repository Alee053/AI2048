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

def calculate_reward(board,merge_score):
    event_reward = 0
    if merge_score > 0:
        event_reward = np.log2(merge_score)

    # Monotonicity score
    mono_score = calculate_monotonicity_score(board)


    empty_score = np.sum(board == 0)

    # Reward for having the max tile in a corner
    max_tile_score = 0
    max_tile_val = np.max(board)
    if max_tile_val > 0:
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        if max_tile_val in corners:
            max_tile_score = np.log2(max_tile_val)


    # The total reward is the log-scaled event reward plus the state bonus
    final_reward = (
            event_reward*1.5 +
            mono_score/3 +
            empty_score * 1 +
            max_tile_score * 1
    )

    return final_reward


def calculate_monotonicity_score(board):
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))
    log_board = log_board.astype(np.int32)

    def get_score(line):
        score = 0
        for i in range(len(line) - 1):
            if (line[i] == line[i + 1]-1 or line[i] == line[i + 1]) and line[i]!=0:
                score += 1
        if score>0:
            score+=1
        return score
    total_score = 0

    for row in log_board:
        total_score += get_score(row)
        total_score += get_score(row[::-1])
    for col in log_board.T:
        total_score += get_score(col)
        total_score += get_score(col[::-1])
    return total_score


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