import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

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

def board_to_tensor(board):
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))
    if log_board.ndim == 2:
        return np.expand_dims(log_board, axis=0)
    else:
        return np.expand_dims(log_board, axis=1)