import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        # Log a bar chart every N completed episodes to avoid cluttering the logs
        self.log_chart_freq = 100
        self.episode_count = 0
        self.log_cl_freq = 2048

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_cl_freq == 0:
            self.training_env.env_method('update_curriculum', self.num_timesteps)
            game_instance = self.training_env.get_attr('game')[0]

            p_helpful = game_instance.p_helpful
            prob_4 = game_instance.prob_4

            wandb.log({
                "Curriculum/p_helpful": p_helpful,
                "Curriculum/prob_4": prob_4
            })

        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episode_count += 1
                info = self.locals['infos'][i]

                # --- Standard Scalar Logging (from method 1) ---
                if 'episode' in info:
                    wandb.log(
                        {"Mean/Episode Reward": info['episode']['r'], "Mean/Episode Length": info['episode']['l']})

                final_score = info.get('score')
                final_max_tile = info.get('max_tile')
                if final_score is not None:
                    wandb.log({"Episode/Final Score": final_score,
                               "Episode/Final Max Tile": 2 ** final_max_tile if final_max_tile > 0 else 0})

                # --- Custom Bar Chart Logging ---
                if self.episode_count % self.log_chart_freq == 0:
                    # THE FIX: Check for the true terminal observation in the info dict
                    if 'terminal_observation' in info:
                        final_board_obs = info['terminal_observation'].squeeze()

                        # The rest of the logic is the same
                        tile_exponents, counts = np.unique(final_board_obs[final_board_obs > 0], return_counts=True)
                        tile_values = [str(int(2 ** exp)) for exp in tile_exponents]

                        table = wandb.Table(data=list(zip(tile_values, counts)), columns=["Tile Value", "Count"])
                        bar_chart = wandb.plot.bar(table, "Tile Value", "Count", title="Final Tile Distribution")
                        wandb.log({"Charts/Final Board State": bar_chart})

        return True

def board_to_tensor(board):
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))
    if log_board.ndim == 2:
        return np.expand_dims(log_board, axis=0)
    else:
        return np.expand_dims(log_board, axis=1)