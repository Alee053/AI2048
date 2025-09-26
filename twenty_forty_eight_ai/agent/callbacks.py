import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbLoggingCallback(BaseCallback):
    """
    A custom callback for logging detailed training information to Weights & Biases.

    This callback logs:
    - Mean reward and episode length at the end of each episode.
    - Raw final score and max tile for creating histograms in W&B.
    - A bar chart of the final tile distribution, logged periodically.
    """

    def __init__(self, log_chart_freq: int = 100, verbose: int = 0):
        """
        Initializes the callback.

        Args:
            log_chart_freq (int): Log the final board state bar chart every N episodes.
            verbose (int): The verbosity level.
        """
        super().__init__(verbose)
        self.episode_count = 0
        self.log_chart_freq = log_chart_freq

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        It checks for completed episodes and logs relevant data to W&B.
        """
        # The `dones` local variable is a boolean array indicating which environments
        # in the VecEnv have finished an episode.
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episode_count += 1
                info = self.locals['infos'][i]

                # --- Log Mean Statistics from the Monitor Wrapper ---
                # These are smoothed averages provided by the SB3 Monitor wrapper.
                if 'episode' in info:
                    wandb.log({
                        "Rollout/Mean Episode Reward": info['episode']['r'],
                        "Rollout/Mean Episode Length": info['episode']['l']
                    })

                # --- Log Raw Episode-End Statistics for Histograms ---
                # These are the raw, unsmoothed values for a single episode.
                # W&B will automatically aggregate these into histograms.
                final_score = info.get('score')
                final_max_tile = info.get('max_tile')
                if final_score is not None and final_max_tile is not None:
                    wandb.log({
                        "Episode/Final Score": final_score,
                        "Episode/Final Max Tile": 2 ** final_max_tile if final_max_tile > 0 else 0
                    })

                # --- Log Final Board State as a Bar Chart Periodically ---
                if self.episode_count % self.log_chart_freq == 0 and 'terminal_observation' in info:
                    final_obs = info['terminal_observation'].squeeze()

                    # Robustness: Sanitize the observation to remove any invalid numbers
                    final_obs = final_obs[np.isfinite(final_obs)]

                    tiles_with_values = final_obs[final_obs > 0]

                    if tiles_with_values.size > 0:
                        exps, counts = np.unique(tiles_with_values, return_counts=True)
                        vals = [str(int(2 ** e)) for e in exps]
                        table = wandb.Table(data=list(zip(vals, counts)), columns=["Tile", "Count"])
                        wandb.log({"Charts/Final Board State": wandb.plot.bar(
                            table, "Tile", "Count", title="Final Tile Distribution"
                        )})
                    elif self.verbose > 0:
                        print(
                            f"\n[WandbLoggingCallback] Skipped bar chart: No non-zero tiles found in terminal observation.")

        return True