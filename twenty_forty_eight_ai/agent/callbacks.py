import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbLoggingCallback(BaseCallback):
    """A custom callback for logging detailed training information to Weights & Biases.

    This callback integrates with Stable Baselines 3 to provide enhanced logging
    during the training process. At the end of each episode, it logs various
    metrics to W&B, providing a comprehensive overview of the agent's performance.

    Logs include:
    - Smoothed mean reward and episode length (from the SB3 Monitor wrapper).
    - Raw final score and max tile value for each episode, which W&B uses to
      generate histograms.
    - A bar chart of the final tile distribution, logged at a specified frequency.

    Attributes:
        episode_count (int): A counter for the number of episodes completed.
        log_chart_freq (int): The frequency (in episodes) at which to log the
            final board state bar chart.
    """

    def __init__(self, log_chart_freq: int = 100, verbose: int = 0):
        """Initializes the WandbLoggingCallback.

        Args:
            log_chart_freq (int, optional): The frequency (in episodes) for
                logging the final board state as a bar chart. Defaults to 100.
            verbose (int, optional): The verbosity level (0 for no output).
                Defaults to 0.
        """
        super().__init__(verbose)
        self.episode_count = 0
        self.log_chart_freq = log_chart_freq

    def _on_step(self) -> bool:
        """Called by Stable Baselines 3 after each step in the environment.

        This method checks for the completion of any episodes within the
        vectorized environment. If an episode is done, it extracts relevant
        information from the `info` dictionary and logs it to Weights & Biases.

        Returns:
            bool: True to continue training, False to stop. This implementation
            always returns True.
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