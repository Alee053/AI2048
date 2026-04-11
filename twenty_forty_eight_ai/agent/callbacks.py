import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


def should_log(episode: int) -> bool:
    """Adaptive logging schedule — more frequent early, less frequent late."""
    if episode < 1000:
        return episode % 10 == 0
    elif episode < 10_000:
        return episode % 100 == 0
    elif episode < 100_000:
        return episode % 500 == 0
    else:
        return episode % 1000 == 0


class WandbLoggingCallback(BaseCallback):
    """Custom callback for W&B logging."""

    def __init__(self, verbose: int = 0):
        """Initialize callback."""
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Log stats after each step."""
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episode_count += 1
                info = self.locals['infos'][i]

                # Mean stats
                if 'episode' in info:
                    wandb.log({
                        "Rollout/Mean Episode Reward": info['episode']['r'],
                        "Rollout/Mean Episode Length": info['episode']['l']
                    })

                # Final episode stats
                final_score = info.get('score')
                final_max_tile = info.get('max_tile')
                if final_score is not None and final_max_tile is not None:
                    wandb.log({
                        "Episode/Final Score": final_score,
                        "Episode/Final Max Tile": 2 ** final_max_tile if final_max_tile > 0 else 0
                    })

                # Tile distribution via table (chart configured in W&B UI)
                if (should_log(self.episode_count)
                        and 'terminal_observation' in info):
                    final_obs = info['terminal_observation'].squeeze()
                    final_obs = final_obs[np.isfinite(final_obs)]
                    tiles_with_values = final_obs[final_obs > 0]

                    if tiles_with_values.size > 0:
                        exps, counts = np.unique(tiles_with_values, return_counts=True)
                        vals = [str(int(2 ** e)) for e in exps]
                        table = wandb.Table(
                            data=list(zip(vals, counts)),
                            columns=["Tile", "Count"]
                        )
                        wandb.log({"Tiles/Distribution": table})
                    elif self.verbose > 0:
                        print("[WandbLoggingCallback] Skipped: no non-zero tiles found.")

        return True