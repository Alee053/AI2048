import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

class AdaptiveCurriculumCallback(BaseCallback):
    """
    A callback that adjusts the environment's difficulty based on the agent's performance.
    """
    def __init__(self, verbose=0):
        super(AdaptiveCurriculumCallback, self).__init__(verbose)
        self.difficulty_level = 0
        self.max_difficulty = 100
        self.reward_buffer = deque(maxlen=100) # Using a 100-episode window
        self.reward_threshold = 9000           # Starting threshold (TUNE THIS from your W&B logs)
        self.threshold_increment = 3000        # Increment value (TUNE THIS)

    def _update_env_difficulty(self):
        progress = self.difficulty_level / self.max_difficulty
        p_helpful = max(0.0, 1.0 - progress)
        prob_4 = min(0.1, progress * 0.1)
        self.training_env.env_method('set_difficulty', p_helpful, prob_4)
        print(f"\nLEVEL UP! New Difficulty: {self.difficulty_level}/{self.max_difficulty}. Threshold now: {self.reward_threshold}")
        # Log the new state, you can also do this in the logging callback
        wandb.log({
            "Curriculum/Difficulty_Level": self.difficulty_level,
            "Curriculum/Current_Reward_Threshold": self.reward_threshold
        })

    def _on_training_start(self) -> None:
        # Set initial difficulty on all environments
        self._update_env_difficulty()

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                if 'episode' in info:
                    self.reward_buffer.append(info['episode']['r'])

                # Check for mastery only when the buffer is full
                if len(self.reward_buffer) == self.reward_buffer.maxlen:
                    mean_reward = np.mean(self.reward_buffer)

                    if mean_reward > self.reward_threshold and self.difficulty_level < self.max_difficulty:
                        self.difficulty_level += 1
                        self.reward_threshold += self.threshold_increment
                        self._update_env_difficulty()
                        self.reward_buffer.clear()  # Clear buffer to start fresh
        return True


class WandbLoggingCallback(BaseCallback):
    """
    A callback that handles all logging to Weights & Biases.
    """
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.episode_count = 0
        self.log_chart_freq = 100

    def _on_step(self) -> bool:
        # Log curriculum parameters (can be done periodically)
        if self.num_timesteps % 2048 == 0:
            game = self.training_env.get_attr('game')[0]
            wandb.log({"Curriculum/p_helpful": game.p_helpful, "Curriculum/prob_4": game.prob_4})

        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episode_count += 1
                info = self.locals['infos'][i]

                if 'episode' in info:
                    wandb.log({"Mean/Episode Reward": info['episode']['r'], "Mean/Episode Length": info['episode']['l']})

                final_score = info.get('score')
                final_max_tile = info.get('max_tile')
                if final_score is not None:
                    wandb.log({"Episode/Final Score": final_score, "Episode/Final Max Tile": 2**final_max_tile if final_max_tile > 0 else 0})

                if self.episode_count % self.log_chart_freq == 0 and 'terminal_observation' in info:
                    final_obs = info['terminal_observation'].squeeze()
                    exps, counts = np.unique(final_obs[final_obs > 0], return_counts=True)
                    vals = [str(int(2**e)) for e in exps]
                    table = wandb.Table(data=list(zip(vals, counts)), columns=["Tile", "Count"])
                    wandb.log({"Charts/Final Board State": wandb.plot.bar(table, "Tile", "Count", title="Final Tile Distribution")})
        return True


def board_to_tensor(board):
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))
    if log_board.ndim == 2:
        return np.expand_dims(log_board, axis=0)
    else:
        return np.expand_dims(log_board, axis=1)