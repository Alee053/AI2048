import numpy as np
import wandb
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os

class AdaptiveCurriculumCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(AdaptiveCurriculumCheckpointCallback, self).__init__(verbose)

        # --- Checkpoint Parameters ---
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        # --- Curriculum State ---
        self.difficulty_level = 0
        self.max_difficulty = 100
        self.reward_buffer = deque(maxlen=100)
        self.reward_threshold = 1600
        self.threshold_increment = 400
        self.is_initialized = False

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        super().init_callback(model)
        if self.is_initialized:
            if self.verbose > 0:
                print("Resumed training detected. Syncing curriculum state with environment.")
            self._update_env_difficulty()

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

    def _on_step(self) -> bool:
        # Initialize curriculum on the first step
        if not self.is_initialized:
            self._update_env_difficulty()
            self.is_initialized = True

        # --- Adaptive Curriculum Logic ---
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                if 'episode' in info:
                    self.reward_buffer.append(info['episode']['r'])

                if len(self.reward_buffer) == self.reward_buffer.maxlen:
                    mean_reward = np.mean(self.reward_buffer)
                    if mean_reward > self.reward_threshold and self.difficulty_level < self.max_difficulty:
                        self.difficulty_level += 1
                        self.reward_threshold += self.threshold_increment
                        self._update_env_difficulty()
                        self.reward_buffer.clear()

        # --- Checkpoint Saving Logic ---
        if self.n_calls % self.save_freq == 0:
            # Package the curriculum state into a dictionary
            state_data = {
                "difficulty_level": self.difficulty_level,
                "reward_threshold": self.reward_threshold,
                "reward_buffer": list(self.reward_buffer),
                "is_initialized": self.is_initialized,
            }

            # THE FIX: Add the state dictionary as a temporary attribute to the model
            self.model.curriculum_state = state_data

            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(path)

            # Optional but good practice: remove the temporary attribute after saving
            del self.model.curriculum_state

            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

        return True


class WandbLoggingCallback(BaseCallback):
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