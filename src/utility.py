import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os

class AdaptiveCurriculumCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str,
                 reward_buffer_size: int = 100,
                 threshold_increment_percent: float = 1.05,
                 name_prefix: str = "rl_model", verbose: int = 0):
        super(AdaptiveCurriculumCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.state = "CALIBRATING"
        self.initial_challenge_ratio = 1.10

        # --- Curriculum State ---
        self.difficulty_level = 0
        self.max_difficulty = 100
        self.reward_buffer = deque(maxlen=reward_buffer_size)
        self.reward_threshold = float('inf')
        self.threshold_increment_percent = threshold_increment_percent
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

    def _update_env_difficulty(self, is_level_up: bool):
        progress = self.difficulty_level / self.max_difficulty
        p_helpful = max(0.0, 1.0 - progress)
        prob_4 = min(0.1, progress * 0.1)
        self.training_env.env_method('set_difficulty', p_helpful, prob_4)

        if is_level_up and self.verbose > 0:
            print(
                f"\nLEVEL UP! New Difficulty: {self.difficulty_level}/{self.max_difficulty}. Threshold now: {self.reward_threshold:.0f}")

        wandb.log({
            "Curriculum/Difficulty_Level": self.difficulty_level,
            "Curriculum/Current_Reward_Threshold": self.reward_threshold
        })

    def _level_up(self, increment_threshold: bool):
        if self.difficulty_level < self.max_difficulty:
            self.difficulty_level += 1
            if increment_threshold:
                self.reward_threshold *= self.threshold_increment_percent
            self._update_env_difficulty(is_level_up=True)
            self.reward_buffer.clear()

    def _on_step(self) -> bool:
        if not self.is_initialized:
            self._update_env_difficulty(is_level_up=False)  # Initial setup call
            self.is_initialized = True

        for i, done in enumerate(self.locals['dones']):
            if done and 'episode' in self.locals['infos'][i]:
                self.reward_buffer.append(self.locals['infos'][i]['episode']['r'])

                # --- State Machine Logic ---
                if self.state == "CALIBRATING":
                    # Wait until the buffer is full to get a stable baseline
                    if len(self.reward_buffer) == self.reward_buffer.maxlen:
                        baseline_reward = np.mean(self.reward_buffer)
                        # Set the first real threshold based on this baseline
                        self.reward_threshold = baseline_reward * self.initial_challenge_ratio
                        self.state = "ADAPTING"

                        if self.verbose > 0:
                            print(f"\n--- CALIBRATION COMPLETE ---")
                            print(f"Baseline reward at Level 0: {baseline_reward:.0f}")

                        # Immediately level up to start the first real challenge
                        self._level_up(increment_threshold=False)

                elif self.state == "ADAPTING":
                    if len(self.reward_buffer) == self.reward_buffer.maxlen:
                        mean_reward = np.mean(self.reward_buffer)
                        if mean_reward > self.reward_threshold:
                            self._level_up(increment_threshold=True)

        # --- Checkpoint Saving Logic ---
        if self.n_calls % self.save_freq == 0:
            # Package the curriculum state into a dictionary
            state_data = {
                "state": self.state,  # Save the current state
                "difficulty_level": self.difficulty_level,
                "reward_threshold": self.reward_threshold,
                "reward_buffer": list(self.reward_buffer),
                "is_initialized": self.is_initialized,
            }
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

                    final_obs = final_obs[np.isfinite(final_obs)]

                    tiles_with_values = final_obs[final_obs > 0]

                    if tiles_with_values.size > 0:
                        exps, counts = np.unique(tiles_with_values, return_counts=True)
                        vals = [str(int(2 ** e)) for e in exps]
                        table = wandb.Table(data=list(zip(vals, counts)), columns=["Tile", "Count"])
                        wandb.log({"Charts/Final Board State": wandb.plot.bar(table, "Tile", "Count",
                                                                              title="Final Tile Distribution")})
                    else:
                        if self.verbose > 0:
                            print(
                                f"\n[WandbLoggingCallback] Skipped bar chart logging at step {self.num_timesteps}: No non-zero tiles found in terminal observation.")
                return True

        return True


def board_to_tensor(board):
    log_board = np.log2(board, out=np.zeros_like(board, dtype=np.float32), where=(board != 0))
    if log_board.ndim == 2:
        return np.expand_dims(log_board, axis=0)
    else:
        return np.expand_dims(log_board, axis=1)