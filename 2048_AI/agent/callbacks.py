import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os

class AdaptiveCurriculumCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(AdaptiveCurriculumCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        # --- NEW: State machine for your proposed curriculum ---
        self.state = "LEARNING_STRATEGY"

        # --- Phase 1: Learning Strategy Parameters ---
        self.milestone_target_tile = 11
        self.max_tile_buffer = deque(maxlen=100)

        # --- Phase 2: Hardening Policy Parameters ---
        self.difficulty_increase_interval = 500_000
        self.last_hardening_check = 0
        self.performance_floor_tile = self.milestone_target_tile

        # --- General Curriculum State ---
        self.difficulty_level = 0
        self.max_difficulty = 100
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
        if self.verbose > 0:
            print(f"\nDifficulty set to Level {self.difficulty_level}. p_helpful={p_helpful:.2f}, prob_4={prob_4:.3f}")
        wandb.log({"Curriculum/Difficulty_Level": self.difficulty_level})

    def _level_up(self):
        if self.difficulty_level < self.max_difficulty:
            self.difficulty_level += 1
            self._update_env_difficulty()

    def _on_save(self) -> None:
        """Called by the checkpointing logic to save the curriculum's state."""
        state_data = {
            "state": self.state,
            "difficulty_level": self.difficulty_level,
            "max_tile_buffer": list(self.max_tile_buffer),
            "last_hardening_check": self.last_hardening_check,
            "is_initialized": self.is_initialized,
        }
        self.model.curriculum_state = state_data

        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
        self.model.save(path)
        del self.model.curriculum_state
        if self.verbose > 1:
            print(f"Saving model checkpoint with curriculum state to {path}")

    def _on_step(self) -> bool:
        if not self.is_initialized:
            self._update_env_difficulty()  # This sets the initial Level 0 difficulty
            self.is_initialized = True

        for i, done in enumerate(self.locals['dones']):
            if done and 'episode' in self.locals['infos'][i]:
                # Always fill the max tile buffer with the latest data
                if 'max_tile' in self.locals['infos'][i]:
                    self.max_tile_buffer.append(self.locals['infos'][i]['max_tile'])

                # --- State Machine Logic ---
                if self.state == "LEARNING_STRATEGY":
                    # In this phase, difficulty is LOCKED at Level 0
                    if len(self.max_tile_buffer) == self.max_tile_buffer.maxlen:
                        avg_max_tile = np.mean(self.max_tile_buffer)
                        wandb.log({"Curriculum/Avg_Max_Tile_Exponent": avg_max_tile})

                        if avg_max_tile >= self.milestone_target_tile:
                            self.state = "HARDENING_POLICY"
                            if self.verbose > 0:
                                print(f"\n--- STRATEGY MASTERED (Avg Max Tile >= 2048) ---")
                                print(f"Beginning policy hardening phase.")
                            self._level_up()  # Immediately level up to start the first real challenge

                elif self.state == "HARDENING_POLICY":
                    # In this phase, difficulty increases over time if performance holds
                    if self.num_timesteps - self.last_hardening_check > self.difficulty_increase_interval:
                        self.last_hardening_check = self.num_timesteps
                        if len(self.max_tile_buffer) == self.max_tile_buffer.maxlen:
                            avg_max_tile = np.mean(self.max_tile_buffer)
                            wandb.log({"Curriculum/Avg_Max_Tile_Exponent": avg_max_tile})

                            if avg_max_tile >= self.performance_floor_tile:
                                self._level_up()  # Performance is good, increase env difficulty
                            else:
                                if self.verbose > 0:
                                    print(
                                        f"\nPERFORMANCE DIP! Avg max tile {2 ** avg_max_tile:.0f} is below floor of 2048. Pausing difficulty increase.")

        # --- Checkpoint Saving Logic ---
        if self.n_calls % self.save_freq == 0:
            self._on_save()  # Call the save method

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

