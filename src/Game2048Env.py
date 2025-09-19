import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from .Fast2048 import Fast2048
from .utility import board_to_tensor
from .reward_function import calculate_reward


class Game2048Env(Env):
    def __init__(self, total_timesteps=100_000_000):
        super(Game2048Env, self).__init__()
        self.game = Fast2048()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0, high=16, shape=(1, 4, 4), dtype=np.float32)

        self.total_timesteps = total_timesteps

    def _update_curriculum(self,global_step):
        progress = global_step / self.total_timesteps

        # --- Dial 1: Probability of spawning a '4' (Continuous) ---
        self.game.prob_4 = min(0.1, progress * 0.1)

        # --- Dial 2: Probability of a "Helpful" Spawn (Continuous) ---
        # Linearly decay the chance of a helpful spawn from 100% to 0%.
        self.game.p_helpful = max(0.0, 1.0 - progress)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.game.reset()
        state = board_to_tensor(self.game.board)
        return state, {}

    def get_valid_actions_mask(self):
        mask = [False] * 4
        for action in range(4):
            if self.game.is_move_valid(action):
                mask[action] = True
        return np.array(mask)


    def step(self, action):
        merge_score, done, moved = self.game.move(action)
        state = board_to_tensor(self.game.board)

        # The reward is now calculated with the new, robust function
        reward = calculate_reward(self.game.board, merge_score, moved)

        info = {}
        if done:
            info['max_tile'] = self.game.max_tile
            info['score'] = self.game.score

        truncated = False
        return state, reward, done, truncated, info

    def action_masks(self):
        return self.get_valid_actions_mask()

    def render(self, mode='human'):
        pass

    def close(self):
        pass