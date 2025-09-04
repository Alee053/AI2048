import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from .Fast2048 import Fast2048
from .utility import board_to_tensor, calculate_reward


class Game2048Env(Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.game = Fast2048()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0, high=16, shape=(1, 4, 4), dtype=np.float32)

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