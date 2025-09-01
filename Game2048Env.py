import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from Fast2048 import Fast2048
from utility import board_to_tensor


class Game2048Env(Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.game = Fast2048()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0, high=1, shape=(16, 4, 4), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.game.reset()
        state = board_to_tensor(self.game.board)
        return state, {}

    def step(self, action):
        reward, done = self.game.move(action)
        state = board_to_tensor(self.game.board)

        info={}

        if done:
            info['max_tile']=self.game.max_tile
            info['score']=self.game.score

        truncated=False

        return state, reward, done,truncated, info

    def render(self, mode='human'):
        self.game.show_board()

    def close(self):
        pass