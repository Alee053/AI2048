from Fast2048 import Fast2048
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


class Game2048Env(Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.game = Fast2048()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0, high=18, shape=(19,))
        self.state = self.get_state()

    def get_state(self):
        current=np.concatenate([self.flatten(self.game.board),[self.game.max_tile, self.game.empty_cells, self.game.sum_tiles]])
        return current


    def reset(self):
        self.game.reset()
        self.state = self.get_state()
        return self.state, {}

    def step(self, action):
        reward, done = self.game.move(action)
        self.state = self.get_state()
        return self.state, reward, done, {}

    def render(self, mode='human'):
        self.game.show_board()

    def close(self):
        pass

    def flatten(self, board):
        return np.array(board).flatten()


