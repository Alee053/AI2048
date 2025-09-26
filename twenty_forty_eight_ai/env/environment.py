import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import Tuple, Dict, Any

from .game import Fast2048
from ..utils.tensor_utils import board_to_tensor
from .reward import calculate_reward


class Game2048Env(Env):
    """
    A custom Gymnasium environment for the 2048 game, designed to work
    with Stable Baselines 3 and MaskablePPO. It wraps the optimized Fast2048 game engine.
    """

    def __init__(self):
        """Initializes the environment, action space, and observation space."""
        super().__init__()
        self.game = Fast2048()

        # Action space: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = Discrete(4)

        # Observation space: 1x4x4 grid with log2 values of the tiles.
        self.observation_space = Box(low=0, high=17, shape=(1, 4, 4), dtype=np.int64)


    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to a starting state as per the Gymnasium API.

        Returns:
            A tuple containing the initial observation and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.game.reset()
        observation = board_to_tensor(self.game.board)
        info = {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment as per the Gymnasium API.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        merge_score, done, moved = self.game.move(action)
        observation = board_to_tensor(self.game.board)

        reward = calculate_reward(self.game.board, merge_score, moved)

        info = {}
        if done:
            info['max_tile'] = self.game.max_tile
            info['score'] = self.game.score

        truncated = False
        return observation, reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        """Generates a mask of valid actions for the current state for MaskablePPO."""
        return np.array([self.game.is_move_valid(act) for act in range(self.action_space.n)], dtype=bool)

    def render(self, mode='human'):
        """Rendering is handled by a separate visualizer."""
        pass

    def close(self):
        """Called when the environment is closed."""
        pass