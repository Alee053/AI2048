import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import Tuple, Dict, Any

from .game import Fast2048
from ..utils.tensor_utils import board_to_tensor
from .reward import calculate_reward


class Game2048Env(Env):
    """A custom Gymnasium environment for the 2048 game.

    This environment is designed to be compatible with Stable Baselines 3
    and specifically for use with MaskablePPO. It wraps a highly optimized
    2048 game engine (`Fast2048`) and provides the standard Gymnasium API.

    Attributes:
        game (Fast2048): The underlying 2048 game engine.
        action_space (Discrete): The action space, with 4 discrete actions
            (UP, RIGHT, DOWN, LEFT).
        observation_space (Box): The observation space, represented as a
            1x4x4 tensor containing the log2 values of the board tiles.
    """

    def __init__(self):
        """Initializes the 2048 game environment."""
        super().__init__()
        self.game = Fast2048()

        # Action space: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = Discrete(4)

        # Observation space: 1x4x4 grid with log2 values of the tiles.
        self.observation_space = Box(low=0, high=16, shape=(1, 4, 4), dtype=np.int64)


    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to a starting state.

        This method conforms to the Gymnasium API, resetting the game board
        and returning the initial observation.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (Dict[str, Any], optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing the initial
            observation and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.game.reset()
        observation = board_to_tensor(self.game.board)
        info = {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step within the environment.

        This method applies an action to the game, calculates the reward,
        and returns the next state, reward, and other info, conforming to the
        Gymnasium API.

        Args:
            action (int): The action to be executed (0-3).

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: A tuple containing
            the next observation, the reward, a 'terminated' flag, a 'truncated'
            flag, and an info dictionary.
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
        """Generates a mask of valid actions for the current state.

        This method is specifically for use with MaskablePPO, providing a
        boolean array indicating which actions are currently legal.

        Returns:
            np.ndarray: A boolean numpy array of shape (4,), where True
            indicates a valid move.
        """
        return np.array([self.game.is_move_valid(act) for act in range(self.action_space.n)], dtype=bool)

    def render(self, mode: str = 'human'):
        """Renders the environment.

        Note:
            Rendering is handled by a separate visualizer in this implementation.
            This method is a placeholder.
        """
        pass

    def close(self):
        """Performs any necessary cleanup.

        This method is called when the environment is no longer in use.
        """
        pass