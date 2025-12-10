import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import Tuple, Dict, Any

from .game import Fast2048
from ..utils.tensor_utils import board_to_tensor
from .reward import calculate_reward


class Game2048Env(Env):
    """Gymnasium environment for 2048, compatible with MaskablePPO."""

    def __init__(self):
        """Initialize environment."""
        super().__init__()
        self.game = Fast2048()

        # Action space: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = Discrete(4)

        # Observation space: 1x4x4 grid with log2 values of the tiles.
        self.observation_space = Box(low=0, high=16, shape=(1, 4, 4), dtype=np.int64)


    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        self.game.reset()
        observation = board_to_tensor(self.game.board)
        info = {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step. Returns (obs, reward, done, truncated, info)."""
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
        """Get valid action mask."""
        return np.array([self.game.is_move_valid(act) for act in range(self.action_space.n)], dtype=bool)

    def render(self, mode='human'):
        """Render (unused)."""
        pass

    def close(self):
        """Close environment."""
        pass