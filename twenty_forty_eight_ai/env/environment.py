import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import Tuple, Dict, Any, Optional

from ..constants import MAX_EXPONENT
from .game import Fast2048
from ..utils.tensor_utils import board_to_tensor
from .reward import calculate_reward
from .d4_transforms import (
    apply_d4,
    transform_action_mask,
    ACTION_TO_CANONICAL,
    ID,
    NUM_TRANSFORMS,
)


class Game2048Env(Env):
    """Gymnasium environment for 2048, compatible with MaskablePPO.

    Optional D4 (dihedral) augmentation: when enabled, every reset() and
    step() presents the board under a uniformly random D4 symmetry, and
    the agent's chosen action is mapped back to canonical coordinates
    before reaching the game logic. Training the policy and value
    network against this augmented stream forces them to be (approximately)
    invariant to all 8 symmetries, which is required for the C++ searcher
    to safely canonicalize boards without biasing leaf evaluations.

    Default behavior (d4_augment=False) is unchanged.
    """

    def __init__(
        self,
        d4_augment: bool = False,
        d4_seed: Optional[int | np.random.SeedSequence] = None,
    ):
        """Initialize environment.

        Args:
            d4_augment: if True, apply a random D4 transform to the board
                observation on every reset and step, and inverse-permute
                the agent's chosen action. Off by default.
            d4_seed: optional integer or SeedSequence for the per-env D4
                sampler. Independent from the game's tile-placement RNG.

        Note on vec_env usage: when constructing a vectorized environment
        with `make_vec_env(..., env_kwargs={'d4_seed': N})`, every parallel
        env receives the SAME seed, so they sample the same transform
        sequence (degraded diversity). Leave d4_seed=None (the default) so
        each env draws its own OS-entropy seed and the transform sequences
        are independent.
        """
        super().__init__()
        self.game = Fast2048()

        # Action space: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = Discrete(4)

        # Observation space: 1x4x4 grid with log2 values of the tiles.
        self.observation_space = Box(
            low=0, high=MAX_EXPONENT, shape=(1, 4, 4), dtype=np.int64
        )

        self.d4_augment = d4_augment
        self._d4_rng = np.random.default_rng(d4_seed)
        self._current_d4: int = ID

    def _sample_d4(self) -> int:
        if self.d4_augment:
            return int(self._d4_rng.integers(0, NUM_TRANSFORMS))
        return ID

    def _view_board(self) -> np.ndarray:
        """Return the canonical board under the current D4 transform."""
        if self._current_d4 == ID:
            return self.game.board
        return apply_d4(self.game.board, self._current_d4)

    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        self.game.reset()
        self._current_d4 = self._sample_d4()
        observation = board_to_tensor(self._view_board())
        info = {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step. Returns (obs, reward, done, truncated, info)."""
        canonical_action = int(ACTION_TO_CANONICAL[self._current_d4, action])

        merge_score, done, moved = self.game.move(canonical_action)

        self._current_d4 = self._sample_d4()

        observation = board_to_tensor(self._view_board())
        reward = calculate_reward(self.game.board, merge_score, moved)

        info = {}
        if done:
            info['max_tile'] = self.game.max_tile
            info['score'] = self.game.score

        truncated = False
        return observation, reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        """Get valid action mask in the current transformed view."""
        canonical = np.array(
            [self.game.is_move_valid(act) for act in range(self.action_space.n)],
            dtype=bool,
        )
        return transform_action_mask(canonical, self._current_d4)

    def render(self, mode='human'):
        """Render (unused)."""
        pass

    def close(self):
        """Close environment."""
        pass
