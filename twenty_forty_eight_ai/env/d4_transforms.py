"""D4 dihedral group symmetries for the 2048 board.

The 4x4 2048 board has 8 D4 symmetries: 4 rotations x 2 reflections.
This module exposes the 8 transforms and the corresponding action
permutations, so the training environment can present every board in
a random orientation. Training the value network to be invariant to all
D4 elements lets the C++ search canonicalize boards without biasing
leaf evaluations.

Transform indices:
    0  ID              identity
    1  ROT90_CW        90-degree clockwise rotation
    2  ROT180          180-degree rotation
    3  ROT270_CW       270-degree clockwise rotation
    4  REFLECT_H       horizontal-axis reflection (flip left-right)
    5  REFLECT_V       vertical-axis reflection (flip top-bottom)
    6  TRANSPOSE       main-diagonal reflection
    7  ANTI_TRANSPOSE  anti-diagonal reflection

Action encoding: 0 = Up, 1 = Right, 2 = Down, 3 = Left.
"""

import numpy as np


ID = 0
ROT90_CW = 1
ROT180 = 2
ROT270_CW = 3
REFLECT_H = 4
REFLECT_V = 5
TRANSPOSE = 6
ANTI_TRANSPOSE = 7

NUM_TRANSFORMS = 8


def apply_d4(board: np.ndarray, t: int) -> np.ndarray:
    """Apply D4 transform t to a (4, 4) integer board.

    Returns a new array; the input is not modified. Tile values are
    unchanged; only their positions are permuted.
    """
    if t == ID:
        return board.copy()
    if t == ROT90_CW:
        return np.rot90(board, -1).copy()
    if t == ROT180:
        return np.rot90(board, 2).copy()
    if t == ROT270_CW:
        return np.rot90(board, 1).copy()
    if t == REFLECT_H:
        return board[:, ::-1].copy()
    if t == REFLECT_V:
        return board[::-1, :].copy()
    if t == TRANSPOSE:
        return board.T.copy()
    if t == ANTI_TRANSPOSE:
        return board[::-1, ::-1].T.copy()
    raise ValueError(f"Invalid D4 transform index: {t}")


# Action permutation induced by each D4 element.
#
# ACTION_TO_CANONICAL[t][a] = the canonical action that, when applied to
# the canonical board, produces the same effect as applying action a to
# the board after transform t. Intuitively: if the agent presses "Up"
# while seeing the board through transform t, we translate that into the
# canonical action so the underlying game logic stays in canonical
# coordinates.
#
# Each row is a permutation of (0, 1, 2, 3). Verified by exhaustive
# single-tile step tests in tests/test_d4_transforms.py.
ACTION_TO_CANONICAL = np.array([
    [0, 1, 2, 3],   # ID
    [3, 0, 1, 2],   # ROT90_CW:        Up->Left, Right->Up, Down->Right, Left->Down
    [2, 3, 0, 1],   # ROT180:          Up->Down, Right->Left, Down->Up, Left->Right
    [1, 2, 3, 0],   # ROT270_CW:       Up->Right, Right->Down, Down->Left, Left->Up
    [0, 3, 2, 1],   # REFLECT_H:       Up->Up, Right->Left, Down->Down, Left->Right
    [2, 1, 0, 3],   # REFLECT_V:       Up->Down, Right->Right, Down->Up, Left->Left
    [3, 2, 1, 0],   # TRANSPOSE:       Up->Left, Right->Down, Down->Right, Left->Up
    [1, 0, 3, 2],   # ANTI_TRANSPOSE:  Up->Right, Right->Up, Down->Left, Left->Down
], dtype=np.int64)


# Inverse of each D4 element.
INVERSE_D4 = np.array([
    ID,          # ID
    ROT270_CW,   # ROT90_CW
    ROT180,      # ROT180
    ROT90_CW,    # ROT270_CW
    REFLECT_H,   # REFLECT_H (involutory)
    REFLECT_V,   # REFLECT_V (involutory)
    TRANSPOSE,   # TRANSPOSE (involutory)
    ANTI_TRANSPOSE,  # ANTI_TRANSPOSE (involutory)
], dtype=np.int64)


def inverse_d4(t: int) -> int:
    """Return the D4 element that undoes transform t."""
    return int(INVERSE_D4[t])


def transform_action_mask(canonical_mask: np.ndarray, t: int) -> np.ndarray:
    """Map a canonical-action validity mask to the transformed-view mask.

    The transformed mask is indexed by the agent's action choices in the
    transformed view. For action a in the transformed view, the canonical
    action actually executed is ACTION_TO_CANONICAL[t][a], so its validity
    is canonical_mask[ACTION_TO_CANONICAL[t][a]].
    """
    canonical_mask = np.asarray(canonical_mask)
    return canonical_mask[ACTION_TO_CANONICAL[t]].copy()
