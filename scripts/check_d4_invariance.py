"""Rotation/reflection invariance check for the retrained value network.

For 100 random boards, compares the value network's output on each board
against the output on all 7 of its non-identity D4 transforms. The fix
for the depth-3 regression depends on the model being (approximately)
D4-invariant, so this is the actual regression test for the root cause.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from twenty_forty_eight_ai.env.d4_transforms import (
    apply_d4,
    ID,
    ROT90_CW,
    ROT180,
    ROT270_CW,
    REFLECT_H,
    REFLECT_V,
    TRANSPOSE,
    ANTI_TRANSPOSE,
    NUM_TRANSFORMS,
)
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor


MODEL_PATH = "data/models/release/Hybrid-PPO-Expectimax-v3.zip"
N_BOARDS = 100
TOLERANCE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORMS = [
    ("ID", ID),
    ("ROT90_CW", ROT90_CW),
    ("ROT180", ROT180),
    ("ROT270_CW", ROT270_CW),
    ("REFLECT_H", REFLECT_H),
    ("REFLECT_V", REFLECT_V),
    ("TRANSPOSE", TRANSPOSE),
    ("ANTI_TRANSPOSE", ANTI_TRANSPOSE),
]


def make_random_board(rng: np.random.Generator) -> np.ndarray:
    """Make a 4x4 board that resembles a real 2048 mid-game state.

    Real game boards are sparse (mostly empty) with low-valued tiles
    dominating. Uniform random 0..15 makes nearly every cell a large
    power of 2, which the model has never seen in training and which
    produces extreme, unrepresentative value estimates.
    """
    board = np.zeros((4, 4), dtype=np.int32)
    # Pick how many non-empty cells (4..10, biased toward sparse).
    n_filled = int(rng.integers(4, 11))
    cells = rng.choice(16, size=n_filled, replace=False)
    for c in cells:
        # Tile log2 value: 90% small (1=2, 2=4, 3=8), 9% medium (4..7),
        # 1% large (8..14).
        r = rng.random()
        if r < 0.90:
            v = int(rng.integers(1, 4))
        elif r < 0.99:
            v = int(rng.integers(4, 8))
        else:
            v = int(rng.integers(8, 15))
        board[c // 4, c % 4] = v
    return board


def evaluate(model: MaskablePPO, board: np.ndarray) -> float:
    """Return the value-network output for one board."""
    obs = board_to_tensor(board)
    with torch.no_grad():
        v = model.policy.predict_values(torch.as_tensor(obs).to(DEVICE))
    return float(v.cpu().numpy().item())


def main() -> int:
    print(f"Loading model: {MODEL_PATH}")
    new_model = MaskablePPO.load(MODEL_PATH, device=DEVICE)
    old_model = None
    old_path = None
    # The OLD release has been replaced by the retrained model on the
    # fix-regression branch; if a backup exists at a known path, load
    # it for comparison. Otherwise the baseline is unavailable.
    candidate_old = Path("data/models/release/Hybrid-PPO-Expectimax-v3.zip.bak")
    if candidate_old.exists() and candidate_old.resolve() != Path(MODEL_PATH).resolve():
        old_path = str(candidate_old)
    elif Path(MODEL_PATH).name == "Hybrid-PPO-Expectimax-v3.zip":
        # The release model now IS the retrained model. Look for a
        # pre-retraining backup if one was kept aside.
        backup = Path("data/models/_pre_d4_baseline.zip")
        if backup.exists():
            old_path = str(backup)
    if old_path is not None:
        print(f"Loading OLD baseline: {old_path}")
        old_model = MaskablePPO.load(old_path, device=DEVICE)
    else:
        print("No OLD baseline available (release model was replaced by retrain).")
    print(f"Device: {DEVICE}")
    print(f"Boards: {N_BOARDS}, tolerance: {TOLERANCE}")
    print()

    rng = np.random.default_rng(42)

    def run_for_model(model: MaskablePPO, label: str) -> tuple[bool, float]:
        local_diffs = {name: [] for name, _ in TRANSFORMS if _ != ID}
        local_max = []
        for _ in range(N_BOARDS):
            board = make_random_board(rng)
            v_orig = evaluate(model, board)
            local_max.append(0.0)
            for name, t in TRANSFORMS:
                if t == ID:
                    continue
                v_t = evaluate(model, apply_d4(board, t))
                d = abs(v_t - v_orig)
                local_diffs[name].append(d)
                local_max[-1] = max(local_max[-1], d)
        worst = max(local_max)
        ok = worst < TOLERANCE
        print(f"\n=== {label} ===")
        print(f"{'transform':<16} {'mean':>10} {'max':>10} {'pass':>6}")
        print("-" * 46)
        for name, _ in TRANSFORMS:
            if name == "ID":
                continue
            diffs = np.array(local_diffs[name])
            m, mx = float(diffs.mean()), float(diffs.max())
            t_ok = mx < TOLERANCE
            print(f"{name:<16} {m:>10.5f} {mx:>10.5f} {'PASS' if t_ok else 'FAIL':>6}")
        print("-" * 46)
        print(f"{'WORST PER-BOARD':<16} {'':>10} {worst:>10.5f}")
        print(f"{'TOLERANCE':<16} {'':>10} {TOLERANCE:>10.5f}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return ok, worst

    new_ok, new_worst = run_for_model(new_model, "NEW (D4-augmented)")
    old_ok, old_worst = (None, None)
    if old_model is not None:
        old_ok, old_worst = run_for_model(old_model, "OLD (release, baseline)")

    print()
    if old_worst is not None:
        print(f"NEW worst: {new_worst:.5f}   OLD worst: {old_worst:.5f}   tol: {TOLERANCE}")
    if new_ok:
        print("OVERALL: NEW model is D4-invariant; ready to replace release.")
        return 0
    if old_worst is not None and new_worst < old_worst:
        print("OVERALL: NEW is more invariant than OLD but still exceeds tolerance — train longer.")
        return 1
    if old_worst is not None:
        print("OVERALL: NEW is no better than OLD — augmentation didn't take effect.")
    else:
        print("OVERALL: model not D4-invariant; train longer.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
