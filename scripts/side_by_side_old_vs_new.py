"""Side-by-side comparison: OLD "first implementation" expectimax vs NEW (current) searcher.

The OLD algorithm (d575532, Dec 9 2025 — what actually ran the depth3_expectimax
benchmark that scored 26,523 +/- 12,750 on n=100) is much simpler than the NEW:

  - gather_leaves: recursively collect UNIQUE boards at depth 0 (uses std::map<Board,bool>
    for dedup; no hashing).
  - chance_node_substitute: weighted sum / empty_cells.size()  (divisor N, correct).
  - max_node_substitute: max over 4 moves, NO alpha-beta, NO TT (just uses the
    precomputed leaf_cache via depth 0 lookup).
  - find_best_move: gather leaves, batch-eval them all in one call, then search.

The NEW (current master + fix-chance-divisor branch) has:
  - Multi-pass deferred batching (interleaves search and batch eval, no
    gather-all-first).
  - Persistent TT with cross-search aging.
  - Correct divisor (was 2N, now N after the fix-chance-divisor fix).
  - Alpha-beta pruning at max_node (never fires, but the code path is there).
  - Per-move iteration cap (100).

The two implementations share the Fast2048 game logic and the model. This script
reimplements the OLD's algorithm in pure Python (slow but exact) and compares
chosen moves / move_scores against the NEW's C++ implementation for fixed boards.

If the two choose the same move for the same board, the score difference is from
game-state evolution (stochastic), not from algorithmic differences. If they
choose different moves, we have a real algorithmic regression to investigate.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor


# ---------------------------------------------------------------------------
# OLD "first implementation" (d575532, Dec 9 2025) reimplemented in pure Python
# ---------------------------------------------------------------------------

# A few board movements: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
def move_board_python(board: np.ndarray, direction: int) -> tuple[np.ndarray, int, bool]:
    """Pure-Python move (simulated, no random spawn).

    Returns (new_board, merge_score, moved). The new_board is a fresh array;
    the input is not modified.
    """
    grid = board.tolist()
    score = 0
    moved = False
    for r in range(4):
        if direction == 0:  # UP: take column r
            col = [grid[i][r] for i in range(4)]
        elif direction == 1:  # DOWN: take column r reversed
            col = [grid[i][r] for i in range(3, -1, -1)]
        elif direction == 2:  # LEFT: take row r
            col = grid[r][:]
        else:  # 3 RIGHT: take row r reversed
            col = grid[r][::-1]

        # compact, merge, pad
        compact = [v for v in col if v != 0]
        merged = []
        i = 0
        while i < len(compact):
            if i + 1 < len(compact) and compact[i] == compact[i + 1]:
                v = compact[i] + 1
                score += 1 << v
                merged.append(v)
                i += 2
            else:
                merged.append(compact[i])
                i += 1
        merged = merged + [0] * (4 - len(merged))

        if direction == 0:
            for i in range(4):
                if grid[i][r] != merged[i]:
                    moved = True
                grid[i][r] = merged[i]
        elif direction == 1:
            for i in range(4):
                if grid[3 - i][r] != merged[i]:
                    moved = True
                grid[3 - i][r] = merged[i]
        elif direction == 2:
            new_row = merged
            if grid[r] != new_row:
                moved = True
            grid[r] = new_row
        else:
            new_row = list(reversed(merged))
            if grid[r] != new_row:
                moved = True
            grid[r] = new_row
    return np.array(grid, dtype=np.int32), score, moved


def get_log_reward(merge_score: int) -> float:
    if merge_score <= 0:
        return 0.0
    return float(np.log2(merge_score))


def old_gather_leaves(board: np.ndarray, depth: int,
                      leaves: list[np.ndarray], seen: set) -> None:
    """Recursively collect all unique leaf boards at depth 0 (or game-over boards)."""
    key = tuple(board.flatten().tolist())
    if depth == 0:
        if key not in seen:
            seen.add(key)
            leaves.append(board.copy())
        return
    for move in range(4):
        post, _, moved = move_board_python(board, move)
        if not moved:
            continue
        empty = [(r, c) for r in range(4) for c in range(4) if post[r, c] == 0]
        if not empty:
            post_key = tuple(post.flatten().tolist())
            if post_key not in seen:
                seen.add(post_key)
                leaves.append(post.copy())
            continue
        for (r, c) in empty:
            # spawn 2 (log2 = 1)
            nb2 = post.copy()
            nb2[r, c] = 1
            old_gather_leaves(nb2, depth - 1, leaves, seen)
            # spawn 4 (log2 = 2)
            nb4 = post.copy()
            nb4[r, c] = 2
            old_gather_leaves(nb4, depth - 1, leaves, seen)


def old_chance_node(board: np.ndarray, depth: int, leaf_cache: dict) -> float:
    empty = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]
    if not empty:
        return old_max_node(board, depth - 1, leaf_cache)
    total = 0.0
    for (r, c) in empty:
        nb2 = board.copy(); nb2[r, c] = 1
        total += 0.9 * old_max_node(nb2, depth - 1, leaf_cache)
        nb4 = board.copy(); nb4[r, c] = 2
        total += 0.1 * old_max_node(nb4, depth - 1, leaf_cache)
    return total / len(empty)  # CORRECT divisor N (matches d575532)


def old_max_node(board: np.ndarray, depth: int, leaf_cache: dict) -> float:
    if depth == 0:
        key = tuple(board.flatten().tolist())
        return leaf_cache.get(key, -100.0)
    max_v = -1e9
    any_move = False
    for move in range(4):
        post, ms, moved = move_board_python(board, move)
        if not moved:
            continue
        any_move = True
        reward = get_log_reward(ms)
        fv = old_chance_node(post, depth, leaf_cache)
        total = reward + fv
        if total > max_v:
            max_v = total
    return max_v if any_move else 0.0


def old_find_best_move(board: np.ndarray, depth: int, batch_eval_fn: Callable) -> tuple[int, list[float]]:
    leaves: list[np.ndarray] = []
    seen: set = set()
    old_gather_leaves(board, depth, leaves, seen)
    if leaves:
        evals = batch_eval_fn(leaves)
    else:
        evals = []
    leaf_cache = {tuple(b.flatten().tolist()): float(v) for b, v in zip(leaves, evals)}

    best_score = -1e9
    best_move = -1
    move_scores = [-1e9, -1e9, -1e9, -1e9]
    for move in range(4):
        post, ms, moved = move_board_python(board, move)
        if not moved:
            continue
        reward = get_log_reward(ms)
        fv = old_chance_node(post, depth, leaf_cache)
        score = reward + fv
        move_scores[move] = score
        if score > best_score:
            best_score = score
            best_move = move
    return (best_move if best_move != -1 else 0), move_scores


# A debug variant: also returns the leaves and the leaf-eval dict
def old_find_best_move_debug(board: np.ndarray, depth: int, batch_eval_fn: Callable):
    leaves: list[np.ndarray] = []
    seen: set = set()
    old_gather_leaves(board, depth, leaves, seen)
    if leaves:
        evals = batch_eval_fn(leaves)
    else:
        evals = []
    leaf_cache = {tuple(b.flatten().tolist()): float(v) for b, v in zip(leaves, evals)}
    best_score = -1e9
    best_move = -1
    move_scores = [-1e9, -1e9, -1e9, -1e9]
    for move in range(4):
        post, ms, moved = move_board_python(board, move)
        if not moved:
            continue
        reward = get_log_reward(ms)
        fv = old_chance_node(post, depth, leaf_cache)
        score = reward + fv
        move_scores[move] = score
        if score > best_score:
            best_score = score
            best_move = move
    leaf_keys = {tuple(b.flatten().tolist()) for b in leaves}
    return (best_move if best_move != -1 else 0), move_scores, leaf_keys, leaf_cache


# ---------------------------------------------------------------------------
# NEW (current): the C++ binding already loaded by the main app
# ---------------------------------------------------------------------------

# We'll import the actual production class via importlib, same trick the
# production wrapper uses, so we exercise the same .so.
def _load_searcher_impl():
    """Load the compiled C++ searcher via the same importlib trick the production wrapper uses.

    Note: the .so is registered under the pybind11 module name 'searcher', not the
    file basename. spec_from_file_location must use that name or pybind11's
    PyInit_<name> symbol won't be found.
    """
    so_path = (Path(__file__).resolve().parent.parent / "twenty_forty_eight_ai" / "utils" /
               "_searcher_cpp.cpython-312-x86_64-linux-gnu.so")
    spec = importlib.util.spec_from_file_location("searcher", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load C++ searcher module from {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_find_best_move(board: np.ndarray, depth: int, batch_eval_fn: Callable) -> tuple[int, list[float]]:
    impl = _load_searcher_impl()
    s = impl.ExpectimaxSearcher(32768)
    stats = s.find_best_move(board, depth, batch_eval_fn)
    move_scores = list(stats.move_scores)
    return int(stats.best_move), move_scores


# Debug variant: returns (best_move, move_scores, batches_count, leaves_count)
# by calling find_best_move twice — once for stats, once for the actual search
# (we can't directly inspect the leaves from Python; this is best-effort).
def new_find_best_move_debug(board: np.ndarray, depth: int, batch_eval_fn: Callable):
    impl = _load_searcher_impl()
    s = impl.ExpectimaxSearcher(32768)
    stats = s.find_best_move(board, depth, batch_eval_fn)
    move_scores = list(stats.move_scores)
    return (int(stats.best_move), move_scores,
            int(stats.batches_eval), int(stats.nodes_visited), int(stats.tt_lookups),
            int(stats.tt_hits), int(stats.tt_collisions),
            int(stats.chance_nodes_evaluated), int(stats.max_nodes_evaluated))


def new_find_best_move_fresh_tt(board: np.ndarray, depth: int, batch_eval_fn: Callable) -> tuple[int, list[float]]:
    """NEW with persistent TT disabled (clear_tt between calls). This matches
    the OLD's per-search behavior. If scores and moves now agree with the OLD,
    the persistent TT (with cross-search aging) was the cause of the difference."""
    impl = _load_searcher_impl()
    s = impl.ExpectimaxSearcher(32768)
    s.clear_tt()
    stats = s.find_best_move(board, depth, batch_eval_fn)
    move_scores = list(stats.move_scores)
    return int(stats.best_move), move_scores


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_eval_fn(model: MaskablePPO, device: str) -> Callable:
    def eval_fn(boards_list):
        if not boards_list:
            return []
        arr = np.array(boards_list, dtype=np.int32)
        bt = board_to_tensor(arr)
        with torch.no_grad():
            values = model.policy.predict_values(torch.as_tensor(bt).to(device))
        return values.cpu().numpy().flatten().tolist()
    return eval_fn


def make_eval_fn_fast(model: MaskablePPO, device: str) -> Callable:
    """Same as make_eval_fn but accepts the NEW C++ batch shape (np.ndarray)."""
    def eval_fn(boards):
        if not isinstance(boards, np.ndarray):
            boards = np.array(boards, dtype=np.int32)
        bt = board_to_tensor(boards)
        with torch.no_grad():
            values = model.policy.predict_values(torch.as_tensor(bt).to(device))
        return values.cpu().numpy().flatten().tolist()
    return eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/release/Hybrid-PPO-Expectimax-v1.zip")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--boards", type=int, default=10,
                    help="Number of synthetic boards to compare on")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh-tt", action="store_true",
                    help="Also run NEW with clear_tt() between boards. If this matches the OLD, "
                         "the persistent TT (with cross-search aging) was the cause of the difference.")
    ap.add_argument("--debug", action="store_true",
                    help="Compare leaf sets (OLD can dump them; NEW can't directly), and dump per-board "
                         "search stats. Helps isolate whether the leaf sets differ or the search aggregation does.")
    args = ap.parse_args()

    model = MaskablePPO.load(args.model, device=args.device)
    device = model.device
    print(f"Loaded model on {device}", flush=True)

    # Boards: same as test_transposition_table.py
    rng = np.random.default_rng(args.seed)
    boards = []
    for _ in range(args.boards):
        b = rng.integers(0, 12, size=(4, 4), dtype=np.int32)
        if not (b == 0).any():
            b[0, 0] = 0
        boards.append(b)

    eval_old = make_eval_fn(model, device)
    eval_new = make_eval_fn_fast(model, device)

    same_move = 0
    same_score = 0
    same_move_fresh = 0
    same_score_fresh = 0
    print(f"\nComparing OLD vs NEW on {len(boards)} random boards at depth {args.depth}\n")
    header = f"{'#':>2}  {'OLD':>5}  {'NEW':>5}  {'NEWf':>5}  match  match_f  {'OLD_score':>11}  {'NEW_score':>11}  {'NEWf_score':>11}"
    print(header)
    for i, board in enumerate(boards):
        t0 = time.perf_counter()
        if args.debug:
            old_move, old_scores, old_leaves, old_leaf_cache = old_find_best_move_debug(board, args.depth, eval_old)
        else:
            old_move, old_scores = old_find_best_move(board, args.depth, eval_old)
            old_leaves = old_leaf_cache = None
        old_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        if args.debug:
            new_move, new_scores, new_batches, new_nodes, new_lookups, new_hits, new_coll, new_chance, new_max = new_find_best_move_debug(board, args.depth, eval_new)
        else:
            new_move, new_scores = new_find_best_move(board, args.depth, eval_new)
        new_t = time.perf_counter() - t0
        fresh_move = None
        fresh_scores = None
        if args.fresh_tt:
            t0 = time.perf_counter()
            fresh_move, fresh_scores = new_find_best_move_fresh_tt(board, args.depth, eval_new)
            fresh_t = time.perf_counter() - t0
        match = "==" if old_move == new_move else "!!"
        match_f = ("==" if old_move == fresh_move else "!!") if fresh_move is not None else "  "
        if old_move == new_move:
            same_move += 1
        if fresh_move is not None and old_move == fresh_move:
            same_move_fresh += 1
        os = old_scores[old_move] if old_move >= 0 else -1e9
        ns = new_scores[new_move] if new_move >= 0 else -1e9
        fs = fresh_scores[old_move] if (fresh_scores and old_move >= 0) else -1e9
        if abs(os - ns) < 0.01:
            same_score += 1
        if fresh_scores and abs(os - fs) < 0.01:
            same_score_fresh += 1
        fresh_str = f"  {fresh_move:>5}  {match_f}  {fs:>11.4f}" if fresh_move is not None else ""
        debug_str = ""
        if args.debug:
            debug_str = (f"   leaves={len(old_leaves):>5}  new[batches={new_batches:>3} "
                         f"chance={new_chance:>9,} max={new_max:>9,} lookups={new_lookups:>9,} "
                         f"hits={new_hits:>9,} coll={new_coll:>9,}]")
        print(f"{i:>2}  {old_move:>5}  {new_move:>5}  {match}     {os:>11.4f}  {ns:>11.4f}  {ns-os:>+8.4f}{fresh_str}"
              f"   (old {old_t:.1f}s, new {new_t:.1f}s){debug_str}")
    print(f"\nMove agreement (OLD vs NEW):       {same_move}/{len(boards)} = {100*same_move/len(boards):.0f}%")
    print(f"Score agreement (OLD vs NEW):      {same_score}/{len(boards)} = {100*same_score/len(boards):.0f}%")
    if args.fresh_tt:
        print(f"Move agreement (OLD vs NEW fresh-tt): {same_move_fresh}/{len(boards)} = {100*same_move_fresh/len(boards):.0f}%")
        print(f"Score agreement (OLD vs NEW fresh-tt): {same_score_fresh}/{len(boards)} = {100*same_score_fresh/len(boards):.0f}%")


if __name__ == "__main__":
    sys.exit(main())
