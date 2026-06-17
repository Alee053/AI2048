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
    result = total / len(empty)  # CORRECT divisor N (matches d575532)
    if _old_trace_log is not None:
        canon = canonicalize_python(board)
        _old_trace_log.write(f"kind=chance depth={depth} board=0x{canon:x} "
                              f"value={result:.9f} src=computed\n")
    return result


def old_max_node(board: np.ndarray, depth: int, leaf_cache: dict) -> float:
    if depth == 0:
        key = tuple(board.flatten().tolist())
        val = leaf_cache.get(key, -100.0)
        if _old_trace_log is not None:
            canon = canonicalize_python(board)
            _old_trace_log.write(f"kind=max depth=0 board=0x{canon:x} "
                                  f"value={val:.9f} src=leaf_cache\n")
        return val
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
    result = max_v if any_move else 0.0
    if _old_trace_log is not None:
        canon = canonicalize_python(board)
        _old_trace_log.write(f"kind=max depth={depth} board=0x{canon:x} "
                              f"value={result:.9f} src=computed\n")
    return result


# Module-level trace log handle for the OLD. None = no tracing.
_old_trace_log = None


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
    # Return leaves in canonical form for direct comparison with the NEW's dump.
    leaf_keys_canon = {canonicalize_python(b) for b in leaves}
    return (best_move if best_move != -1 else 0), move_scores, leaf_keys_canon, leaves


# Pure-Python canonical form of a 4x4 board (16-cell, log2 values).
# Mirrors BoardEncoder::canonicalize in C++: pack 4 bits per cell, then take the
# minimum over 8 D4 symmetries (4 rotations x 2 reflections).
def pack_python(board: np.ndarray) -> int:
    p = 0
    for r in range(4):
        for c in range(4):
            p |= (int(board[r, c]) & 0xF) << ((r * 4 + c) * 4)
    return p


def symmetries_python(p: int) -> list[int]:
    """All 8 D4 symmetries of a packed 4x4 board (cell-major 4 bits)."""
    def rot90(p: int) -> int:
        # (r,c) -> (c, 3-r) ; cell-major: index = r*4+c
        out = 0
        for r in range(4):
            for c in range(4):
                v = (p >> ((r * 4 + c) * 4)) & 0xF
                out |= v << ((c * 4 + (3 - r)) * 4)
        return out

    def reflect_h(p: int) -> int:
        out = 0
        for r in range(4):
            for c in range(4):
                v = (p >> ((r * 4 + c) * 4)) & 0xF
                out |= v << ((r * 4 + (3 - c)) * 4)
        return out

    out = [p]
    for _ in range(3):
        out.append(rot90(out[-1]))
    reflected = reflect_h(out[0])
    out.append(reflected)
    r = reflected
    for _ in range(3):
        r = rot90(r)
        out.append(r)
    return out


def canonicalize_python(board: np.ndarray) -> int:
    return min(symmetries_python(pack_python(board)))


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
    leaves_dump = s.dump_leaves()
    leaves_set = {}
    for line in leaves_dump.splitlines()[1:]:
        key_str, val_str = line.split(" ", 1)
        leaves_set[int(key_str, 16)] = float(val_str)
    return (int(stats.best_move), move_scores,
            int(stats.batches_eval), int(stats.nodes_visited), int(stats.tt_lookups),
            int(stats.tt_hits), int(stats.tt_collisions),
            int(stats.chance_nodes_evaluated), int(stats.max_nodes_evaluated),
            leaves_set, int(stats.unique_leaves_evaluated))


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
    ap.add_argument("--old-trace", default=None,
                    help="Path to write the OLD's search-tree trace (chance_node, max_node values). "
                         "Enables direct comparison with the NEW's C++ trace.")
    ap.add_argument("--new-trace", default=None,
                    help="Path to write the NEW's search-tree trace. Requires --debug. "
                         "The NEW is traced via set_trace_log() on the C++ searcher.")
    ap.add_argument("--trace-board", type=int, default=26,
                    help="When --old-trace or --new-trace is set, only run the trace on this board index "
                         "(0-based). Default 26, which has a 22-point gap.")
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
        # Optionally open the OLD's trace log only for the chosen trace_board
        old_trace_open = args.old_trace and i == args.trace_board
        if old_trace_open:
            global _old_trace_log
            _old_trace_log = open(args.old_trace, "w")
        try:
            if args.debug:
                old_move, old_scores, old_leaves_canon, old_leaves_raw = old_find_best_move_debug(board, args.depth, eval_old)
            else:
                old_move, old_scores = old_find_best_move(board, args.depth, eval_old)
                old_leaves_canon = old_leaves_raw = None
        finally:
            if old_trace_open:
                _old_trace_log.close()
                _old_trace_log = None
        old_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        new_trace_open = args.new_trace and i == args.trace_board
        if new_trace_open:
            # We need to instrument the C++ searcher. The Python wrapper
            # exposes _impl which is the actual C++ class. Open the trace
            # on a fresh searcher instance, run, close.
            new_impl = _load_searcher_impl().ExpectimaxSearcher(32768)
            new_impl.set_trace_log(args.new_trace)
            try:
                stats = new_impl.find_best_move(board, args.depth, eval_new)
                new_move = int(stats.best_move)
                new_scores = list(stats.move_scores)
                if args.debug:
                    new_batches = int(stats.batches_eval)
                    new_nodes = int(stats.nodes_visited)
                    new_lookups = int(stats.tt_lookups)
                    new_hits = int(stats.tt_hits)
                    new_coll = int(stats.tt_collisions)
                    new_chance = int(stats.chance_nodes_evaluated)
                    new_max = int(stats.max_nodes_evaluated)
                    new_leaves_dump = new_impl.dump_leaves()
                    new_leaves = {}
                    for line in new_leaves_dump.splitlines()[1:]:
                        k_str, v_str = line.split(" ", 1)
                        new_leaves[int(k_str, 16)] = float(v_str)
                    new_unique_leaves = int(stats.unique_leaves_evaluated)
            finally:
                new_impl.close_trace_log()
        elif args.debug:
            (new_move, new_scores, new_batches, new_nodes, new_lookups, new_hits,
             new_coll, new_chance, new_max, new_leaves, new_unique_leaves) = new_find_best_move_debug(board, args.depth, eval_new)
        else:
            new_move, new_scores = new_find_best_move(board, args.depth, eval_new)
            new_leaves = new_batches = None
            new_unique_leaves = 0
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
            # Leaf-set comparison (OLD raw leaves canonicalized -> compare to NEW's canonical set)
            new_leaf_keys = set(new_leaves.keys()) if new_leaves else set()
            only_old = old_leaves_canon - new_leaf_keys if old_leaves_canon else set()
            only_new = new_leaf_keys - (old_leaves_canon or set())
            common = (old_leaves_canon & new_leaf_keys) if old_leaves_canon else set()
            # Value-disagreement on common leaves
            value_diffs = 0
            max_value_diff = 0.0
            for k in common:
                # Get OLD's value for this canonical key from the eval_fn
                # We don't have it cached; re-evaluate would slow things down. Skip for now.
                pass
            debug_str = (f"   old_leaves={len(old_leaves_canon):>5}  new_uniq={new_unique_leaves:>5}  "
                         f"only_old={len(only_old):>4} only_new={len(only_new):>4}  common={len(common):>5}")
        print(f"{i:>2}  {old_move:>5}  {new_move:>5}  {match}     {os:>11.4f}  {ns:>11.4f}  {ns-os:>+8.4f}{fresh_str}"
              f"   (old {old_t:.1f}s, new {new_t:.1f}s){debug_str}")
    print(f"\nMove agreement (OLD vs NEW):       {same_move}/{len(boards)} = {100*same_move/len(boards):.0f}%")
    print(f"Score agreement (OLD vs NEW):      {same_score}/{len(boards)} = {100*same_score/len(boards):.0f}%")
    if args.fresh_tt:
        print(f"Move agreement (OLD vs NEW fresh-tt): {same_move_fresh}/{len(boards)} = {100*same_move_fresh/len(boards):.0f}%")
        print(f"Score agreement (OLD vs NEW fresh-tt): {same_score_fresh}/{len(boards)} = {100*same_score_fresh/len(boards):.0f}%")


if __name__ == "__main__":
    sys.exit(main())
