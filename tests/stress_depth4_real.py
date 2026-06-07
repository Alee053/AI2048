"""
Real-model depth-4 stress test.

Two modes:
  1. Single search (default) — runs depth-N search on a chosen board and
     reports stats + warning count. Useful for quick iteration on a board
     snapshot. <5s typical.
  2. Cumulative (--cumulative N) — runs N consecutive depth-N searches on
     a single searcher, simulating a game. Reproduces the cumulative TT
     thrashing pathology that the real benchmark exhibits: TT grows
     unboundedly across moves and collision count explodes.

The fake-evaluator convergence test (test_depth4_convergence.py) gives
false confidence because the fake evaluator produces a deterministic,
highly-cacheable value surface; real neural net values don't have the
same dedup properties, so the TT thrashes on real boards.

Usage:
    python tests/stress_depth4_real.py [OPTIONS]

Options:
    --depth N             Search depth (default: 4)
    --model PATH          Path to .zip model (default: release model)
    --play-to-midgame N   If >0, play N moves with raw PPO policy to generate
                          the test board instead of using a hardcoded one.
                          (Ignored in --cumulative mode; the env plays itself.)
    --cumulative N        Run N consecutive depth-N searches on a single
                          searcher. Reproduces cumulative TT thrashing.
    --max-seconds S       Stop cumulative run after S seconds (default: 60).
    --seed S              Seed (default: 42)
    --target-batch-size N Pass-through to ExpectimaxSearcher (default: 32768)
    --board NAME          Preset: 'hardcoded' (default) | 'convergence' (the
                          easy board from test_depth4_convergence.py)

Exit code:
    0  Search converged (no cap hits)
    1  Cap hit at least once — search did not converge
    2  Interrupted (Ctrl-C)
"""
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running directly: python tests/stress_depth4_real.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor


# Boards store log2(tile), so 10 = 1024, 9 = 512, 8 = 256, etc.
PRESETS = {
    "hardcoded": np.array([
        # Mid-game: 1024 in corner, 512/256 scattered, no easy merges
        [10,  8,  7,  0],
        [ 0,  9,  6,  5],
        [ 6,  0,  8,  4],
        [ 5,  4,  3,  2],
    ], dtype=np.int32),
    "convergence": np.array([
        [1, 2, 1, 0],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ], dtype=np.int32),
}


def play_to_midgame(model, n_moves, seed):
    """Use raw PPO policy to play n_moves, return the final board state."""
    env = Game2048Env()
    obs, _ = env.reset(seed=seed)
    for i in range(n_moves):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        obs, _, done, _, _ = env.step(int(action))
        if done:
            obs, _ = env.reset(seed=seed + i + 1)
    return env.game.board.copy()


def make_evaluator(model, device):
    def evaluator(boards):
        tensor = board_to_tensor(np.array(boards))
        with torch.no_grad():
            values = model.policy.predict_values(torch.as_tensor(tensor).to(device))
        return values.cpu().numpy().flatten().tolist()
    return evaluator


def run_single(model, args, board):
    """Single depth-N search on a board. Returns (stats, warning_count, elapsed)."""
    evaluator = make_evaluator(model, model.device)
    searcher = ExpectimaxSearcher(target_batch_size=args.target_batch_size)
    stderr_capture = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stderr(stderr_capture):
        stats = searcher.find_best_move(board, args.depth, evaluator)
    elapsed = time.perf_counter() - t0
    warnings = [w for w in stderr_capture.getvalue().splitlines() if "[WARNING]" in w]
    return stats, len(warnings), elapsed


def run_cumulative(model, args):
    """Run N consecutive depth-N searches on a single searcher. Returns aggregate stats."""
    evaluator = make_evaluator(model, model.device)
    env = Game2048Env()
    obs, _ = env.reset(seed=args.seed)
    # Warm up to mid-game so cumulative stats reflect real-game conditions
    for _ in range(100):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        obs, _, done, _, _ = env.step(int(action))
        if done:
            obs, _ = env.reset(seed=args.seed)

    searcher = ExpectimaxSearcher(target_batch_size=args.target_batch_size)
    print(f"Running {args.cumulative} cumulative depth-{args.depth} searches "
          f"(budget: {args.max_seconds}s)...\n")
    print(f"{'move':>5} {'time':>7} {'nodes':>10} {'tt_size':>10} {'hits%':>5} "
          f"{'coll':>8} {'cap':>4} {'unr':>4} {'batch':>6}")

    rows = []
    total_warnings = 0
    cumulative_t0 = time.perf_counter()
    stopped_early = False

    try:
        for i in range(args.cumulative):
            t0 = time.perf_counter()
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                stats = searcher.find_best_move(env.game.board, args.depth, evaluator)
            dt = time.perf_counter() - t0
            total_warnings += len([w for w in stderr_capture.getvalue().splitlines() if "[WARNING]" in w])
            obs, _, done, _, _ = env.step(int(stats['best_move']))
            if done:
                obs, _ = env.reset(seed=args.seed)

            lk = max(1, stats.get('tt_lookups', 1))
            hr = stats.get('tt_hits', 0) / lk * 100
            row = (i, dt, stats['nodes_visited'], stats['tt_size'], hr,
                   stats['tt_collisions'], stats['cap_hits'],
                   stats['moves_unresolved'], stats['batches_eval'])
            rows.append(row)
            if i < 5 or i % 10 == 9 or dt > 5:
                print(f"{row[0]:>5} {row[1]:>6.2f}s {row[2]:>10,} {row[3]:>10,} "
                      f"{row[4]:>4.1f}% {row[5]:>8,} {row[6]:>4} {row[7]:>4} {row[8]:>6}")
            elapsed_total = time.perf_counter() - cumulative_t0
            if elapsed_total > args.max_seconds:
                print(f"--- stopped at move {i} after {elapsed_total:.1f}s (budget {args.max_seconds}s) ---")
                stopped_early = True
                break
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] at move {len(rows)}")
        stopped_early = True

    if not rows:
        print("No moves completed.")
        return None

    # Aggregate
    total_nodes = sum(r[2] for r in rows)
    final_tt_size = rows[-1][3]
    total_coll = sum(r[5] for r in rows)
    total_cap = sum(r[6] for r in rows)
    total_unr = sum(r[7] for r in rows)
    print()
    print("=" * 56)
    print(f"  CUMULATIVE SUMMARY ({len(rows)} moves)")
    print("=" * 56)
    print(f"  final tt_size:       {final_tt_size:>10,}")
    print(f"  total nodes:         {total_nodes:>10,}")
    print(f"  total collisions:    {total_coll:>10,}")
    print(f"  total cap_hits:      {total_cap:>10,}")
    print(f"  total unresolved:    {total_unr:>10,}")
    print(f"  total warnings:      {total_warnings:>10,}")
    print("=" * 56)
    return {
        "moves": len(rows),
        "final_tt_size": final_tt_size,
        "total_collisions": total_coll,
        "total_cap_hits": total_cap,
        "total_warnings": total_warnings,
        "stopped_early": stopped_early,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="data/models/release/Hybrid-PPO-Expectimax-v1.zip")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--play-to-midgame", type=int, default=0, metavar="N")
    parser.add_argument("--cumulative", type=int, default=0, metavar="N",
                        help="If >0, run N consecutive depth-N searches on one searcher.")
    parser.add_argument("--max-seconds", type=int, default=60,
                        help="Time budget for --cumulative mode (default: 60).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-batch-size", type=int, default=32768)
    parser.add_argument("--board", choices=PRESETS.keys(), default="hardcoded")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model)

    if args.cumulative > 0:
        summary = run_cumulative(model, args)
        if summary is None:
            sys.exit(2)
        # Exit 1 if collisions grew unboundedly (TT didn't evict)
        if summary["stopped_early"] and summary["final_tt_size"] > 5_000_000:
            print(f"\n[FAIL] TT grew to {summary['final_tt_size']:,} entries — eviction not keeping up")
            sys.exit(1)
        sys.exit(0)

    if args.play_to_midgame > 0:
        print(f"Playing {args.play_to_midgame} moves with raw policy (seed={args.seed})...")
        board = play_to_midgame(model, args.play_to_midgame, args.seed)
    else:
        board = PRESETS[args.board].copy()

    max_tile_log = int(board.max())
    nonzero = int((board != 0).sum())
    print(f"\nTest board (max tile = 2^{max_tile_log} = {1 << max_tile_log}, filled = {nonzero}/16):")
    print(board)
    print()

    print(f"Running depth-{args.depth} search...")
    try:
        stats, warning_count, elapsed = run_single(model, args, board)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(2)

    print()
    print("=" * 56)
    print(f"  DEPTH-{args.depth} STRESS TEST RESULTS")
    print("=" * 56)
    print(f"  wall time:           {elapsed:7.2f}s")
    print(f"  warnings (cap hits): {warning_count:7d}")
    print(f"  best_move:           {stats.get('best_move', 'N/A')}")
    print(f"  move_scores:         {stats.get('move_scores', [])}")
    print(f"  moves_resolved:      {stats.get('moves_resolved', 'N/A')} / 4")
    print(f"  moves_unresolved:    {stats.get('moves_unresolved', 'N/A')} / 4")
    print(f"  cap_hits:            {stats.get('cap_hits', 'N/A')}")
    print(f"  batches_eval:        {stats.get('batches_eval', 'N/A'):,}")
    print(f"  nodes_visited:       {stats.get('nodes_visited', 0):,}")
    lookups = stats.get("tt_lookups", 0)
    hits = stats.get("tt_hits", 0)
    hit_rate = (hits / lookups * 100) if lookups > 0 else 0.0
    print(f"  tt_size:             {stats.get('tt_size', 0):,}")
    print(f"  tt_hit_rate:         {hit_rate:6.2f}%")
    print(f"  tt_collisions:       {stats.get('tt_collisions', 0):,}")
    print(f"  tt_same_key_overwrites: {stats.get('tt_same_key_overwrites', 0):,}")
    print("=" * 56)

    if warning_count > 0:
        print(f"\n[FAIL] {warning_count} cap-hit warning(s) — search did not converge")
        sys.exit(1)
    print(f"\n[PASS] No cap hits — search converged cleanly on this board")
    sys.exit(0)


if __name__ == "__main__":
    main()
