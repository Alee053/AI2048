"""
Headless benchmark for 2048 AI.

Usage:
    python scripts/benchmark.py <model_path> [OPTIONS]

Arguments:
    model_path (required): Path to trained model .zip file
    --n_runs <int>: Number of episodes to simulate (default: 10)
    --depth <int>: Expectimax search depth; 0 = raw policy (default: 0)
    --output <name>: Custom run name for output folder
    --device <str>: Device for model inference: cpu, cuda, auto (default: auto)

Examples:
    # Full 100-episode benchmark with depth-3 search
    python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --n_runs 100 --depth 3

    # Quick test with raw policy
    python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --n_runs 10 --depth 0
"""

import argparse
import numpy as np
import torch
import json
import sys
import os
import time
from typing import List, Dict, Any
from pathlib import Path

# tqdm is lightweight; auto-disable if not a TTY (CI, logs, pipes)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


def _tqdm_iter(iterable, **kwargs):
    """Lightweight tqdm wrapper that disables itself in non-TTY environments."""
    if tqdm is None:
        return iterable
    disable = not sys.stdout.isatty()
    return tqdm(iterable, disable=disable, **kwargs)

# Attempt to import Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting functionality will be disabled.")

from sb3_contrib import MaskablePPO
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor

# Module-level helper to convert numpy scalars to Python types
def to_py(val):
    return val.item() if isinstance(val, np.generic) else val

# Attempt to import the ExpectimaxSearcher extension
try:
    from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher
except ImportError:
    ExpectimaxSearcher = None
    print("Warning: ExpectimaxSearcher C++ extension not found. Search functionality will be unavailable.")

class Benchmarker:
    def __init__(self, model_path: str, use_expectimax: bool, search_depth: int, device: str = "auto"):
        self.use_expectimax = use_expectimax
        self.search_depth = search_depth
        
        # Load Model
        print(f"Loading model from: {model_path}")
        self.model = MaskablePPO.load(model_path, device=device)
        self.device = self.model.device
        
        # Init Env
        self.env = Game2048Env()
        
        # Init Searcher
        self.searcher = None
        if self.use_expectimax:
            if ExpectimaxSearcher is None:
                raise ImportError("ExpectimaxSearcher is required for search but could not be imported.")
            self.searcher = ExpectimaxSearcher()
            print(f"Initialized Expectimax Searcher with depth {search_depth}")
        else:
            print("Using Raw PPO Policy (No Search)")

    def _evaluate_batch(self, boards_list: List[np.ndarray]) -> List[float]:
        """Critic evaluation callback."""
        if not boards_list:
            return []
        
        # Convert list of boards (N, 4, 4) to tensor (N, 1, 4, 4)
        batch_array = np.array(boards_list)
        batch_tensor = board_to_tensor(batch_array)
        
        with torch.no_grad():
            values = self.model.policy.predict_values(
                torch.as_tensor(batch_tensor).to(self.device)
            )
            
        return values.cpu().numpy().flatten().tolist()

    def run_episode(self, verbose: bool = False) -> Dict[str, Any]:
        """Run single episode."""
        obs, _ = self.env.reset()
        done = False
        steps = 0

        # Accumulate search stats across all moves in the episode
        episode_stats = {
            'think_ms': 0.0,
            'nodes_visited': 0,
            'batches_eval': 0,
            'tt_size': 0,
            'tt_hits': 0,
            'tt_lookups': 0,
            'tt_collisions': 0,
            'tt_same_key_overwrites': 0,
            'moves_resolved': 0,
            'moves_unresolved': 0,
            'cap_hits': 0,
            'move_scores': [],
        }
        while not done:
            if self.searcher:
                stats = self.searcher.find_best_move(
                    self.env.game.board,
                    self.search_depth,
                    self._evaluate_batch
                )
                action = int(stats['best_move'])
                episode_stats['think_ms'] += stats.get('think_ms', 0)
                episode_stats['nodes_visited'] += stats.get('nodes_visited', 0)
                episode_stats['batches_eval'] += stats.get('batches_eval', 0)
                episode_stats['tt_hits'] += stats.get('tt_hits', 0)
                episode_stats['tt_lookups'] += stats.get('tt_lookups', 0)
                episode_stats['tt_collisions'] += stats.get('tt_collisions', 0)
                episode_stats['tt_same_key_overwrites'] += stats.get('tt_same_key_overwrites', 0)
                episode_stats['moves_resolved'] += stats.get('moves_resolved', 0)
                episode_stats['moves_unresolved'] += stats.get('moves_unresolved', 0)
                episode_stats['cap_hits'] += stats.get('cap_hits', 0)
                # Keep the latest move_scores and tt_size for reference
                episode_stats['move_scores'] = list(stats.get('move_scores', []))
                episode_stats['tt_size'] = stats.get('tt_size', 0)
            else:
                action_mask = self.env.action_masks()
                action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
                action = int(action)

            obs, _, done, _, info = self.env.step(action)
            steps += 1

        result = {
            "score": int(self.env.game.score),
            "max_tile": int(2 ** self.env.game.max_tile),
            "steps": steps,
            "search_stats": episode_stats,
        }
        if verbose:
            print(f"Episode N │ Score: {result['score']:,} │ Max tile: {result['max_tile']:,} │ Steps: {result['steps']}")
        return result

    def _create_plot(self, scores: List[int], avg_score: float, output_path: str, config: Dict):
        """Generate score histogram."""
        if not MATPLOTLIB_AVAILABLE:
            return

        plt.figure(figsize=(10, 6))
        
        # Determine bins dynamically
        bins = min(20, max(5, len(set(scores))))
        
        plt.hist(scores, bins=bins, edgecolor='black', alpha=0.7, color='#776e65')
        plt.axvline(avg_score, color='#edc22e', linestyle='--', linewidth=2,
                    label=f"Mean: {avg_score:.0f}")
        
        title_mode = f"Search Depth {config['search_depth']}" if config['use_expectimax'] else "Raw PPO Policy"
        plt.title(f'Score Distribution - {title_mode} ({config["n_runs"]} games)')
        plt.xlabel('Final Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_path}")

    def benchmark(self, n_runs: int, run_name: str):
        """Run benchmark loop."""
        stats: List[Dict[str, Any]] = []
        print(f"\nStarting benchmark for {n_runs} runs...")

        # Quick warm-up to estimate per-episode time for a human-readable ETA
        if n_runs > 1:
            t0 = time.perf_counter()
            _ = self.run_episode()
            t1 = time.perf_counter()
            warmup_s = t1 - t0
            est_total_s = warmup_s * n_runs
            print(f"  Warm-up episode: {warmup_s:.2f}s — estimated total: {est_total_s/60:.1f}m")
            # Don't count warm-up in stats
        else:
            warmup_s = None

        iterator = _tqdm_iter(range(n_runs), desc="Benchmarking", unit="game")
        start_time = time.perf_counter()

        try:
            for i, _ in enumerate(iterator):
                result = self.run_episode()
                stats.append(result)
                elapsed = time.perf_counter() - start_time
                throughput = (i + 1) / elapsed if elapsed > 0 else 0.0
                avg_score = sum(s['score'] for s in stats) / len(stats)
                postfix = {
                    "Score": result['score'],
                    "MaxTile": result['max_tile'],
                    "Avg": f"{avg_score:.0f}",
                    "GPS": f"{throughput:.2f}",
                }
                if hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(postfix)
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user. Saving collected data...")
        
        if not stats:
            print("No runs completed.")
            return

        # Data integrity sanity check
        if len(stats) != n_runs:
            print(f"WARNING: Expected {n_runs} runs, but only collected {len(stats)}. Data may be incomplete.")

        # Aggregate stats
        scores = [s['score'] for s in stats]
        max_tiles = [s['max_tile'] for s in stats]
        steps = [s['steps'] for s in stats]
        
        total_elapsed = time.perf_counter() - start_time

        # Aggregate search metrics (only if using expectimax)
        search_metrics = {}
        if self.use_expectimax:
            think_times = [s['search_stats']['think_ms'] for s in stats if s.get('search_stats')]
            nodes_vis = [s['search_stats']['nodes_visited'] for s in stats if s.get('search_stats')]
            batches = [s['search_stats']['batches_eval'] for s in stats if s.get('search_stats')]
            tt_lookups = [s['search_stats']['tt_lookups'] for s in stats if s.get('search_stats')]
            tt_hits = [s['search_stats']['tt_hits'] for s in stats if s.get('search_stats')]
            tt_collisions = [s['search_stats'].get('tt_collisions', 0) for s in stats if s.get('search_stats')]
            tt_same_key_overwrites = [s['search_stats'].get('tt_same_key_overwrites', 0) for s in stats if s.get('search_stats')]
            moves_resolved = [s['search_stats'].get('moves_resolved', 0) for s in stats if s.get('search_stats')]
            moves_unresolved = [s['search_stats'].get('moves_unresolved', 0) for s in stats if s.get('search_stats')]
            cap_hits = [s['search_stats'].get('cap_hits', 0) for s in stats if s.get('search_stats')]
            alpha_beta_cuts = [s['search_stats'].get('alpha_beta_cuts', 0) for s in stats if s.get('search_stats')]
            chance_nodes = [s['search_stats'].get('chance_nodes_evaluated', 0) for s in stats if s.get('search_stats')]
            max_nodes = [s['search_stats'].get('max_nodes_evaluated', 0) for s in stats if s.get('search_stats')]
            chance_value_sums = [s['search_stats'].get('chance_value_sum', 0.0) for s in stats if s.get('search_stats')]
            chance_value_counts = [s['search_stats'].get('chance_value_count', 0) for s in stats if s.get('search_stats')]

            if think_times:
                avg_think = np.mean(think_times)
                avg_nodes = np.mean(nodes_vis)
                # Aggregate chance value: sum the per-move sums/counts to get the
                # true mean across all chance-node evaluations in all games.
                total_cv_sum = float(np.sum(chance_value_sums))
                total_cv_count = int(np.sum(chance_value_counts))
                avg_chance_value = (total_cv_sum / total_cv_count) if total_cv_count > 0 else 0.0
                search_metrics = {
                    "avg_think_ms": round(float(avg_think), 3),
                    "avg_nodes_visited": round(float(avg_nodes), 1),
                    "avg_batches_eval": round(float(np.mean(batches)), 1) if batches else 0.0,
                    "avg_nodes_per_sec": round(float(avg_nodes / (avg_think / 1000)), 1) if avg_think > 0 else 0.0,
                    "avg_tt_hit_rate": round(float(np.mean([h / l * 100 if l > 0 else 0 for h, l in zip(tt_hits, tt_lookups)])), 2) if tt_lookups else 0.0,
                    "avg_tt_collisions": round(float(np.mean(tt_collisions)), 1) if tt_collisions else 0.0,
                    "avg_tt_same_key_overwrites": round(float(np.mean(tt_same_key_overwrites)), 1) if tt_same_key_overwrites else 0.0,
                    "avg_moves_resolved": round(float(np.mean(moves_resolved)), 2) if moves_resolved else 0.0,
                    "avg_moves_unresolved": round(float(np.mean(moves_unresolved)), 2) if moves_unresolved else 0.0,
                    "avg_cap_hits": round(float(np.mean(cap_hits)), 1) if cap_hits else 0.0,
                    # Diagnostic fields (added on fix-chance-divisor branch)
                    "avg_alpha_beta_cuts": round(float(np.mean(alpha_beta_cuts)), 1) if alpha_beta_cuts else 0.0,
                    "avg_chance_nodes": round(float(np.mean(chance_nodes)), 1) if chance_nodes else 0.0,
                    "avg_max_nodes": round(float(np.mean(max_nodes)), 1) if max_nodes else 0.0,
                    "avg_chance_value": round(avg_chance_value, 4),
                }

        summary = {
            "config": {
                "use_expectimax": self.use_expectimax,
                "search_depth": self.search_depth,
                "n_runs": len(stats)
            },
            "metrics": {
                "avg_score": to_py(np.mean(scores)),
                "std_score": to_py(np.std(scores)),
                "min_score": to_py(np.min(scores)),
                "max_score": to_py(np.max(scores)),
                "avg_steps": to_py(np.mean(steps)),
                "total_time_s": round(total_elapsed, 3),
                "games_per_sec": round(len(stats) / total_elapsed, 3) if total_elapsed > 0 else 0.0,
                **search_metrics,
            },
            "max_tile_dist": {str(t): max_tiles.count(t) for t in sorted(set(max_tiles))},
            "raw_data": stats
        }
        
        # Output setup
        base_dir = Path("data/benchmarks")
        if not run_name:
            run_name = f"run_{int(time.time())}"
        
        output_dir = base_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / "results.json"
        plot_path = output_dir / "score_distribution.png"

        # Report
        print("\n" + "="*40)
        print("           BENCHMARK RESULTS            ")
        print("="*40)
        print(f"Runs Completed: {len(stats)}")
        print(f"Total Time:     {total_elapsed:.1f}s ({summary['metrics']['games_per_sec']:.2f} games/s)")
        print(f"Average Score:  {summary['metrics']['avg_score']:.2f} +/- {summary['metrics']['std_score']:.2f}")
        print(f"Average Steps:  {summary['metrics']['avg_steps']:.2f}")
        print("-" * 40)
        print("Max Tile Distribution:")
        for tile, count in summary['max_tile_dist'].items():
            percentage = (count / len(stats)) * 100
            print(f"  {tile}: {count} ({percentage:.1f}%)")
        if self.use_expectimax and search_metrics:
            print("-" * 40)
            print("Search Diagnostics (per move, averaged):")
            print(f"  think_ms:            {search_metrics.get('avg_think_ms', 0):.2f}")
            print(f"  nodes_visited:       {search_metrics.get('avg_nodes_visited', 0):.1f}")
            print(f"  batches_eval:        {search_metrics.get('avg_batches_eval', 0):.1f}")
            print(f"  nodes_per_sec:       {search_metrics.get('avg_nodes_per_sec', 0):.1f}")
            print(f"  tt_hit_rate (%):     {search_metrics.get('avg_tt_hit_rate', 0):.2f}")
            print(f"  tt_collisions:       {search_metrics.get('avg_tt_collisions', 0):.1f}")
            print(f"  tt_same_key_overwrites: {search_metrics.get('avg_tt_same_key_overwrites', 0):.1f}")
            print(f"  moves_resolved:      {search_metrics.get('avg_moves_resolved', 0):.2f} / 4")
            print(f"  moves_unresolved:    {search_metrics.get('avg_moves_unresolved', 0):.2f} / 4")
            print(f"  cap_hits:            {search_metrics.get('avg_cap_hits', 0):.1f}")
            print(f"  alpha_beta_cuts:     {search_metrics.get('avg_alpha_beta_cuts', 0):.1f}  (unsound in expectimax; counts max-node beta cuts)")
            print(f"  chance_nodes:        {search_metrics.get('avg_chance_nodes', 0):.1f}")
            print(f"  max_nodes:           {search_metrics.get('avg_max_nodes', 0):.1f}")
            print(f"  avg_chance_value:    {search_metrics.get('avg_chance_value', 0):.4f}  (sanity check: 2x here => buggy build)")
        print("="*40)
        
        # Save JSON
        try:
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"\nDetailed results saved to: {json_path}")
        except Exception as e:
            print(f"Error saving JSON results: {e}")
            
        # Save Plot
        if MATPLOTLIB_AVAILABLE:
            try:
                self._create_plot(scores, summary['metrics']['avg_score'], str(plot_path), summary['config'])
            except Exception as e:
                print(f"Error creating plot: {e}")


def benchmark_multi_seed(model_dir: str, n_runs: int, search_depth: int, device: str, output: str, verbose: bool, parallel: bool):
    """Run benchmark across all seed_N/ subdirs in model_dir."""
    import re

    # Discover seed subdirs
    seed_dirs = []
    for entry in os.scandir(model_dir):
        if entry.is_dir() and re.match(r"^seed_\d+$", entry.name):
            seed_dirs.append(entry.path)

    seed_dirs.sort(key=lambda p: int(re.search(r"seed_(\d+)", p).group(1)))

    if not seed_dirs:
        print(f"Error: no seed_N/ directories found in {model_dir}")
        sys.exit(1)

    print(f"Found {len(seed_dirs)} seed directories: {[os.path.basename(d) for d in seed_dirs]}")

    # Output setup
    base_dir = Path("data/benchmarks")
    output_dir = base_dir / output
    output_dir.mkdir(parents=True, exist_ok=True)

    if parallel:
        # Parallel execution (background jobs)
        import subprocess
        procs = []
        for seed_path in seed_dirs:
            seed_match = re.search(r"seed_(\d+)", seed_path)
            seed_n = seed_match.group(1)
            model_file = os.path.join(seed_path, "final_model.zip")

            if not os.path.exists(model_file):
                print(f"Warning: {model_file} not found, skipping seed {seed_n}")
                continue

            cmd = [
                sys.executable, __file__,
                model_file,
                "--n_runs", str(n_runs),
                "--depth", str(search_depth),
                "--output", output,
                "--device", device,
                "--verbose" if verbose else "",
            ]
            cmd = [c for c in cmd if c]
            print(f"Launching seed {seed_n} in background...")
            procs.append(subprocess.Popen(cmd, cwd=os.getcwd()))
        # Wait for all and check return codes
        for p in procs:
            ret = p.wait()
            if ret != 0:
                print(f"Warning: subprocess exited with code {ret}")
        print("All parallel benchmark jobs complete.")
    else:
        # Sequential execution
        all_results = []
        for seed_path in seed_dirs:
            seed_match = re.search(r"seed_(\d+)", seed_path)
            seed_n = seed_match.group(1)
            model_file = os.path.join(seed_path, "final_model.zip")
            out_file = output_dir / f"results_seed_{seed_n}.json"

            if not os.path.exists(model_file):
                print(f"Warning: {model_file} not found, skipping seed {seed_n}")
                continue

            print(f"\n=== Benchmarking seed {seed_n} ({os.path.basename(seed_path)}) ===")
            bencher = Benchmarker(model_file, search_depth > 0, search_depth, device)

            stats: List[Dict[str, Any]] = []
            iterator = _tqdm_iter(range(n_runs), desc=f"Seed {seed_n}", unit="game")
            seed_start = time.perf_counter()
            for i, _ in enumerate(iterator):
                result = bencher.run_episode(verbose=verbose)
                stats.append(result)
                elapsed = time.perf_counter() - seed_start
                throughput = (i + 1) / elapsed if elapsed > 0 else 0.0
                avg_score = sum(s['score'] for s in stats) / len(stats)
                postfix = {
                    "Score": result['score'],
                    "MaxTile": result['max_tile'],
                    "Avg": f"{avg_score:.0f}",
                    "GPS": f"{throughput:.2f}",
                }
                if hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(postfix)

            # Save per-seed results
            scores = [s['score'] for s in stats]
            max_tiles = [s['max_tile'] for s in stats]
            steps = [s['steps'] for s in stats]
            seed_elapsed = time.perf_counter() - seed_start

            # Aggregate search metrics for this seed
            seed_search_metrics = {}
            if search_depth > 0:
                think_times = [s['search_stats']['think_ms'] for s in stats if s.get('search_stats')]
                nodes_vis = [s['search_stats']['nodes_visited'] for s in stats if s.get('search_stats')]
                batches = [s['search_stats']['batches_eval'] for s in stats if s.get('search_stats')]
                tt_lookups = [s['search_stats']['tt_lookups'] for s in stats if s.get('search_stats')]
                tt_hits = [s['search_stats']['tt_hits'] for s in stats if s.get('search_stats')]
                
                if think_times:
                    avg_think = np.mean(think_times)
                    avg_nodes = np.mean(nodes_vis)
                    seed_search_metrics = {
                        "avg_think_ms": round(float(avg_think), 3),
                        "avg_nodes_visited": round(float(avg_nodes), 1),
                        "avg_batches_eval": round(float(np.mean(batches)), 1) if batches else 0.0,
                        "avg_nodes_per_sec": round(float(avg_nodes / (avg_think / 1000)), 1) if avg_think > 0 else 0.0,
                        "avg_tt_hit_rate": round(float(np.mean([h / l * 100 if l > 0 else 0 for h, l in zip(tt_hits, tt_lookups)])), 2) if tt_lookups else 0.0,
                    }

            seed_summary = {
                "config": {"use_expectimax": search_depth > 0, "search_depth": search_depth, "n_runs": len(stats)},
                "metrics": {
                    "avg_score": to_py(np.mean(scores)), "std_score": to_py(np.std(scores)),
                    "min_score": to_py(np.min(scores)), "max_score": to_py(np.max(scores)),
                    "avg_steps": to_py(np.mean(steps)),
                    "total_time_s": round(seed_elapsed, 3),
                    "games_per_sec": round(len(stats) / seed_elapsed, 3) if seed_elapsed > 0 else 0.0,
                    **seed_search_metrics,
                },
                "max_tile_dist": {str(t): max_tiles.count(t) for t in sorted(set(max_tiles))},
                "raw_data": stats
            }

            with open(out_file, "w") as f:
                json.dump(seed_summary, f, indent=4)
            print(f"Seed {seed_n} results saved to {out_file}")
            all_results.append((seed_n, seed_summary))

        print(f"\n=== All seeds benchmarked: {len(all_results)}/{len(seed_dirs)} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_path", type=str, help="Path to the trained model .zip file")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of games to simulate (default: 10)")
    parser.add_argument("--depth", type=int, default=0, help="Expectimax search depth. 0 = Raw Policy (default: 0)")
    parser.add_argument("--output", type=str, default=None, help="Name of the run (folder name). Defaults to 'run_<timestamp>'")
    parser.add_argument("--device", type=str, default="auto", help="Device to run model on (cpu, cuda, auto)")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode progress line.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing seed_N/ subdirectories for multi-seed benchmarking.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run seed benchmarks in parallel (valid for benchmark.py).")
    args = parser.parse_args()

    # Validate output naming convention (required for aggregate.py)
    import re
    output_name = args.output or f"run_{int(time.time())}"
    if args.model_dir and not re.match(r"^[\w-]+_depth\d+$", output_name):
        print(f"Error: --output must follow pattern {{sweep_name}}_depth{{N}} when using --model-dir.")
        print(f"  Got: '{output_name}' — expected something like 'sweep-v1_depth3'")
        sys.exit(1)

    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            print(f"Error: --model-dir path is not a directory: {args.model_dir}")
            sys.exit(1)
        benchmark_multi_seed(args.model_dir, args.n_runs, args.depth, args.device,
                             output_name, args.verbose, args.parallel)
    else:
        # Original single-model behavior
        use_search = args.depth > 0

        # Check for file existence
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            sys.exit(1)

        bencher = Benchmarker(args.model_path, use_search, args.depth, args.device)
        bencher.benchmark(args.n_runs, args.output)
