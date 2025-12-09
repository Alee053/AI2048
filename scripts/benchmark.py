import argparse
import numpy as np
import torch
import json
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor

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
        """Callback for C++ searcher to evaluate a batch of boards using the PPO critic."""
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

    def run_episode(self) -> Dict[str, Any]:
        """Runs a single episode of the game."""
        obs, _ = self.env.reset()
        done = False
        steps = 0
        
        while not done:
            if self.searcher:
                # Use C++ Expectimax with Model Value Function
                action = self.searcher.find_best_move(
                    self.env.game.board, 
                    self.search_depth, 
                    self._evaluate_batch
                )
            else:
                # Use Raw Policy
                action_mask = self.env.action_masks()
                action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
                action = int(action)
            
            obs, _, done, _, info = self.env.step(action)
            steps += 1
            
        return {
            "score": int(self.env.game.score),
            "max_tile": int(2 ** self.env.game.max_tile),
            "steps": steps
        }

    def benchmark(self, n_runs: int, output_file: str):
        """Runs the benchmark loop."""
        stats = []
        print(f"\nStarting benchmark for {n_runs} runs...")
        
        iterator = tqdm(range(n_runs), desc="Benchmarking", unit="game")
        
        try:
            for _ in iterator:
                result = self.run_episode()
                stats.append(result)
                # Update progress bar with latest result
                iterator.set_postfix({"Score": result['score'], "MaxTile": result['max_tile']})
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user. Saving collected data...")
        
        if not stats:
            print("No runs completed.")
            return

        # --- Aggregation ---
        scores = [s['score'] for s in stats]
        max_tiles = [s['max_tile'] for s in stats]
        steps = [s['steps'] for s in stats]
        
        # Helper to safely handle numpy types for JSON serialization
        def to_py(val): return val.item() if isinstance(val, np.generic) else val

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
            },
            "max_tile_dist": {str(t): max_tiles.count(t) for t in sorted(set(max_tiles))},
            "raw_data": stats
        }
        
        # --- Reporting ---
        print("\n" + "="*30)
        print("       BENCHMARK RESULTS       ")
        print("="*30)
        print(f"Runs Completed: {len(stats)}")
        print(f"Average Score:  {summary['metrics']['avg_score']:.2f} +/- {summary['metrics']['std_score']:.2f}")
        print(f"Average Steps:  {summary['metrics']['avg_steps']:.2f}")
        print("-" * 30)
        print("Max Tile Distribution:")
        for tile, count in summary['max_tile_dist'].items():
            percentage = (count / len(stats)) * 100
            print(f"  {tile}: {count} ({percentage:.1f}%)")
        print("="*30)
        
        # --- Save to File ---
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(summary, f, indent=4)
                print(f"\nDetailed results saved to: {output_file}")
            except Exception as e:
                print(f"Error saving results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Benchmark for 2048 AI Agent.")
    parser.add_argument("model_path", type=str, help="Path to the trained model .zip file")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of games to simulate (default: 10)")
    parser.add_argument("--depth", type=int, default=0, help="Expectimax search depth. 0 = Raw Policy (default: 0)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to save JSON results (default: benchmark_results.json)")
    parser.add_argument("--device", type=str, default="auto", help="Device to run model on (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    use_search = args.depth > 0
    
    # Check for file existence
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    bencher = Benchmarker(args.model_path, use_search, args.depth, args.device)
    bencher.benchmark(args.n_runs, args.output)
