"""
Evaluate trained 2048 agent with visualization.

Usage:
    python scripts/evaluate.py <model_path> [OPTIONS]

Arguments:
    model_path (required): Path to trained model .zip file
    --no-search: Disable Expectimax search (use raw PPO policy)
    --depth <int>: Search depth for Expectimax (default: 3)

Examples:
    # Depth-3 Expectimax (recommended)
    python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v3.zip --depth 3

    # Raw policy (no search)
    python scripts/evaluate.py data/models/release/Hybrid-PPO-Expectimax-v3.zip --no-search
"""

import argparse
from twenty_forty_eight_ai.utils.visualizer import Visualizer


def main():
    """Run visualizer."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Arguments
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved model .zip file."
    )
    parser.add_argument(
        "--no-search",
        action="store_false",
        dest="use_expectimax",
        help="Disable the Expectimax search and use the raw PPO policy instead."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="The search depth for the Expectimax algorithm (default: 3)."
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable enhanced stats panel (faster rendering)."
    )

    args = parser.parse_args()

    try:
        vis = Visualizer(
            model_path=args.model_path,
            use_expectimax=args.use_expectimax,
            search_depth=args.depth,
            show_stats=not args.no_stats
        )
        vis.run()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()