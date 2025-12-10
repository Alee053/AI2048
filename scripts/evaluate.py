"""
Evaluate trained 2048 agent.

Modes:
1. Raw PPO
2. PPO + Expectimax (uses value function for tactical search)

Usage:
    python scripts/evaluate.py data/models/my_model.zip
    python scripts/evaluate.py data/models/my_model.zip --no-search
    python scripts/evaluate.py data/models/my_model.zip --depth 5
"""

import argparse
from twenty_forty_eight_ai.utils.visualizer import Visualizer


def main():
    """Run visualizer."""
    parser = argparse.ArgumentParser(description="Evaluate a trained 2048 AI agent.")

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

    args = parser.parse_args()

    try:
        # Instantiate the visualizer with the arguments from the command line
        vis = Visualizer(
            model_path=args.model_path,
            use_expectimax=args.use_expectimax,
            search_depth=args.depth
        )
        vis.run()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()