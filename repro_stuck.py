"""Reproduction script for stuck-at-30-moves bug."""
import numpy as np
import time
import signal
import sys

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher
from twenty_forty_eight_ai.env.game import Fast2048


def fake_batch_eval(boards):
    """Deterministic fake evaluator."""
    return [float(sum(sum(row) for row in b)) for b in boards]


def run_game(depth, max_moves=200):
    """Run a full game at given depth, returning move count or -1 if stuck."""
    searcher = ExpectimaxSearcher()
    game = Fast2048()
    
    for move_num in range(max_moves):
        board = game.board.copy()
        
        # Add timeout to detect infinite loops
        def timeout_handler(signum, frame):
            print(f"\nSLOW BOARD at move {move_num}, depth {depth}:")
            print(board.tolist())
            raise TimeoutError(f"Search hung at move {move_num}, depth {depth}")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout per move
        
        try:
            stats = searcher.find_best_move(board, depth, fake_batch_eval)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        best_move = stats['best_move']
        
        # Try to execute the move
        merge_score, terminated, moved = game.move(best_move)
        
        if not moved and not terminated:
            print(f"  Move {move_num}: best_move={best_move} but not moved! board=\n{game.board}", flush=True)
        
        if terminated:
            return move_num + 1
    
    return max_moves


def test_many_games(depth, num_games=20):
    """Run many games and report if any get stuck."""
    stuck = []
    for i in range(num_games):
        try:
            moves = run_game(depth)
            print(f"Game {i+1}: {moves} moves", flush=True)
        except TimeoutError as e:
            print(f"Game {i+1}: STUCK - {e}", flush=True)
            stuck.append(i+1)
    return stuck


if __name__ == "__main__":
    for depth in [2, 3]:
        print(f"\n=== Testing depth {depth} ===", flush=True)
        stuck = test_many_games(depth, num_games=20)
        if stuck:
            print(f"STUCK GAMES at depth {depth}: {stuck}", flush=True)
        else:
            print(f"No stuck games at depth {depth}", flush=True)
