"""Capture board states that cause slow searches."""
import numpy as np
import signal
import sys

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher
from twenty_forty_eight_ai.env.game import Fast2048


def fake_batch_eval(boards):
    return [float(sum(sum(row) for row in b)) for b in boards]


def run_game(depth, max_moves=200):
    searcher = ExpectimaxSearcher()
    game = Fast2048()
    
    for move_num in range(max_moves):
        board = game.board.copy()
        
        def timeout_handler(signum, frame):
            print(f"\nSLOW BOARD at move {move_num}, depth {depth}:")
            print(board.tolist())
            print(f"TT size before: {searcher.find_best_move(board, depth, fake_batch_eval)['tt_size']}")
            raise TimeoutError(f"Search hung at move {move_num}, depth {depth}")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        try:
            stats = searcher.find_best_move(board, depth, fake_batch_eval)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        best_move = stats['best_move']
        _, terminated, moved = game.move(best_move)
        
        if terminated:
            return move_num + 1
    
    return max_moves


if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Testing depth {depth}...", flush=True)
    for game_num in range(100):
        try:
            moves = run_game(depth)
            print(f"Game {game_num+1}: {moves} moves", flush=True)
        except TimeoutError:
            print("Captured slow board, exiting.", flush=True)
            break
