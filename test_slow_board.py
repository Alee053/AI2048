"""Test a specific slow board."""
import numpy as np
import time

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher

def fake_batch_eval(boards):
    return [float(sum(sum(row) for row in b)) for b in boards]

# A known slow board from repro_stuck.py
board = np.array([
    [0, 1, 5, 1],
    [0, 1, 2, 6],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
], dtype=np.int32)

searcher = ExpectimaxSearcher()

print("Testing depth 2 on slow board...")
start = time.time()
stats = searcher.find_best_move(board, 2, fake_batch_eval)
elapsed = time.time() - start
print(f"Result: best_move={stats['best_move']}, time={elapsed:.3f}s")
print(f"nodes={stats['nodes_visited']}, batches={stats['batches_eval']}, tt_size={stats['tt_size']}")
print(f"move_scores={stats['move_scores']}")
