"""Python wrapper around the C++ ExpectimaxSearcher.

Converts SearchStats C++ struct to a Python dict for compatibility with
Python callers (visualizer, benchmark, etc.).

The raw pybind11 module (searcher.cpython-312-x86_64-linux-gnu.so) is
explicitly loaded via importlib to avoid shadowing by this pure Python
module. This wrapper adds a dict conversion layer.
"""
from __future__ import annotations

import importlib.util
import sys


def _load_searcher_impl():
    """Explicitly load the C++ extension module from the .so file."""
    so_path = __file__.replace('searcher.py', '_searcher_cpp.cpython-312-x86_64-linux-gnu.so')
    spec = importlib.util.spec_from_file_location('searcher', so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load C++ searcher module from {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules['_searcher_impl'] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_searcher_impl()


class ExpectimaxSearcher:
    """Python wrapper that converts C++ SearchStats to a dict."""

    def __init__(self) -> None:
        self._impl = _impl.ExpectimaxSearcher()

    def find_best_move(
        self,
        board: "np.ndarray",
        depth: int,
        batch_eval_func: "callable",
    ) -> dict:
        """Find the best move for the given board.

        Args:
            board: 4x4 numpy array of tile values (0-16, log2-encoded).
            depth: Search depth for expectimax.
            batch_eval_func: Callback(list[np.ndarray]) -> list[float] for CNN evaluation.

        Returns:
            dict with keys: best_move, think_ms, nodes_visited, batches_eval,
            move_scores, tt_size, tt_lookups, tt_hits
        """
        stats = self._impl.find_best_move(board, depth, batch_eval_func)
        # Handle move_scores - pybind11 may return a float (0th element) instead of
        # the full array on some platforms, so we work around this.
        raw_scores = stats.move_scores
        if isinstance(raw_scores, float):
            # pybind11 quirk: C-style float[4] array returned as scalar
            move_scores = [float(raw_scores)] * 4
        else:
            move_scores = [float(raw_scores[i]) for i in range(4)]
        return {
            'best_move': int(stats.best_move),
            'think_ms': float(stats.think_ms),
            'nodes_visited': int(stats.nodes_visited),
            'batches_eval': int(stats.batches_eval),
            'move_scores': move_scores,
            'tt_size': int(stats.tt_size),
            'tt_lookups': int(stats.tt_lookups),
            'tt_hits': int(stats.tt_hits),
        }