"""Verify the C++ ExpectimaxSearcher is deterministic across calls.

The benchmark harness relies on the C++ chance-node evaluation being
deterministic (no RandomUtil calls during search). If a future change
adds randomness here, per-worker seed wiring must be updated.

This test runs the same search twice on the same board and asserts the
counters are identical.
"""
from __future__ import annotations

import numpy as np

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher


def _fake_batch_eval(boards):
    return [0.0] * len(boards)


_BOARD = np.array([
    [0, 1, 0, 2],
    [1, 0, 2, 0],
    [0, 2, 0, 1],
    [2, 0, 1, 0],
], dtype=np.int32)


def test_searcher_chance_node_is_deterministic():
    s1 = ExpectimaxSearcher(target_batch_size=8192)
    s2 = ExpectimaxSearcher(target_batch_size=8192)
    r1 = s1.find_best_move(_BOARD, depth=2, batch_eval_func=_fake_batch_eval)
    r2 = s2.find_best_move(_BOARD, depth=2, batch_eval_func=_fake_batch_eval)
    # Every counter except think_ms (wall-clock-dependent) must match.
    timing_keys = {"think_ms"}
    for key in r1.keys() & r2.keys():
        if key in timing_keys:
            continue
        assert r1[key] == r2[key], f"{key}: {r1[key]} != {r2[key]}"