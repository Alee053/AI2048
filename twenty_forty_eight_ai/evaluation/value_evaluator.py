"""Shared neural value evaluators used by production search callers."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch

from twenty_forty_eight_ai.env.d4_transforms import NUM_TRANSFORMS, apply_d4
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor


class D4ValueEvaluator:
    """Evaluate canonical-search boards with exact eight-way D4 averaging.

    The C++ searcher owns canonicalization and passes canonical leaf boards to
    this callback. Root post-move ordering boards may be non-canonical; the
    averaging operation is invariant for either input orientation, so this
    evaluator deliberately does not canonicalize again.
    """

    def __init__(self, policy, device=None):
        self.policy = policy
        self.device = device if device is not None else policy.device
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset callback and neural-evaluation counters."""
        self.callback_count = 0
        self.input_board_count = 0
        self.expanded_board_count = 0
        self.neural_time_s = 0.0
        self.total_time_s = 0.0

    def stats(self) -> dict[str, float | int]:
        """Return cumulative evaluator instrumentation for the caller."""
        return {
            "evaluator_callback_count": self.callback_count,
            "evaluator_input_board_count": self.input_board_count,
            "evaluator_expanded_board_count": self.expanded_board_count,
            "evaluator_neural_time_s": self.neural_time_s,
            "evaluator_total_time_s": self.total_time_s,
        }

    def __call__(self, boards_list: Iterable[np.ndarray]) -> list[float]:
        """Return one eight-way averaged value for every input board."""
        boards = np.asarray(list(boards_list), dtype=np.int32)
        if boards.size == 0:
            return []
        if boards.ndim != 3 or boards.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected boards with shape (N, 4, 4), got {boards.shape}"
            )

        started_at = time.perf_counter()
        expanded = np.stack(
            [
                apply_d4(board, transform)
                for board in boards
                for transform in range(NUM_TRANSFORMS)
            ]
        ).astype(np.int32, copy=False)
        tensor = board_to_tensor(expanded)

        neural_started_at = time.perf_counter()
        with torch.no_grad():
            values = self.policy.predict_values(
                torch.as_tensor(tensor, device=self.device)
            ).flatten()
        neural_elapsed = time.perf_counter() - neural_started_at

        expected_values = len(boards) * NUM_TRANSFORMS
        if values.numel() != expected_values:
            raise ValueError(
                "Value evaluator returned an invalid number of values: "
                f"expected {expected_values}, got {values.numel()}"
            )
        means = values.reshape(len(boards), NUM_TRANSFORMS).mean(dim=1)

        self.callback_count += 1
        self.input_board_count += len(boards)
        self.expanded_board_count += expected_values
        self.neural_time_s += neural_elapsed
        self.total_time_s += time.perf_counter() - started_at
        return means.detach().cpu().numpy().astype(np.float32).tolist()
