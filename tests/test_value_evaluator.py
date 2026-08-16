from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from twenty_forty_eight_ai.env.d4_transforms import NUM_TRANSFORMS, apply_d4
from twenty_forty_eight_ai.evaluation.value_evaluator import D4ValueEvaluator
from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher


class _FakePolicy:
    """Deterministic, deliberately orientation-sensitive critic."""

    device = torch.device("cpu")

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.batches: list[np.ndarray] = []

    def eval(self):
        return self

    def predict_values(self, observations: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(observations.shape[0]))
        self.batches.append(observations.detach().cpu().numpy().copy())
        weights = torch.arange(
            1, observations.shape[-1] * observations.shape[-2] + 1,
            dtype=torch.float32,
        ).reshape(1, -1)
        flat = observations.float().flatten(start_dim=1)
        return (flat * weights).sum(dim=1, keepdim=True)


def _board() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 0, 2],
            [3, 0, 4, 0],
            [0, 5, 0, 0],
            [1, 0, 0, 2],
        ],
        dtype=np.int32,
    )


def test_eight_way_average_is_invariant_for_all_d4_inputs():
    policy = _FakePolicy()
    evaluator = D4ValueEvaluator(policy)
    board = _board()

    values = [
        evaluator([apply_d4(board, transform)])[0]
        for transform in range(NUM_TRANSFORMS)
    ]

    assert max(values) - min(values) <= 1e-6
    assert policy.batch_sizes == [NUM_TRANSFORMS] * NUM_TRANSFORMS


def test_evaluator_expands_inputs_in_one_batch_and_matches_individual_calls():
    boards = np.stack([_board(), np.rot90(_board(), 1)])
    batched_policy = _FakePolicy()
    batched = D4ValueEvaluator(batched_policy)(boards)

    individual_policy = _FakePolicy()
    individual_evaluator = D4ValueEvaluator(individual_policy)
    individual = [individual_evaluator([board])[0] for board in boards]

    np.testing.assert_allclose(batched, individual, rtol=0.0, atol=1e-6)
    assert batched_policy.batch_sizes == [2 * NUM_TRANSFORMS]
    assert individual_policy.batch_sizes == [NUM_TRANSFORMS, NUM_TRANSFORMS]


def test_evaluator_starts_from_received_orientation_without_recanonicalizing():
    policy = _FakePolicy()
    evaluator = D4ValueEvaluator(policy)
    board = _board()

    evaluator([board])

    first_transformed_input = policy.batches[0][0, 0]
    np.testing.assert_array_equal(first_transformed_input, board)


def test_empty_callback_does_not_run_neural_evaluation():
    policy = _FakePolicy()
    evaluator = D4ValueEvaluator(policy)

    assert evaluator([]) == []
    assert policy.batch_sizes == []
    assert evaluator.stats()["evaluator_callback_count"] == 0


def test_diagnostic_harness_c_matches_official_evaluator(monkeypatch):
    from scripts import diagnose_canonicalization_sensitivity as diagnostic

    policy = _FakePolicy()
    policy_model = SimpleNamespace(policy=policy, device=torch.device("cpu"))
    monkeypatch.setattr(
        diagnostic,
        "load_ppo_model",
        lambda *_args, **_kwargs: policy_model,
    )

    harness = diagnostic.CriticEvaluator(Path("unused.zip"), "cpu", 32768)
    official = D4ValueEvaluator(policy)
    board = _board()

    assert harness.average_value(board) == pytest.approx(
        official([board])[0], abs=1e-6
    )


def test_cpp_leaf_callback_receives_canonicalized_boards():
    from twenty_forty_eight_ai.utils.searcher import _load_searcher_impl

    encoder = _load_searcher_impl().BoardEncoder
    board = np.asarray(
        [
            [1, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    callback_batches: list[list[np.ndarray]] = []

    def callback(boards):
        callback_batches.append([np.asarray(item, dtype=np.int32) for item in boards])
        return [0.0] * len(boards)

    ExpectimaxSearcher(target_batch_size=8192).find_best_move(board, 1, callback)

    # The first callback is root post-move ordering. Subsequent callbacks are
    # leaf batches, whose C++ boards are unpacked from canonical TT keys.
    assert len(callback_batches) >= 2
    for batch in callback_batches[1:]:
        for leaf in batch:
            canonical = np.asarray(
                encoder.unpack(encoder.canonicalize_board(leaf.tolist())),
                dtype=np.int32,
            )
            np.testing.assert_array_equal(leaf, canonical)


def test_official_evaluator_preserves_search_determinism():
    board = _board()
    evaluator_one = D4ValueEvaluator(_FakePolicy())
    evaluator_two = D4ValueEvaluator(_FakePolicy())
    result_one = ExpectimaxSearcher(target_batch_size=8192).find_best_move(
        board, 2, evaluator_one
    )
    result_two = ExpectimaxSearcher(target_batch_size=8192).find_best_move(
        board, 2, evaluator_two
    )

    for key in result_one.keys() & result_two.keys():
        if key != "think_ms":
            assert result_one[key] == result_two[key], key


def test_benchmarker_and_visualizer_share_the_same_evaluator_class():
    from twenty_forty_eight_ai.evaluation import benchmarker
    from twenty_forty_eight_ai.utils import visualizer

    assert benchmarker.D4ValueEvaluator is D4ValueEvaluator
    assert visualizer.D4ValueEvaluator is D4ValueEvaluator
