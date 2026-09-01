"""Measure action-level sensitivity to Expectimax leaf canonicalization.

This is intentionally a diagnostic harness, not a production search path. It
keeps raw leaf orientations available so the existing canonicalized evaluator
can be compared with raw and exact eight-way averaged value predictions
without changing the C++ searcher or training code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

# Keep direct ``python scripts/diagnostics/<file>.py`` execution compatible with models
# whose cloudpickle metadata references the repository's ``scripts`` package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from twenty_forty_eight_ai.agent.ppo import load_ppo_model
from twenty_forty_eight_ai.evaluation.value_evaluator import D4ValueEvaluator
from twenty_forty_eight_ai.env.d4_transforms import NUM_TRANSFORMS, apply_d4
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.env.game import Fast2048
from twenty_forty_eight_ai.utils.searcher import (
    ExpectimaxSearcher,
    _load_searcher_impl,
)
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor


ACTION_COUNT = 4
DEFAULT_OUTPUT = Path("data/diagnostics/canonicalization_sensitivity.json")
DEFAULT_ROOT_SEED = 7000
DEFAULT_EPISODES = 8
DEFAULT_MAX_ROOTS = 512
DEFAULT_EVAL_EPISODES = 8
DEFAULT_BATCH_SIZE = 32768
ACTION_TIE_TOLERANCE = 1e-4

_CPP = _load_searcher_impl()


def _as_board(board: np.ndarray | Iterable[Iterable[int]]) -> np.ndarray:
    result = np.asarray(board, dtype=np.int32)
    if result.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 board, got {result.shape}")
    return np.ascontiguousarray(result)


def _board_key(board: np.ndarray) -> bytes:
    return _as_board(board).tobytes()


def _pack(board: np.ndarray) -> int:
    return int(_CPP.BoardEncoder.pack(_as_board(board).tolist()))


def canonical_board(board: np.ndarray) -> np.ndarray:
    packed = _CPP.BoardEncoder.canonicalize_board(_as_board(board).tolist())
    return _as_board(_CPP.BoardEncoder.unpack(int(packed)))


def symmetry_boards(board: np.ndarray) -> np.ndarray:
    board = _as_board(board)
    return np.stack(
        [apply_d4(board, transform) for transform in range(NUM_TRANSFORMS)]
    ).astype(np.int32, copy=False)


def _f32(value: float) -> float:
    return float(np.float32(value))


def _reward(merge_score: int) -> float:
    if merge_score <= 0:
        return 0.0
    return _f32(np.log2(np.float32(merge_score)))


def _merge_line(line: Iterable[int]) -> tuple[np.ndarray, int]:
    tiles = [int(tile) for tile in line if int(tile) != 0]
    merged: list[int] = []
    merge_score = 0
    index = 0
    while index < len(tiles):
        if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
            exponent = tiles[index] + 1
            merged.append(exponent)
            merge_score += 1 << exponent
            index += 2
        else:
            merged.append(tiles[index])
            index += 1
    return np.asarray(merged + [0] * (4 - len(merged)), dtype=np.int32), merge_score


def _simulate_without_spawn(
    board: np.ndarray, action: int
) -> tuple[np.ndarray, int, bool]:
    result = _as_board(board).copy()
    merge_score = 0
    if action in (1, 3):
        for row in range(4):
            line = result[row]
            if action == 1:
                line = line[::-1]
            moved_line, score = _merge_line(line)
            if action == 1:
                moved_line = moved_line[::-1]
            result[row] = moved_line
            merge_score += score
    else:
        for column in range(4):
            line = result[:, column]
            if action == 2:
                line = line[::-1]
            moved_line, score = _merge_line(line)
            if action == 2:
                moved_line = moved_line[::-1]
            result[:, column] = moved_line
            merge_score += score
    return result, merge_score, not np.array_equal(result, board)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


class CriticEvaluator:
    """Batched, cached critic values for one already-trained model."""

    def __init__(self, model_path: Path, device: str, batch_size: int):
        self.model_path = str(model_path)
        self.model = load_ppo_model(model_path, device=device)
        self.model.policy.eval()
        self.device = self.model.device
        self.batch_size = batch_size
        self._raw_cache: dict[bytes, float] = {}
        self.official_evaluator = D4ValueEvaluator(
            self.model.policy, self.device
        )

    def predict_many(self, boards: Iterable[np.ndarray]) -> list[float]:
        board_list = [_as_board(board) for board in boards]
        if not board_list:
            return []

        values: list[float | None] = [None] * len(board_list)
        missing: dict[bytes, tuple[list[int], np.ndarray]] = {}
        for index, board in enumerate(board_list):
            key = _board_key(board)
            cached = self._raw_cache.get(key)
            if cached is not None:
                values[index] = cached
            elif key in missing:
                missing[key][0].append(index)
            else:
                missing[key] = ([index], board)

        missing_items = list(missing.values())
        for start in range(0, len(missing_items), self.batch_size):
            chunk = missing_items[start : start + self.batch_size]
            batch = board_to_tensor(np.stack([item[1] for item in chunk]))
            with torch.no_grad():
                predictions = self.model.policy.predict_values(
                    torch.as_tensor(batch, device=self.device)
                ).flatten()
            prediction_list = predictions.detach().cpu().numpy().astype(np.float32)
            for (indices, board), prediction in zip(chunk, prediction_list):
                value = float(prediction)
                key = _board_key(board)
                self._raw_cache[key] = value
                for index in indices:
                    values[index] = value

        if any(value is None for value in values):
            raise RuntimeError("Critic cache did not resolve every requested board")
        return [float(value) for value in values]

    def raw_value(self, board: np.ndarray) -> float:
        return self.predict_many([board])[0]

    def canonical_value(self, board: np.ndarray) -> float:
        return self.raw_value(canonical_board(board))

    def average_value(self, board: np.ndarray) -> float:
        return self.official_evaluator([board])[0]

    def mode_value(self, board: np.ndarray, mode: str) -> float:
        if mode == "raw":
            return self.raw_value(board)
        if mode == "canonical":
            return self.canonical_value(board)
        if mode == "average":
            return self.average_value(board)
        raise ValueError(f"Unknown evaluator mode: {mode}")

    def symmetry_metrics(self, board: np.ndarray) -> dict[str, float | list[float]]:
        values = self.predict_many(symmetry_boards(board))
        raw = values[0]
        canonical = self.canonical_value(board)
        average = _f32(np.mean(np.asarray(values, dtype=np.float32), dtype=np.float32))
        return {
            "values": values,
            "spread": float(max(values) - min(values)),
            "raw_vs_canonical": abs(raw - canonical),
            "canonical_vs_average": abs(canonical - average),
            "raw_vs_average": abs(raw - average),
        }

    def current_a_callback(self, boards: list[np.ndarray]) -> list[float]:
        """Callback for C++ search; C++ has already canonicalized these boards."""
        return self.predict_many(boards)


class MoveSimulator:
    """Reuse the repository's Python LUT game implementation for diagnostics."""

    def __init__(self):
        self.game = Fast2048(seed=0)

    def move(self, board: np.ndarray, action: int) -> tuple[np.ndarray, int] | None:
        board = _as_board(board)
        self.game.board = board.copy()
        self.game.score = 0
        self.game.done = False
        self.game.max_tile = int(board.max())
        if not self.game.is_move_valid(action):
            return None
        child, merge_score, moved = _simulate_without_spawn(board, action)
        if not moved:
            return None
        return child, int(merge_score)


@dataclass(frozen=True)
class Leaf:
    board: np.ndarray
    probability: float


@dataclass(frozen=True)
class RootAction:
    action: int
    immediate_reward: float
    leaves: tuple[Leaf, ...]
    empty_cell_count: int


def expand_depth_one(board: np.ndarray, simulator: MoveSimulator) -> tuple[RootAction, ...]:
    actions: list[RootAction] = []
    for action in range(ACTION_COUNT):
        moved = simulator.move(board, action)
        if moved is None:
            continue
        post_board, merge_score = moved
        empty_cells = np.argwhere(post_board == 0)
        if len(empty_cells) == 0:
            leaves = (Leaf(post_board, 1.0),)
            empty_cell_count = 1
        else:
            leaf_list: list[Leaf] = []
            for row, col in empty_cells:
                for exponent, tile_probability in ((1, 0.9), (2, 0.1)):
                    child = post_board.copy()
                    child[row, col] = exponent
                    leaf_list.append(
                        Leaf(child, tile_probability)
                    )
            leaves = tuple(leaf_list)
            empty_cell_count = len(empty_cells)
        actions.append(
            RootAction(action, _reward(merge_score), leaves, empty_cell_count)
        )
    return tuple(actions)


def score_expansion(
    expansion: tuple[RootAction, ...],
    evaluator: CriticEvaluator,
    mode: str,
) -> list[float]:
    scores = [float("-inf")] * ACTION_COUNT
    for root_action in expansion:
        chance_total = np.float32(0.0)
        for leaf in root_action.leaves:
            chance_total = np.float32(
                chance_total
                + np.float32(leaf.probability)
                * np.float32(evaluator.mode_value(leaf.board, mode))
            )
        future = np.float32(chance_total / np.float32(root_action.empty_cell_count))
        scores[root_action.action] = _f32(
            np.float32(root_action.immediate_reward) + future
        )
    return scores


def best_action(scores: list[float]) -> int:
    finite = [(score, action) for action, score in enumerate(scores) if math.isfinite(score)]
    if not finite:
        return 0
    maximum = max(score for score, _ in finite)
    return min(
        action
        for score, action in finite
        if score >= maximum - ACTION_TIE_TOLERANCE
    )


def score_margin(scores: list[float]) -> float | None:
    finite_scores = sorted(
        (score for score in scores if math.isfinite(score)), reverse=True
    )
    if len(finite_scores) < 2:
        return None
    return float(finite_scores[0] - finite_scores[1])


def root_diagnostic(
    board: np.ndarray,
    evaluator: CriticEvaluator,
    simulator: MoveSimulator,
) -> dict:
    expansion = expand_depth_one(board, simulator)
    scores = {
        mode: score_expansion(expansion, evaluator, mode)
        for mode in ("canonical", "raw", "average")
    }
    leaf_metrics = [
        evaluator.symmetry_metrics(leaf.board)
        for root_action in expansion
        for leaf in root_action.leaves
    ]
    max_leaf_spread = max(
        (float(item["spread"]) for item in leaf_metrics), default=0.0
    )
    max_raw_canonical = max(
        (float(item["raw_vs_canonical"]) for item in leaf_metrics), default=0.0
    )
    max_canonical_average = max(
        (float(item["canonical_vs_average"]) for item in leaf_metrics), default=0.0
    )
    max_raw_average = max(
        (float(item["raw_vs_average"]) for item in leaf_metrics), default=0.0
    )
    actions = {mode: best_action(scores[mode]) for mode in scores}

    def max_score_delta(first: str, second: str) -> float:
        return max(
            abs(scores[first][action] - scores[second][action])
            for action in range(ACTION_COUNT)
            if math.isfinite(scores[first][action])
            and math.isfinite(scores[second][action])
        )

    return {
        "board": board.astype(int).tolist(),
        "scores": scores,
        "actions": actions,
        "margins": {mode: score_margin(scores[mode]) for mode in scores},
        "max_leaf_symmetry_spread": max_leaf_spread,
        "max_leaf_raw_vs_canonical": max_raw_canonical,
        "max_leaf_canonical_vs_average": max_canonical_average,
        "max_leaf_raw_vs_average": max_raw_average,
        "max_root_score_delta_ab": max_score_delta("canonical", "raw"),
        "max_root_score_delta_ac": max_score_delta("canonical", "average"),
        "max_root_score_delta_bc": max_score_delta("raw", "average"),
    }


def fixed_symmetry_summary(
    boards: np.ndarray, evaluator: CriticEvaluator
) -> dict:
    boards = np.asarray(boards, dtype=np.int32)
    if boards.ndim == 4 and boards.shape[1] == 1:
        boards = boards[:, 0]
    boards = np.stack([_as_board(board) for board in boards])

    all_symmetries = np.concatenate(
        [symmetry_boards(board) for board in boards], axis=0
    )
    all_values = np.asarray(evaluator.predict_many(all_symmetries), dtype=np.float32)
    values = all_values.reshape(len(boards), NUM_TRANSFORMS)
    packed_symmetries = np.asarray(
        [[_pack(symmetry) for symmetry in symmetry_boards(board)] for board in boards],
        dtype=np.uint64,
    )
    canonical_indices = np.argmin(packed_symmetries, axis=1)
    raw_values = values[:, 0]
    canonical_values = values[np.arange(len(boards)), canonical_indices]
    average_values = values.mean(axis=1, dtype=np.float32)
    spreads = values.max(axis=1) - values.min(axis=1)
    raw_canonical = np.abs(raw_values - canonical_values)
    canonical_average = np.abs(canonical_values - average_values)
    raw_average = np.abs(raw_values - average_values)

    def stats(array: np.ndarray) -> dict:
        return _distribution(array.tolist())

    return {
        "boards": int(len(boards)),
        "transforms_per_board": NUM_TRANSFORMS,
        "symmetry_spread": stats(spreads),
        "raw_vs_canonical_abs": stats(raw_canonical),
        "canonical_vs_average_abs": stats(canonical_average),
        "raw_vs_average_abs": stats(raw_average),
        "signed_raw_minus_canonical_mean": float(
            np.mean(raw_values - canonical_values)
        ),
        "signed_canonical_minus_average_mean": float(
            np.mean(canonical_values - average_values)
        ),
    }


def collect_current_a_roots(
    evaluator: CriticEvaluator,
    episodes: int,
    base_seed: int,
    max_roots: int,
) -> np.ndarray:
    roots: list[np.ndarray] = []
    searcher = ExpectimaxSearcher()
    env = Game2048Env()
    try:
        for episode in range(episodes):
            searcher.clear_tt()
            observation, _ = env.reset(seed=base_seed + episode)
            del observation
            done = False
            while not done:
                board = env.game.board.copy()
                stats = searcher.find_best_move(
                    board, 1, evaluator.current_a_callback
                )
                roots.append(board)
                _, _, done, _, _ = env.step(int(stats["best_move"]))
    finally:
        env.close()

    if not roots:
        return np.empty((0, 4, 4), dtype=np.int32)
    roots_array = np.stack(roots).astype(np.int32, copy=False)
    if len(roots_array) <= max_roots:
        return roots_array
    selector = np.linspace(0, len(roots_array) - 1, max_roots, dtype=int)
    return roots_array[selector]


def validate_diagnostic_a(
    boards: np.ndarray,
    evaluator: CriticEvaluator,
    simulator: MoveSimulator,
    sample_size: int = 32,
) -> dict:
    sample = boards[:sample_size]
    searcher = ExpectimaxSearcher()
    max_score_error = 0.0
    action_mismatches = 0
    compared = 0
    mismatch_examples: list[dict] = []
    try:
        for board in sample:
            diagnostic = root_diagnostic(board, evaluator, simulator)
            current = searcher.find_best_move(
                board, 1, evaluator.current_a_callback
            )
            expected_scores = diagnostic["scores"]["canonical"]
            actual_scores = current["move_scores"]
            for expected, actual in zip(expected_scores, actual_scores):
                if math.isfinite(expected) and math.isfinite(actual):
                    max_score_error = max(max_score_error, abs(expected - actual))
            if diagnostic["actions"]["canonical"] != current["best_move"]:
                action_mismatches += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(
                        {
                            "board": board.astype(int).tolist(),
                            "diagnostic_action": diagnostic["actions"]["canonical"],
                            "current_action": int(current["best_move"]),
                            "diagnostic_scores": expected_scores,
                            "current_scores": actual_scores,
                        }
                    )
            compared += 1
    finally:
        searcher.clear_tt()
    return {
        "boards_compared": compared,
        "max_score_abs_error": max_score_error,
        "action_mismatches": action_mismatches,
        "mismatch_examples": mismatch_examples,
        "passed": action_mismatches == 0 and max_score_error <= 1e-4,
    }


def action_sensitivity_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"roots": 0}

    pairs = (("canonical", "raw", "ab"), ("canonical", "average", "ac"), ("raw", "average", "bc"))
    pair_summary = {}
    for first, second, label in pairs:
        flips = [
            row["actions"][first] != row["actions"][second] for row in rows
        ]
        pair_summary[label] = {
            "agreement_pct": 100.0 * (1.0 - float(np.mean(flips))),
            "flip_count": int(sum(flips)),
            "flip_pct": 100.0 * float(np.mean(flips)),
        }

    margins = {
        mode: _distribution(
            [row["margins"][mode] for row in rows if row["margins"][mode] is not None]
        )
        for mode in ("canonical", "raw", "average")
    }
    current_margin_rows = [
        row for row in rows if row["margins"]["canonical"] is not None
    ]

    def conditional_counts(
        pair: str,
        error_key: str,
        root_delta_key: str,
    ) -> dict[str, int]:
        first, second = {
            "ab": ("canonical", "raw"),
            "ac": ("canonical", "average"),
            "bc": ("raw", "average"),
        }[pair]
        flip_rows = [
            row
            for row in current_margin_rows
            if row["actions"][first] != row["actions"][second]
        ]
        return {
            "flip_count": len(flip_rows),
            "flip_and_leaf_symmetry_error_gt_a_margin": sum(
                row[error_key] > row["margins"]["canonical"]
                for row in flip_rows
            ),
            "flip_and_root_score_delta_gt_a_margin": sum(
                row[root_delta_key] > row["margins"]["canonical"]
                for row in flip_rows
            ),
        }

    all_three_agree = sum(
        len({row["actions"][mode] for mode in ("canonical", "raw", "average")}) == 1
        for row in rows
    )
    any_flip = len(rows) - all_three_agree
    return {
        "roots": len(rows),
        "all_three_agreement_pct": 100.0 * all_three_agree / len(rows),
        "any_action_flip_pct": 100.0 * any_flip / len(rows),
        "pairwise": pair_summary,
        "score_margin": margins,
        "flip_conditioning": {
            "ab": conditional_counts(
                "ab", "max_leaf_raw_vs_canonical", "max_root_score_delta_ab"
            ),
            "ac": conditional_counts(
                "ac", "max_leaf_canonical_vs_average", "max_root_score_delta_ac"
            ),
            "bc": conditional_counts(
                "bc", "max_leaf_raw_vs_average", "max_root_score_delta_bc"
            ),
        },
        "leaf_symmetry_error": {
            "max_spread": _distribution(
                row["max_leaf_symmetry_spread"] for row in rows
            ),
            "max_raw_vs_canonical": _distribution(
                row["max_leaf_raw_vs_canonical"] for row in rows
            ),
            "max_canonical_vs_average": _distribution(
                row["max_leaf_canonical_vs_average"] for row in rows
            ),
        },
    }


def run_small_episodes(
    evaluator: CriticEvaluator,
    mode: str,
    seeds: list[int],
) -> dict:
    scores: list[int] = []
    steps: list[int] = []
    max_tiles: list[int] = []
    unresolved = 0
    cap_hits = 0
    env = Game2048Env()
    current_searcher = ExpectimaxSearcher() if mode == "current" else None
    simulator = MoveSimulator()
    try:
        for seed in seeds:
            if current_searcher is not None:
                current_searcher.clear_tt()
            env.reset(seed=seed)
            done = False
            episode_steps = 0
            while not done:
                board = env.game.board.copy()
                if current_searcher is not None:
                    stats = current_searcher.find_best_move(
                        board, 1, evaluator.current_a_callback
                    )
                    action = int(stats["best_move"])
                    unresolved += int(stats["moves_unresolved"])
                    cap_hits += int(stats["cap_hits"])
                else:
                    expansion = expand_depth_one(board, simulator)
                    action = best_action(score_expansion(expansion, evaluator, mode))
                _, _, done, _, _ = env.step(action)
                episode_steps += 1
            scores.append(int(env.game.score))
            steps.append(episode_steps)
            max_tiles.append(2 ** int(env.game.max_tile))
    finally:
        env.close()

    return {
        "mode": mode,
        "episodes": len(seeds),
        "eval_seeds": seeds,
        "score": _distribution(scores),
        "steps": _distribution(steps),
        "max_tile": {
            "mean": float(np.mean(max_tiles)),
            "median": float(np.median(max_tiles)),
            "max": int(max(max_tiles)),
            "counts": {str(tile): max_tiles.count(tile) for tile in sorted(set(max_tiles))},
        },
        "moves_unresolved": unresolved,
        "cap_hits": cap_hits,
    }


def run_condition(
    label: str,
    model_path: Path,
    fixed_boards: np.ndarray,
    device: str,
    root_episodes: int,
    root_seed: int,
    max_roots: int,
    eval_episodes: int,
    batch_size: int,
) -> dict:
    evaluator = CriticEvaluator(model_path, device, batch_size)
    simulator = MoveSimulator()
    fixed_summary = fixed_symmetry_summary(fixed_boards, evaluator)
    roots = collect_current_a_roots(
        evaluator, root_episodes, root_seed, max_roots
    )
    root_summary = fixed_symmetry_summary(roots, evaluator)
    validation = validate_diagnostic_a(fixed_boards, evaluator, simulator)
    rows = [root_diagnostic(board, evaluator, simulator) for board in roots]
    eval_seeds = [root_seed + index for index in range(eval_episodes)]
    episode_results = [
        run_small_episodes(evaluator, mode, eval_seeds)
        for mode in ("current", "raw", "average")
    ]
    return {
        "model_path": str(model_path),
        "model_num_timesteps": int(evaluator.model.num_timesteps),
        "fixed_corpus": fixed_summary,
        "episode_roots": {
            **root_summary,
            "root_episodes": root_episodes,
            "root_seed": root_seed,
            "root_states_used": int(len(roots)),
        },
        "diagnostic_a_validation": validation,
        "action_sensitivity": action_sensitivity_summary(rows),
        "small_episode_evaluation": episode_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d4-model", type=Path, required=True,
        help="Path to the D4-trained model artifact.",
    )
    parser.add_argument(
        "--no-d4-model", type=Path, required=True,
        help="Path to the comparison model artifact.",
    )
    parser.add_argument(
        "--fixed-corpus", type=Path, required=True,
        help="Path to the fixed board corpus (.npz).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--root-episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--max-root-states", type=int, default=DEFAULT_MAX_ROOTS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.fixed_corpus.is_file():
        raise FileNotFoundError(f"Missing fixed corpus: {args.fixed_corpus}")
    for model_path in (args.d4_model, args.no_d4_model):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")

    corpus = np.load(args.fixed_corpus)
    fixed_boards = corpus["boards"]
    if fixed_boards.ndim == 4 and fixed_boards.shape[1] == 1:
        fixed_boards = fixed_boards[:, 0]
    result = {
        "diagnostic_schema_version": "1.0.0",
        "depth": 1,
        "fixed_corpus_path": str(args.fixed_corpus),
        "root_seed": args.root_seed,
        "root_episodes": args.root_episodes,
        "max_root_states": args.max_root_states,
        "evaluation_episodes": args.eval_episodes,
        "evaluation_seed_strategy": "same consecutive seeds for current/raw/average",
        "tt_mode": "disabled in diagnostic A/B/C; production A validated separately",
        "conditions": {
            "d4": run_condition(
                "d4",
                args.d4_model,
                fixed_boards,
                args.device,
                args.root_episodes,
                args.root_seed,
                args.max_root_states,
                args.eval_episodes,
                args.batch_size,
            ),
            "no_d4": run_condition(
                "no_d4",
                args.no_d4_model,
                fixed_boards,
                args.device,
                args.root_episodes,
                args.root_seed,
                args.max_root_states,
                args.eval_episodes,
                args.batch_size,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output), "status": "completed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
