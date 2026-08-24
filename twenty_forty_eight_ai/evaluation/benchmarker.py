"""Benchmarker: runs episodes of 2048 and returns structured EpisodeResult.

Lives in a neutral evaluation module, not agent/. Imports cleanly from
both the worker subprocess and the master process.
"""
from __future__ import annotations

import hashlib
import math
import os
import statistics
import time
from collections.abc import Sequence
from numbers import Real

import numpy as np
from sb3_contrib import MaskablePPO  # Compatibility seam for legacy tests/models.

from twenty_forty_eight_ai.evaluation.value_evaluator import D4ValueEvaluator
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.agent.ppo import load_ppo_model
from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher

from scripts.benchmark_io import (
    EpisodeResult, MoveRecord, EPISODE_SCHEMA_VERSION, ACTION_NAMES,
)


_WIN_THRESHOLDS = (1024, 2048, 4096, 8192)
_SEARCH_INT_METRICS = (
    "nodes_visited",
    "batches_eval",
    "tt_lookups",
    "tt_hits",
    "tt_collisions",
    "tt_same_key_overwrites",
    "moves_resolved",
    "moves_unresolved",
    "cap_hits",
    "chance_nodes_evaluated",
    "max_nodes_evaluated",
    "chance_value_count",
)
_SEARCH_FLOAT_METRICS = ("think_ms", "chance_value_sum")


class BenchmarkEpisodeError(RuntimeError):
    """Raised when an episode cannot safely produce its next action."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


class Benchmarker:
    """Runs episodes of 2048 with optional expectimax search."""

    def __init__(
        self,
        model_path: str,
        use_expectimax: bool,
        search_depth: int,
        device: str = "auto",
        max_steps: int = 10000,
    ) -> None:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        self.use_expectimax = use_expectimax
        self.search_depth = search_depth
        self.model_path = model_path
        self.max_steps = max_steps

        self.model = load_ppo_model(model_path, device=device)
        self.model.policy.eval()
        self.device = self.model.device
        self.value_evaluator = D4ValueEvaluator(self.model.policy, self.device)

        self.env = Game2048Env()

        self.searcher: ExpectimaxSearcher | None = None
        if self.use_expectimax:
            self.searcher = ExpectimaxSearcher()

        self._force_crash = os.environ.get("BENCHMARK_FORCE_CRASH") == "1"

    # The C++ searcher calls this callback per batch of leaf boards.
    def _evaluate_batch(self, boards_list: list) -> list:
        return self.value_evaluator(boards_list)

    def run_episode(
        self,
        eval_seed: int,
        log_moves: bool,
        run_id: str,
        worker_id: int = 0,
        episode_idx: int = 0,
    ) -> EpisodeResult:
        """Run one episode, return an EpisodeResult with full per-episode metrics.

        Per-episode accumulators (move times, empty cells, merge score,
        search counters) are tracked regardless of `log_moves`. The full
        MoveRecord list is appended only when `log_moves=True`.
        """
        if self._force_crash:
            raise RuntimeError("simulated worker crash (BENCHMARK_FORCE_CRASH=1)")

        if self.searcher is not None:
            self.searcher.clear_tt()

        # The environment owns independent game and D4 RNG streams. Seeding
        # the reset makes this episode independent of worker order.
        obs, _ = self.env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        termination_reason = ""
        steps = 0
        t_episode_start = time.perf_counter()

        move_times_ms: list = []
        empty_cells_samples: list = []
        merge_score_samples: list = []
        move_records: list = []

        total_think_ms = 0.0
        total_nodes = 0
        total_batches = 0
        total_tt_lookups = 0
        total_tt_hits = 0
        total_tt_collisions = 0
        total_tt_same_key_overwrites = 0
        total_moves_resolved = 0
        total_moves_unresolved = 0
        total_cap_hits = 0
        total_chance_nodes = 0
        total_max_nodes = 0
        chance_value_sum = 0.0
        chance_value_count = 0

        while True:
            board_before = self.env.game.board.copy()
            empty_cells_before = int((board_before == 0).sum())
            max_log_tile_before = int(board_before.max()) if board_before.any() else 0
            max_tile_before = 2 ** max_log_tile_before
            action_space = getattr(self.env, "action_space", None)
            action_space_n = getattr(action_space, "n", None)
            mask = self._validate_action_mask(
                self.env.action_masks(), action_space_n,
            )
            n_legal_actions = int(mask.sum())

            if not mask.any():
                terminated = True
                termination_reason = "terminal_board"
                break

            t0 = time.perf_counter()
            stats = None
            scores: list = [float("nan")] * 4
            if self.searcher is not None:
                stats = self.searcher.find_best_move(
                    board_before, self.search_depth, self._evaluate_batch
                )
                self._validate_search_stats(stats, mask)
                action = stats["best_move"]
                scores = list(stats["move_scores"])
            else:
                action, _ = self.model.predict(
                    obs, action_masks=mask, deterministic=True
                )
            action = self._validate_action(action, mask)

            obs, reward, step_terminated, step_truncated, info = self.env.step(action)
            if (
                type(step_terminated) is not bool
                or type(step_truncated) is not bool
                or (step_terminated and step_truncated)
            ):
                raise BenchmarkEpisodeError(
                    "invalid_env_flags",
                    "environment flags must be bool and mutually exclusive",
                )
            t1 = time.perf_counter()
            move_time_ms = (t1 - t0) * 1000.0

            merge_score = self._require_merge_score(info)
            move_times_ms.append(move_time_ms)
            empty_cells_samples.append(empty_cells_before)
            merge_score_samples.append(merge_score)

            if stats is not None:
                total_think_ms += float(stats["think_ms"])
                total_nodes += int(stats["nodes_visited"])
                total_batches += int(stats["batches_eval"])
                total_tt_lookups += int(stats["tt_lookups"])
                total_tt_hits += int(stats["tt_hits"])
                total_tt_collisions += int(stats["tt_collisions"])
                total_tt_same_key_overwrites += int(stats["tt_same_key_overwrites"])
                total_moves_resolved += int(stats["moves_resolved"])
                total_moves_unresolved += int(stats["moves_unresolved"])
                total_cap_hits += int(stats["cap_hits"])
                total_chance_nodes += int(stats["chance_nodes_evaluated"])
                total_max_nodes += int(stats["max_nodes_evaluated"])
                chance_value_sum += float(stats["chance_value_sum"])
                chance_value_count += int(stats["chance_value_count"])

            if log_moves:
                move_records.append(self._build_move_record(
                    run_id=run_id,
                    episode_idx=episode_idx,
                    move_idx=steps,
                    worker_id=worker_id,
                    action=action,
                    board_before=board_before,
                    empty_cells_before=empty_cells_before,
                    max_tile_before=max_tile_before,
                    max_log_tile_before=max_log_tile_before,
                    n_legal_actions=n_legal_actions,
                    merge_score=merge_score,
                    move_time_ms=move_time_ms,
                    scores=scores,
                    stats=stats,
                ))

            steps += 1

            terminated = bool(step_terminated)
            truncated = bool(step_truncated)
            if terminated:
                termination_reason = "board_full"
                break
            if truncated:
                termination_reason = "env_truncated"
                break
            if steps >= self.max_steps:
                truncated = True
                termination_reason = "max_steps"
                break

        t_episode_end = time.perf_counter()
        episode_time_s = t_episode_end - t_episode_start

        score = int(self.env.game.score)
        max_log_tile = int(self.env.game.max_tile)
        max_tile = 2 ** max_log_tile

        win_booleans = {t: (max_tile >= t) for t in _WIN_THRESHOLDS}

        mean_move_time_ms = statistics.fmean(move_times_ms) if move_times_ms else 0.0
        median_move_time_ms = (
            statistics.median(move_times_ms) if move_times_ms else 0.0
        )
        p95_move_time_ms = (
            float(np.percentile(move_times_ms, 95)) if move_times_ms else 0.0
        )
        max_move_time_ms = max(move_times_ms) if move_times_ms else 0.0
        mean_empty_cells = (
            statistics.fmean(empty_cells_samples) if empty_cells_samples else 0.0
        )
        min_empty_cells = (
            int(min(empty_cells_samples)) if empty_cells_samples else 0
        )
        mean_merge_score = (
            statistics.fmean(merge_score_samples) if merge_score_samples else 0.0
        )
        mean_nps = (
            total_nodes / (total_think_ms / 1000.0) if total_think_ms > 0 else 0.0
        )
        mean_tt_hit_rate = (
            total_tt_hits / total_tt_lookups if total_tt_lookups > 0 else 0.0
        )
        mean_nodes_per_batch_call = (
            total_nodes / total_batches if total_batches > 0 else 0.0
        )
        mean_chance_value = (
            chance_value_sum / chance_value_count if chance_value_count > 0 else 0.0
        )

        return EpisodeResult(
            schema_version=EPISODE_SCHEMA_VERSION,
            run_id=run_id,
            episode_idx=episode_idx,
            worker_id=worker_id,
            train_seed=None,
            eval_seed=eval_seed,
            requested_depth=self.search_depth,
            effective_depth=self.search_depth,
            use_expectimax=self.use_expectimax,
            score=score,
            max_tile=max_tile,
            max_log_tile=max_log_tile,
            steps=steps,
            episode_time_s=episode_time_s,
            mean_move_time_ms=mean_move_time_ms,
            median_move_time_ms=median_move_time_ms,
            p95_move_time_ms=p95_move_time_ms,
            max_move_time_ms=max_move_time_ms,
            termination_reason=termination_reason,
            terminated=terminated,
            truncated=truncated,
            win_1024=win_booleans[1024],
            win_2048=win_booleans[2048],
            win_4096=win_booleans[4096],
            win_8192=win_booleans[8192],
            total_think_ms=total_think_ms,
            total_nodes=total_nodes,
            total_batches=total_batches,
            total_tt_lookups=total_tt_lookups,
            total_tt_hits=total_tt_hits,
            total_tt_collisions=total_tt_collisions,
            total_tt_same_key_overwrites=total_tt_same_key_overwrites,
            total_moves_resolved=total_moves_resolved,
            total_moves_unresolved=total_moves_unresolved,
            total_cap_hits=total_cap_hits,
            total_chance_nodes=total_chance_nodes,
            total_max_nodes=total_max_nodes,
            mean_chance_value=mean_chance_value,
            mean_empty_cells=mean_empty_cells,
            min_empty_cells=min_empty_cells,
            mean_merge_score=mean_merge_score,
            mean_nps=mean_nps,
            mean_tt_hit_rate=mean_tt_hit_rate,
            mean_nodes_per_batch_call=mean_nodes_per_batch_call,
            move_records=move_records,
        )

    @staticmethod
    def _validate_search_stats(stats: dict, action_mask: np.ndarray) -> None:
        """Reject incomplete search output before it can drive an environment step."""
        if not isinstance(stats, dict):
            raise BenchmarkEpisodeError(
                "invalid_search_result", "search result must be a dict"
            )
        for field in _SEARCH_INT_METRICS:
            value = stats.get(field)
            if (
                field not in stats
                or isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    f"search result field {field} must be a non-negative integer",
                )
        for field in _SEARCH_FLOAT_METRICS:
            value = stats.get(field)
            if (
                field not in stats
                or isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or (field == "think_ms" and value < 0)
            ):
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    f"search result field {field} must be a finite real number",
                )
        for field in ("search_complete", "has_legal_move"):
            if field not in stats or type(stats[field]) is not bool:
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    f"search result field {field} must be a bool",
                )
        for field in ("cap_hits", "moves_unresolved"):
            value = stats.get(field)
            if (
                field not in stats
                or isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    f"search result field {field} must be a non-negative integer",
                )
        if stats["cap_hits"] > 0:
            raise BenchmarkEpisodeError("search_cap_hit", "search reported cap hits")
        if stats["moves_unresolved"] > 0:
            raise BenchmarkEpisodeError(
                "search_moves_unresolved", "search reported unresolved moves"
            )
        if not stats["search_complete"]:
            raise BenchmarkEpisodeError(
                str(stats.get("failure_reason") or "search_incomplete"),
                "search did not complete",
            )
        if not stats["has_legal_move"]:
            raise BenchmarkEpisodeError(
                str(stats.get("failure_reason") or "no_legal_move"),
                "search found no legal move",
            )
        scores = stats.get("move_scores")
        if (
            isinstance(scores, (str, bytes))
            or not isinstance(scores, Sequence)
            or len(scores) != 4
        ):
            raise BenchmarkEpisodeError(
                "invalid_search_result",
                "search result move_scores must contain four values",
            )
        for index, score in enumerate(scores):
            if isinstance(score, bool) or not isinstance(score, Real):
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    "search result move_scores must contain real numbers",
                )
            try:
                score_value = float(score)
            except (OverflowError, TypeError, ValueError) as exc:
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    "search result move_scores must contain real numbers",
                ) from exc
            if math.isnan(score_value) or score_value == math.inf:
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    "search result move_scores contains an invalid infinity",
                )
            if score_value == -math.inf and bool(action_mask[index]):
                raise BenchmarkEpisodeError(
                    "invalid_search_result",
                    "negative infinity is only valid for illegal moves",
                )
        best_move = stats.get("best_move")
        try:
            is_negative = best_move < 0
        except TypeError:
            is_negative = False
        if best_move is None or is_negative:
            raise BenchmarkEpisodeError(
                "invalid_search_move", "search returned an invalid best move"
            )

    @staticmethod
    def _require_merge_score(info) -> int:
        """Require the environment to report the score produced by this move."""
        if not isinstance(info, dict) or "merge_score" not in info:
            raise BenchmarkEpisodeError(
                "missing_merge_score",
                "environment step info must include merge_score",
            )
        merge_score = info["merge_score"]
        if (
            isinstance(merge_score, bool)
            or not isinstance(merge_score, (int, np.integer))
            or merge_score < 0
        ):
            raise BenchmarkEpisodeError(
                "invalid_merge_score",
                "environment merge_score must be a non-negative integer",
            )
        return int(merge_score)

    @staticmethod
    def _validate_action_mask(action_mask, action_space_n=None) -> np.ndarray:
        """Require the fixed four-action boolean contract before selecting a move."""
        try:
            mask = np.asarray(action_mask)
        except (TypeError, ValueError) as exc:
            raise BenchmarkEpisodeError(
                "invalid_action_mask", "action mask must be four boolean actions"
            ) from exc

        if action_space_n is not None and action_space_n != 4:
            raise BenchmarkEpisodeError(
                "invalid_action_mask", "environment action_space.n must be 4"
            )
        if mask.shape != (4,) or mask.dtype != np.dtype(bool):
            raise BenchmarkEpisodeError(
                "invalid_action_mask", "action mask must be four boolean actions"
            )
        return mask

    @staticmethod
    def _validate_action(action, mask: np.ndarray) -> int:
        """Validate an action's type, range, and legality against the current mask."""
        if isinstance(action, np.ndarray):
            if action.ndim != 0:
                raise BenchmarkEpisodeError(
                    "invalid_action", "action must be a scalar integer"
                )
            action = action.item()

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise BenchmarkEpisodeError(
                "invalid_action", "action must be an integer"
            )

        action = int(action)
        if action < 0 or action > 3:
            raise BenchmarkEpisodeError(
                "invalid_action", f"action {action} is outside the action space"
            )
        if not bool(mask[action]):
            raise BenchmarkEpisodeError(
                "illegal_action", f"action {action} is not legal"
            )
        return action

    def _build_move_record(
        self,
        run_id: str,
        episode_idx: int,
        move_idx: int,
        worker_id: int,
        action: int,
        board_before: np.ndarray,
        empty_cells_before: int,
        max_tile_before: int,
        max_log_tile_before: int,
        n_legal_actions: int,
        merge_score: int,
        move_time_ms: float,
        scores: list,
        stats,
    ) -> MoveRecord:
        """Build a MoveRecord from the pre-action snapshot and the move's outcome."""
        from twenty_forty_eight_ai.utils.searcher import _load_searcher_impl
        cpp = _load_searcher_impl()
        canonical_hash = str(cpp.BoardEncoder.canonicalize_board(board_before))

        if stats is not None:
            nodes = int(stats["nodes_visited"])
            batches = int(stats["batches_eval"])
            tt_lookups = int(stats["tt_lookups"])
            tt_hits = int(stats["tt_hits"])
            tt_hit_rate = (tt_hits / tt_lookups) if tt_lookups > 0 else 0.0
            think_ms = float(stats["think_ms"])
            moves_resolved = int(stats["moves_resolved"])
            moves_unresolved = int(stats["moves_unresolved"])
            cap_hits = int(stats["cap_hits"])
            is_unresolved = moves_unresolved > 0
            is_cap_hit = cap_hits > 0
            best_move = int(stats["best_move"])
        else:
            nodes = 0
            batches = 0
            tt_lookups = 0
            tt_hits = 0
            tt_hit_rate = 0.0
            think_ms = 0.0
            moves_resolved = 0
            moves_unresolved = 0
            cap_hits = 0
            is_unresolved = False
            is_cap_hit = False
            best_move = action

        nodes_per_batch_call = (nodes / batches) if batches > 0 else 0.0
        board_state = ",".join(str(int(v)) for v in board_before.flatten().tolist())

        return MoveRecord(
            schema_version=EPISODE_SCHEMA_VERSION,
            run_id=run_id,
            episode_idx=episode_idx,
            move_idx=move_idx,
            worker_id=worker_id,
            action=action,
            action_name=ACTION_NAMES.get(action, "?"),
            canonical_board_hash=canonical_hash,
            board_state=board_state,
            empty_cells_before=empty_cells_before,
            max_tile_before=max_tile_before,
            max_log_tile_before=max_log_tile_before,
            merge_score=merge_score,
            move_time_ms=move_time_ms,
            think_ms=think_ms,
            nodes_visited=nodes,
            batches_eval=batches,
            nodes_per_batch_call=nodes_per_batch_call,
            tt_hit_rate=tt_hit_rate,
            tt_lookups=tt_lookups,
            tt_hits=tt_hits,
            n_legal_actions=n_legal_actions,
            score_up=scores[0],
            score_right=scores[1],
            score_down=scores[2],
            score_left=scores[3],
            is_unresolved=is_unresolved,
            is_cap_hit=is_cap_hit,
            best_move=best_move,
        )


def md5_of_file(path: str) -> str | None:
    """Return the MD5 hex digest of a file, or None if unreadable."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None
