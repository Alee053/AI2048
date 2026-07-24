"""Benchmarker: runs episodes of 2048 and returns structured EpisodeResult.

Lives in a neutral evaluation module, not agent/. Imports cleanly from
both the worker subprocess and the master process.
"""
from __future__ import annotations

import hashlib
import os
import statistics
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor

from scripts.benchmark_io import (
    EpisodeResult, MoveRecord, EPISODE_SCHEMA_VERSION, ACTION_NAMES,
)


_WIN_THRESHOLDS = (1024, 2048, 4096, 8192)


class Benchmarker:
    """Runs episodes of 2048 with optional expectimax search."""

    def __init__(
        self,
        model_path: str,
        use_expectimax: bool,
        search_depth: int,
        device: str = "auto",
    ) -> None:
        self.use_expectimax = use_expectimax
        self.search_depth = search_depth
        self.model_path = model_path

        self.model = MaskablePPO.load(model_path, device=device)
        self.model.policy.eval()
        self.device = self.model.device

        self.env = Game2048Env()

        self.searcher: ExpectimaxSearcher | None = None
        if self.use_expectimax:
            self.searcher = ExpectimaxSearcher()

        self._force_crash = os.environ.get("BENCHMARK_FORCE_CRASH") == "1"

    # The C++ searcher calls this callback per batch of leaf boards.
    def _evaluate_batch(self, boards_list: list) -> list:
        if not boards_list:
            return []
        batch_array = np.array(boards_list)
        batch_tensor = board_to_tensor(batch_array)
        with torch.no_grad():
            values = self.model.policy.predict_values(
                torch.as_tensor(batch_tensor).to(self.device)
            )
        return values.cpu().numpy().flatten().tolist()

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

        # Seed global numpy RNG with eval_seed so episode outcomes are
        # deterministic regardless of order or worker count. The Python
        # Fast2048.generate_random() uses the global np.random (not the
        # Gymnasium-env-scoped RNG), so this is the authoritative seed
        # source for tile spawns during this episode.
        np.random.seed(eval_seed)
        obs, _ = self.env.reset(seed=eval_seed)
        done = False
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
        total_alpha_beta_cuts = 0
        total_chance_nodes = 0
        total_max_nodes = 0
        chance_value_sum = 0.0
        chance_value_count = 0

        while not done:
            board_before = self.env.game.board.copy()
            empty_cells_before = int((board_before == 0).sum())
            max_log_tile_before = int(board_before.max()) if board_before.any() else 0
            max_tile_before = 2 ** max_log_tile_before
            mask = self.env.action_masks()
            n_legal_actions = int(mask.sum())

            t0 = time.perf_counter()
            stats = None
            scores: list = [float("nan")] * 4
            action = 0
            if self.searcher is not None:
                stats = self.searcher.find_best_move(
                    board_before, self.search_depth, self._evaluate_batch
                )
                action = int(stats["best_move"])
                scores = list(stats["move_scores"])
            else:
                action, _ = self.model.predict(
                    obs, action_masks=mask, deterministic=True
                )
                action = int(action)

            obs, reward, done, _, info = self.env.step(action)
            t1 = time.perf_counter()
            move_time_ms = (t1 - t0) * 1000.0

            merge_score = int(info.get("merge_score", 0))
            move_times_ms.append(move_time_ms)
            empty_cells_samples.append(empty_cells_before)
            merge_score_samples.append(merge_score)

            if stats is not None:
                total_think_ms += float(stats.get("think_ms", 0.0))
                total_nodes += int(stats.get("nodes_visited", 0))
                total_batches += int(stats.get("batches_eval", 0))
                total_tt_lookups += int(stats.get("tt_lookups", 0))
                total_tt_hits += int(stats.get("tt_hits", 0))
                total_tt_collisions += int(stats.get("tt_collisions", 0))
                total_tt_same_key_overwrites += int(
                    stats.get("tt_same_key_overwrites", 0)
                )
                total_moves_resolved += int(stats.get("moves_resolved", 0))
                total_moves_unresolved += int(stats.get("moves_unresolved", 0))
                total_cap_hits += int(stats.get("cap_hits", 0))
                total_alpha_beta_cuts += int(stats.get("alpha_beta_cuts", 0))
                total_chance_nodes += int(stats.get("chance_nodes_evaluated", 0))
                total_max_nodes += int(stats.get("max_nodes_evaluated", 0))
                chance_value_sum += float(stats.get("chance_value_sum", 0.0))
                chance_value_count += int(stats.get("chance_value_count", 0))

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
            termination_reason="board_full",
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
            total_alpha_beta_cuts=total_alpha_beta_cuts,
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
            nodes = int(stats.get("nodes_visited", 0))
            batches = int(stats.get("batches_eval", 0))
            tt_lookups = int(stats.get("tt_lookups", 0))
            tt_hits = int(stats.get("tt_hits", 0))
            tt_hit_rate = (tt_hits / tt_lookups) if tt_lookups > 0 else 0.0
            think_ms = float(stats.get("think_ms", 0.0))
            moves_resolved = int(stats.get("moves_resolved", 0))
            moves_unresolved = int(stats.get("moves_unresolved", 0))
            cap_hits = int(stats.get("cap_hits", 0))
            is_unresolved = moves_unresolved > 0
            is_cap_hit = cap_hits > 0
            best_move = int(stats.get("best_move", action))
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
