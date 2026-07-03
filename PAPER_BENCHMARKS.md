# Paper Benchmarks — Hybrid-PPO-Expectimax-v3

Final release-model benchmarks for the paper. Each run folder under
`data/benchmarks/<output>/` contains `config.json`, `summary.json`,
`episodes.csv`, and (only for B5) `moves.csv`. `data/benchmarks/` is
gitignored, so each result folder is force-added (`git add -f`) on commit.

**Model:** `data/models/release/Hybrid-PPO-Expectimax-v3.zip`
**Device convention:** `--device cuda --workers 1` for paper-grade latency.

## Benchmarks

- [x] **B1** `paper_d3_n100` — depth=3, 100 episodes, no move logs
- [x] **B2** `paper_d0_n100` — depth=0, 100 episodes, no move logs (raw policy baseline)
- [x] **B3** `paper_d1_n100` — depth=1, 100 episodes, no move logs
- [x] **B4** `paper_d2_n100` — depth=2, 100 episodes, no move logs
- [x] **B5** `paper_d3_n100_logged` — depth=3, 100 episodes, with move logs

All runs share `--base-eval-seed 20482048`, model md5 `fab18d67…`, and `--device cuda --workers 1`. Per-episode tile-spawn sequences are identical across all five runs → scores are directly comparable across depths.

**Total wall time across all benchmarks:** ≈ 31 h
(B1: 8.1 h, B2: 1.5 min, B3: 4 min, B4: 23 min, B5: 8.3 h)

## B1 Results

| Metric | Value |
|---|---|
| avg_score | 38,430.76 |
| 95% CI | 35,316 — 41,546 |
| std_score | 15,893.73 |
| median_score | 35,508 |
| min — max | 14,640 — 79,204 |
| win_rate 1024+ | 1.00 |
| win_rate 2048+ | 0.87 |
| win_rate 4096+ | 0.24 |
| win_rate 8192+ | 0.00 |
| max_tile 1024 / 2048 / 4096 / 8192 | 13 / 63 / 24 / 0 |
| avg_steps | 1,920.66 |
| total_wall_time_s | 29,136.24 (≈ 8.1 h) |
| avg_time_per_game_s | 291.35 |
| nodes/sec | 485,708 |
| TT hit rate | 0.2034 |

Artifacts: `data/benchmarks/paper_d3_n100/{config.json, episodes.csv, summary.json}`

## B5 Results (depth=3, with move logs)

Identical configuration to B1 except `--log-moves` is enabled.

| Metric | B5 (logged) | B1 (no logs) | Δ |
|---|---:|---:|---:|
| avg_score | 38,430.76 | 38,430.76 | 0.00 |
| std_score | 15,893.73 | 15,893.73 | 0.00 |
| 95% CI | 35,316–41,546 | 35,316–41,546 | — |
| win_rate 1024+ / 2048+ / 4096+ | 1.00 / 0.87 / 0.24 | 1.00 / 0.87 / 0.24 | — |
| max_tile 1024 / 2048 / 4096 / 8192 | 13 / 63 / 24 / 0 | 13 / 63 / 24 / 0 | — |
| avg_steps | 1,920.66 | 1,920.66 | 0.00 |
| avg_nodes_visited | 140,691,276.85 | 140,691,276.85 | 0.00 |
| avg_nodes_per_sec | 473,255 | 485,708 | **−2.5 %** |
| avg_time_per_game (s) | 300.17 | 291.35 | **+3.0 %** |
| total_wall_time_s | 30,018.34 (≈ 8.34 h) | 29,136.24 | +882 s |
| moves.csv size | 54.3 MB / ~691k rows | — | — |

**Determinism validation:** B5 reproduces B1 **exactly** on every behavioral metric (score, win-rate, max-tile distribution, node counts, avg_steps). The only deltas are throughput-related (`nodes/sec` −2.5 %, `t/game` +3.0 %), attributable to per-move CSV write I/O. This confirms the harness is deterministic under fixed `(model, seed, depth)` and that `--log-moves` is a pure side-channel.

Artifacts: `data/benchmarks/paper_d3_n100_logged/{config.json, episodes.csv, moves.csv, summary.json}`

## Depth Ablation Summary (B1–B4)

All four runs share `--base-eval-seed 20482048`, model md5 `fab18d67…`, `--device cuda --workers 1`, 100 episodes each. Per-episode tile-spawn sequences are identical across depths, so score deltas are attributable to search depth alone.

| Depth | avg_score | 95% CI | std | median | win_1024+ | win_2048+ | win_4096+ | t/game (s) | wall (s) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6,080 | 5,473–6,687 | 3,096 | 5,598 | 0.07 | 0.00 | 0.00 | 0.92 | 92.6 |
| 1 | 7,930 | 7,192–8,668 | 3,764 | 7,510 | 0.22 | 0.01 | 0.00 | 2.38 | 239.1 |
| 2 | 20,696 | 18,770–22,621 | 9,824 | 16,574 | 0.90 | 0.35 | 0.02 | 14.01 | 1,401.8 |
| 3 | 38,431 | 35,316–41,546 | 15,894 | 35,508 | 1.00 | 0.87 | 0.24 | 291.35 | 29,136.2 |

**Observations**

- avg_score roughly **doubles per depth increment**: 6k → 8k → 21k → 38k.
- win_rate_2048+ exhibits a step change at depth 3 (0.35 → 0.87); depth 2 is the first depth where 4096+ becomes non-trivial.
- Time per game grows ~5–20× per depth (search branching factor); depth 3 is the first depth that takes longer than a typical interactive session.

## All artifacts

- `data/benchmarks/paper_d0_n100/{config.json, episodes.csv, summary.json}`
- `data/benchmarks/paper_d1_n100/{config.json, episodes.csv, summary.json}`
- `data/benchmarks/paper_d2_n100/{config.json, episodes.csv, summary.json}`
- `data/benchmarks/paper_d3_n100/{config.json, episodes.csv, summary.json}`
- `data/benchmarks/paper_d3_n100_logged/{config.json, episodes.csv, moves.csv, summary.json}`

## Per-benchmark checklist (reused)

For each row above, after running:

1. Run the benchmark command and wait for it to finish (or verify `status: "completed"` in `config.json`).
2. Sanity-check `summary.json` — `metrics.avg_score`, `n_completed`, `win_rates`, `max_tile_dist`.
3. Commit the run folder + this checklist update.

## Notes

- Prior 30-run depth-3 result lives at `data/benchmarks/v3_depth3_final/` (avg_score ≈ 36,276, win-rate 2048+ ≈ 0.80) — kept for comparison against the 100-run paper benchmark.
- Each run records `model_md5`, `git_commit`, `cuda_device_name`, and `base_eval_seed` in `config.json` for reproducibility.
- B5 (`paper_d3_n100_logged`) is a behavioral twin of B1 — same scores, same node counts, same per-episode outcomes — confirming that `--log-moves` is a pure side-channel. Use B5 for any per-move analysis (move timing, distribution of chosen actions, heatmaps), B1 for clean headline numbers.