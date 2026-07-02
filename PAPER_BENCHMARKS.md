# Paper Benchmarks — Hybrid-PPO-Expectimax-v3

Final release-model benchmarks for the paper. Each row produces one run folder
under `data/benchmarks/<output>/` containing `config.json`, `results.json`,
`per_run.csv`, and `score_distribution.png`. `data/benchmarks/` is gitignored,
so each result folder is force-added (`git add -f`) when we commit.

**Model:** `data/models/release/Hybrid-PPO-Expectimax-v3.zip`
**Device convention:** `--device cuda --workers 1` for paper-grade latency.

## Benchmarks

- [x] **B1** `paper_d3_n100` — depth=3, 100 episodes, no move logs
- [x] **B2** `paper_d0_n100` — depth=0, 100 episodes, no move logs (raw policy baseline)
- [x] **B3** `paper_d1_n100` — depth=1, 100 episodes, no move logs
- [x] **B4** `paper_d2_n100` — depth=2, 100 episodes, no move logs
- [ ] **B5** `paper_d3_n100_logged` — depth=3, 100 episodes, with move logs (deferred — tomorrow)

## Schedule

- **Tonight (sequential, no move logs):** B2 → B3 → B4. Total wall time ≈ 3 h (depth 0 ≈ 8 min, depth 1 ≈ 35 min, depth 2 ≈ 2.2 h).
- **Tomorrow:** B5 (depth 3 with `--log-moves`). Wall time ≈ 8 h, produces ~700k-row moves.csv.

All runs share `--base-eval-seed 20482048` so per-episode tile-spawn sequences are identical across depths → scores are directly comparable for the depth ablation.

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

## B2 / B3 / B4 artifacts

- `data/benchmarks/paper_d0_n100/{config.json, episodes.csv, summary.json}`
- `data/benchmarks/paper_d1_n100/{config.json, episodes.csv, summary.json}`
- `data/benchmarks/paper_d2_n100/{config.json, episodes.csv, summary.json}`

## Per-benchmark checklist (reused)

For each row above, after running:

1. Run the benchmark command and wait for it to finish (or verify `status: "completed"` in `config.json`).
2. Sanity-check `results.json` — `metrics.avg_score`, `n_runs`, `win_rate_2048+`, `max_tile_dist`.
3. Commit the run folder + this checklist update.

## Notes

- Prior 30-run depth-3 result lives at `data/benchmarks/v3_depth3_final/` (avg_score ≈ 36276, depth-3 win-rate 2048+ ≈ 0.80) — kept for comparison.
- Wall-time estimate at depth 3, CUDA, 1 worker: ~7.6 h for 100 episodes (≈274 s/game from prior run). Depth 0 and depth 1 will be far faster.
- Each run records `model_md5`, `git_commit`, `cuda_device_name`, and `base_eval_seed` in `config.json` for reproducibility.