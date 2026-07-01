# Paper Benchmarks — Hybrid-PPO-Expectimax-v3

Final release-model benchmarks for the paper. Each row produces one run folder
under `data/benchmarks/<output>/` containing `config.json`, `results.json`,
`per_run.csv`, and `score_distribution.png`. `data/benchmarks/` is gitignored,
so each result folder is force-added (`git add -f`) when we commit.

**Model:** `data/models/release/Hybrid-PPO-Expectimax-v3.zip`
**Device convention:** `--device cuda --workers 1` for paper-grade latency.

## Benchmarks

- [ ] **B1** `paper_d3_n100` — depth=3, 100 episodes, no move logs
- [ ] **B2** `paper_d3_n100_logged` — depth=3, 100 episodes, with move logs
- [ ] **B3** `paper_d0_n100` — depth=0, 100 episodes (raw policy baseline)
- [ ] **B4** `paper_d1_n100` — depth=1, 100 episodes

## Per-benchmark checklist (reused)

For each row above, after running:

1. Run the benchmark command and wait for it to finish (or verify `status: "completed"` in `config.json`).
2. Sanity-check `results.json` — `metrics.avg_score`, `n_runs`, `win_rate_2048+`, `max_tile_dist`.
3. Commit the run folder + this checklist update.

## Notes

- Prior 30-run depth-3 result lives at `data/benchmarks/v3_depth3_final/` (avg_score ≈ 36276, depth-3 win-rate 2048+ ≈ 0.80) — kept for comparison.
- Wall-time estimate at depth 3, CUDA, 1 worker: ~7.6 h for 100 episodes (≈274 s/game from prior run). Depth 0 and depth 1 will be far faster.
- Each run records `model_md5`, `git_commit`, `cuda_device_name`, and `base_eval_seed` in `config.json` for reproducibility.