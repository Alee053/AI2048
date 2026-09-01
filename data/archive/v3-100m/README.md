# Archived v3 100M artifacts

These models and benchmark outputs are retained for historical reproducibility
and provenance. They are not paper-grade evidence and are intentionally outside
the default `data/benchmarks/` discovery path used by the current pipeline.

The D4 seed0 200M pilot is not stored here; it remains under
`data/official_200m/` and is diagnostic only.

`benchmarks/paper_d3_n100_logged/` is additionally marked historical/corrupt:
its `moves.csv` has non-unique `(run_id, episode_idx, move_idx)` keys. The
original `moves.csv` is preserved byte-for-byte and was not repaired; only
metadata paths were updated to reflect this archive location.
