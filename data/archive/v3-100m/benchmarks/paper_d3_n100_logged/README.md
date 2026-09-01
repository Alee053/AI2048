# Historical/corrupt per-move log

This archived artifact must not be used as paper-grade row-level evidence.

The original `moves.csv` was checked without modification and contains 192,066
data rows but only 3,667 unique `(run_id, episode_idx, move_idx)` keys. There
are 3,424 duplicated key values and 188,399 duplicate rows beyond the first
occurrence. It contains one `run_id` and one `episode_idx`; the file is retained
unchanged solely for historical provenance.
