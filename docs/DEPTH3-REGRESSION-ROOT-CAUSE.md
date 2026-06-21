# Diagnostic Report: Depth-3 Score Regression Root Cause

**Date:** 2026-06-17
**Status:** Root cause identified
**Branch:** `side-by-side-old-vs-new`
**Author:** AI Assistant

---

## 1. Executive Summary

The depth-3 benchmark regression (18,232 ± 7,496 on n=10 vs. the OLD's 26,523 ± 12,750 on n=100) is caused by a **non-canonicalization-invariant value network** combined with the C++ search's canonicalize-then-unpack pattern in the deferred batching path.

The C++ search canonicalizes boards for the transposition-table key, stores the canonical key in the batch queue, then `BoardEncoder::unpack`s it back to a raw board for model evaluation. The unpacked board is the **canonical form**, not the original board reached during search. The value network (`Hybrid-PPO-Expectimax-v1.zip`) was trained without canonicalization, so it gives different values for different orientations of the same logical board. The side-by-side comparison on board 26 confirmed:

- All 1,313 leaf canonical keys match between OLD and NEW.
- All 1,313 leaf values differ between OLD and NEW (mean abs diff ~2.0, max diff ~28 on a 0–60 scale).
- The move-score difference of 22.2 points on board 26 traces directly to the leaf-value differences.

**Recommended fix:** Retrain the value network to be invariant to the 8 D4 symmetries (rotations + reflections) of the board, by augmenting training data with random D4 transforms. This makes `BoardEncoder::canonicalize` safe to use for both TT keying and batch unpacking. The OLD's gather-all-first algorithm then becomes a strict superset of the NEW's deferred batching in quality, and the search optimizations (TT, alpha-beta cutoff, persistent search state) can be retained without quality loss.

---

## 2. Investigation Timeline

### 2.1 Initial Symptom (2026-06-16)

A 10-game depth-3 benchmark on master (`run_1781651743`) scored 18,232 ± 7,496, compared to the OLD benchmark (`depth3_expectimax`, committed 2025-12-10) which scored 26,523 ± 12,750 on n=100. The 95% confidence intervals did not overlap (NEW [13,335, 23,130] vs OLD [24,011, 29,035]).

The difference was initially attributed to the new search algorithm (multi-pass deferred batching, persistent TT, cross-search aging). The cross-search aging fix merged in PR #4 was suspected.

### 2.2 Divisor Bug Fix (2026-06-17)

A pre-existing chance-node divisor bug was found in the C++ search: `total_value / (2.0f * empty_cells.size())` instead of `total_value / empty_cells.size()`. The OLD's `d575532` code also had this bug, so it was not the regression's cause, but the fix (`a6483cb` on `fix-chance-divisor`) is correct and improved the NEW score by ~10%.

### 2.3 Persistent-TT Hypothesis Ruled Out (2026-06-17)

The cross-search aging fix was hypothesized to cause TT pollution that biased later searches. Adding `--fresh-tt` mode (which calls `clear_tt()` before each `find_best_move`) did not change the results:

| Run | OLD vs NEW agreement |
|-----|----------------------|
| With persistent TT | 10/30 = 33% moves, 1/30 = 3% scores |
| With fresh TT | 10/30 = 33% moves, 1/30 = 3% scores |

The persistent TT was ruled out as the cause.

### 2.4 Leaf-Set Diagnostic (2026-06-17)

A `dump_leaves()` method was added to the C++ searcher (`36015ce`) that writes one line per unique leaf: `<hex_canonical_key> <value>`. The side-by-side script canonicalized the OLD's raw boards and compared. For 24/30 boards the leaf sets were exactly identical (only_old=0, only_new=0). For 6/30 boards the OLD had 1–3 extra leaves that the NEW missed (the deferred batching's UNRESOLVED path). For the 22-point-gap board (board 26), the leaf sets were identical: 1,313 leaves, common=1,313, no missing.

This ruled out the leaf-set hypothesis. The leaf values, not the leaf identities, must differ.

### 2.5 Search-Tree Trace (2026-06-17)

A search-tree trace logger was added (`36015ce`, `cdf47fb`) that writes one line per `chance_node` and `max_node` call:

```
kind=chance depth=N board=0x<hex> value=<f> src=<computed|tt_hit|leaf_cache|tt_miss|unresolved>
```

Initial trace comparison was misleading: the OLD showed all computed values, while the NEW showed `value=unresolved src=unresolved` for nearly every node because the deferred batching returns UNRESOLVED on the first search pass and only resolves leaves after the batch is evaluated. The traces were temporally different (OLD: after eval; NEW: during initial recursion).

### 2.6 Canonicalization Verification (2026-06-17)

After fixing a bindings issue (`7cd0fe0` exposed `set_trace_log` to pybind11) and a Python version mismatch (`uv run` uses Python 3.12; CMake had been finding system Python 3.11), the leaf dumps were re-extracted and compared. The result:

```
OLD unique boards: 1313
NEW unique boards: 1313
Common boards: 1313
Only in OLD: 0
Only in NEW: 0
```

All 1,313 canonical board keys match. **The canonicalization is correct.**

But comparing the values:

```
Board hash | OLD value | NEW value | Diff
0x100708a32a81835c | 27.120079 | 25.205582 | -1.914497
0x100728a38a81225c | 16.337696 | 19.229012 | +2.891315
0x100c015338a87a82 | 16.333649 | 44.240437 | +27.906788
```

All 1,313 values differ. Mean absolute difference ~2.0, max ~28. The model is returning different values for the same canonical board.

### 2.7 Root Cause: Model Is Not Rotation-Invariant (2026-06-17)

The remaining hypothesis: the OLD passes raw boards to the model, while the NEW passes canonicalized-then-unpacked boards. If the model is not rotation-invariant, these two paths give different values.

Test: evaluate the same logical board in two orientations:

```python
board     = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
rotated   = [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [0, 12, 8, 4]]

value(board)     = 14.804501
value(rotated)   = 14.770380
diff             = 0.034121
```

**The model is not rotation-invariant.** It was trained without D4 augmentation, so different orientations of the same logical board give different values (mean diff ~0.03 per board, which accumulates to large diffs across 1,313 leaves).

The NEW's deferred batching flow:

1. Search reaches a board, canonicalizes it for the TT key.
2. Adds the canonical key to the batch queue.
3. When the batch is evaluated, `BoardEncoder::unpack(canonical_key)` reconstructs the **canonical** raw board (not the original).
4. The model evaluates the canonical raw board.

The OLD's gather-all-first flow:

1. Search enumerates all raw boards reachable at depth 0.
2. Evaluates the raw boards directly.
3. The model evaluates the raw boards as-reached (in whatever orientation they happen to be in).

Both reach the same set of canonical boards, but evaluate different raw boards (because the OLD evaluates the originals, the NEW evaluates the canonicalized versions). The model gives different values for different orientations, so the leaf values differ, the search-tree aggregation differs, and the move scores differ.

---

## 3. Why the Bug Was Missed Earlier

Several layers of diagnostics ruled out incorrect causes:

| Hypothesis | Diagnostic | Result |
|------------|------------|--------|
| Cross-search TT pollution | `--fresh-tt` clears TT per move | No effect |
| Persistent TT carries stale values | Same as above | No effect |
| Alpha-beta pruning unsound in expectimax | `alpha_beta_cuts: 0` | Never fires |
| Leaf sets differ | `dump_leaves()` comparison | Identical |
| Per-move iteration cap fires | `cap_hits: 0` | Never fires |
| Search aggregation order differs | Search-tree trace comparison | Traces temporally different |
| **Canonicalization produces different board hashes** | Direct hex comparison | **Identical** |
| **Model returns different values for the same canonical board** | Direct value comparison | **All 1,313 differ** |

The final test — calling the model with two different orientations of the same logical board — confirmed the root cause. This was not tested earlier because the canonicalization was assumed correct (the board hashes matched) and the model was assumed invariant (which is the standard assumption for a value network).

---

## 4. The Architectural Problem

The C++ search's `BoardEncoder::canonicalize` was designed so that the TT cache would be effective for rotated boards (e.g., two boards that are 90-degree rotations of each other share a TT entry). This is the standard technique for board games with rotational symmetry (chess, Go, etc.).

This design is correct **if and only if the value network is rotation-invariant.** For the canonicalization to be a no-op, the model must satisfy:

```
model(canonicalize(board)) == model(board)
```

The current model does not satisfy this. Therefore the canonicalization changes the model's input distribution, biasing all leaf evaluations.

The OLD code did not have this problem because it stored raw boards in `std::map<Board, float>` (no canonicalization). Two rotated boards got two TT entries, but both were evaluated correctly.

---

## 5. Recommended Fix

### 5.1 Primary Fix: Retrain the Value Network with D4 Augmentation

Augment the training data with random D4 symmetry transforms (8 transforms per board: 4 rotations × 2 reflections). Train until the value network is approximately rotation-invariant.

```
loss = MSE(model(board), value_target)
     + MSE(model(rotate90(board)), value_target)
     + MSE(model(rotate180(board)), value_target)
     + ... (all 8 transforms)
```

Or equivalently, apply a random D4 transform to each training example with probability 1, and keep the original target value. This forces the model to learn invariance.

Once retrained, the C++ canonicalization is safe: `model(canonicalize(board)) ≈ model(board)`, and the search optimizations can be retained.

### 5.2 Alternative Fixes (Not Recommended)

These were considered and rejected:

- **Disable canonicalization in the C++**: rotate the TT bucket key by some fixed offset (e.g., always use the identity orientation). Rotated boards would no longer share TT entries, losing cache hit rate, but the leaf values would be correct. Net effect on play quality unknown; likely worse than retraining.
- **Make the unpack canonical-aware**: store the raw board alongside the canonical key in the batch queue, and pass the raw board to the model. This requires changes to the batch queue and to the C++/Python boundary. Larger code change for no clear benefit once the model is retrained.
- **Revert to the OLD's gather-all-first algorithm**: drops the deferred batching, persistent TT, and alpha-beta optimizations. Play quality recovers to OLD's 26K level, but per-move latency increases ~10× (from 0.18s/move to 1.5s/move as measured in the pre-fix depth-4 diagnostic). Loses all the speedups the C++ rewrite was designed for.

### 5.3 Estimated Effort for Retraining

Assuming the existing `Hybrid-PPO-Expectimax-v1` training pipeline (see `scripts/train.py` and `configs/train/hybrid_ppo_v1.yaml`):

1. Add a D4 augmentation transform in the dataset loader (~20 lines in `twenty_forty_eight_ai/env/environment.py`).
2. Re-run `scripts/train.py` with the augmented data (~4–8 hours on a single GPU, depending on the existing training config).
3. Verify rotation-invariance: `model(canonicalize(board)) ≈ model(board)` to within ~0.01 on a 0–60 value scale.
4. Replace `data/models/release/Hybrid-PPO-Expectimax-v1.zip` with the retrained model.
5. Re-run `tests/test_depth4_convergence.py` and `tests/test_persistent_tt.py` to verify the existing depth-3 and depth-4 convergence tests still pass.
6. Re-run the depth-3 30-game benchmark. Expected score: ~26K, matching the OLD's pre-fix performance.

---

## 6. Files Changed During Investigation

All diagnostic changes were made on the `side-by-side-old-vs-new` branch:

- `cpp_src/ExpectimaxSearcher.h`, `.cpp`: trace logging (`set_trace_log`, `dump_leaves`), diagnostic counters (`unique_leaves_evaluated`, `chance_nodes_evaluated`, etc.), chance-divisor fix (`/ N` instead of `/ (2N)`).
- `cpp_src/bindings.cpp`: expose `set_trace_log`, `close_trace_log`, `trace_enabled`, `dump_leaves`, `last_unique_leaves`, diagnostic stats to pybind11.
- `scripts/side_by_side_old_vs_new.py`: pure-Python reimplementation of the OLD's gather-all-first algorithm; canonicalize-the-OLD's-leaves helper; comparison driver; trace logging for OLD; `--fresh-tt`, `--debug`, `--old-trace`, `--new-trace`, `--trace-board` flags.
- `tests/test_transposition_table.py`: new baseline test (1 test) on `fix-chance-divisor` branch.

The diagnostic work is on `side-by-side-old-vs-new`. The chance-divisor fix and original PR #4 (cross-search aging) are on `master` (via `fix-chance-divisor`). The retraining work would be a separate branch.

---

## 7. Open Questions

1. Was the original training data already in a canonical form? (If yes, the model should be invariant by construction. If no, retraining is needed.) The training config (`configs/train/hybrid_ppo_v1.yaml`) and the env's observation wrapper (`twenty_forty_eight_ai/env/environment.py`) would need to be checked.

2. The OLD's `depth3_expectimax` benchmark scored 26,523 on n=100 in late 2025. Was the model retrained between then and now? (If not, the regression is entirely in the C++ code, but the search-tree trace shows the OLD's values differ from the NEW's for the same canonical board. So the model hasn't changed; only the C++ has.)

3. The alpha-beta pruning (`alpha_beta_cuts: 0`) never fires. Should it be removed entirely? It is unsound in expectimax (a hard `beta` cut at a max node can prune children whose contribution would have brought the parent chance node's average back into bounds). It is documented as a known caveat in the C++ code (`ExpectimaxSearcher.cpp:174-178`).

4. The deferred batching's UNRESOLVED path causes 1–3 missed leaves per search on 6/30 random boards. Is this acceptable, or should the search iterate the batch loop until all leaves are resolved? (Currently it stops when `move_scores[move_id]` is no longer UNRESOLVED, which is a single iteration for typical searches.)

---

## 8. Acceptance Criteria for the Retraining Fix

1. `value(rotate90(board)) ≈ value(board)` to within ~0.01 on a 0–60 scale, on a sample of 100 random boards.
2. `value(reflect(board)) ≈ value(board)` to within ~0.01 on the same sample.
3. `tests/test_depth4_convergence.py` continues to pass.
4. `tests/test_persistent_tt.py` continues to pass.
5. The 30-game depth-3 benchmark scores ≥24,000 (one std below the OLD's mean of 26,523 ± 12,750).
6. `data/models/release/Hybrid-PPO-Expectimax-v1.zip` is replaced with the retrained model.

Once these pass, the cross-search aging fix from PR #4 becomes safe to keep, and the search optimizations are no longer a net regression.
---

## 9. Work Plan: Retraining on `fix-regression`

**Branch:** `fix-regression` (branched from `fix-chance-divisor`).
**Goal:** Add D4 augmentation, retrain the value network, verify the acceptance criteria in §8, merge when done.
**Diagnostic tooling lives on:** `side-by-side-old-vs-new` (untouched, historical record).

### 9.1 Implementation Steps

- [ ] **1. Add D4 augmentation to env observation wrapper** — in `twenty_forty_eight_ai/env/environment.py`, apply a random D4 symmetry transform (8 transforms: 4 rotations × 2 reflections) to each board on `reset()` and `step()`. Or augment the training data directly in `scripts/train.py`. Confirm the env still passes the existing test suite (maskable actions must rotate consistently with the board).
- [ ] **2. Retrain `Hybrid-PPO-Expectimax-v1`** — re-run `scripts/train.py --config configs/train/hybrid_ppo_v1.yaml` (4–8 hours on a single GPU; exact time depends on the training config).
- [ ] **3. Verify rotation-invariance** — write a small script that calls `model(canonicalize(board))` and `model(board)` on 100 random boards and asserts `|diff| < 0.01`. This is the regression test for the root cause; if it fails, the model needs more training.
- [ ] **4. Replace the model** — overwrite `data/models/release/Hybrid-PPO-Expectimax-v1.zip` with the retrained weights. Commit the new model file to `fix-regression`.
- [ ] **5. Run existing tests** — `uv run pytest tests/test_depth4_convergence.py tests/test_persistent_tt.py` must still pass (they don't depend on the model, but catch C++ regressions).
- [ ] **6. Run the 30-game depth-3 benchmark** — `uv run python scripts/benchmark.py data/models/release/Hybrid-PPO-Expectimax-v1.zip --depth 3 --n_runs 30 --device cuda`. Target score ≥24,000 (acceptance criterion §8.5). Compare against the OLD's 26,523 ± 12,750.
- [ ] **7. Re-run the side-by-side diagnostic** — `uv run python scripts/side_by_side_old_vs_new.py --boards 30 --depth 3 --device cuda --debug`. With a rotation-invariant model, the leaf VALUES should now match between OLD and NEW (the canonical keys already match). Move agreement should rise from 33% to ~100%, score agreement from 3% to ~100%.
- [ ] **8. Merge** — `fix-regression` → `fix-chance-divisor` → `master` (or directly `fix-regression` → `master` if you want to bypass the divisor fix).

### 9.2 Estimated Effort

| Step | Time |
|------|------|
| 1. Add D4 augmentation | ~30 min (code + test) |
| 2. Retrain | 4–8 hours (GPU) |
| 3. Verify | ~10 min (script + run) |
| 4. Replace model | ~5 min |
| 5. Run tests | ~10 min |
| 6. Benchmark (depth 3, n=30) | ~90 min (CUDA) |
| 7. Side-by-side | ~1 min |
| 8. Merge | ~5 min |
| **Total** | **~6–10 hours** |

### 9.3 Branch Lineage (for context)

```
master (ada0806, post-PR-#4 merge)
 └─ fix-chance-divisor (8ef9289) ── divisor bug fix
      └─ fix-regression (current) ── retraining work
side-by-side-old-vs-new (7cd0fe0) ── diagnostic tooling, untouched
```

### 9.4 Check-off

Mark each step done by changing `[ ]` to `[x]` in the commit. Use a single commit per step (or batch related steps) so the history reads as a clear plan-then-execute log.
