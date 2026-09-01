# Historical Design Note: Deferred Batching for Neural-Expectimax

> Retained for design history. This proposal is superseded by the current v3
> implementation and is not part of the PRE-FREEZE paper protocol. References
> to alpha-beta pruning below describe this 2026-05-05 proposal, not the
> current behavior.

**Date:** 2026-05-05
**Status:** Historical / superseded
**Branch:** `searcher-improvements` (historical)
**Author:** AI Assistant (brainstorming session)

---

## 1. Problem Statement

At Expectimax depth 4, the current two-phase architecture (gather all leaves per root move, then batch-evaluate in 512-board chunks) produces ~200 tiny Python crossings per turn. Each crossing incurs pybind11 IPC overhead and launches undersized GPU kernels, leaving an RTX 3070 Ti ~99% idle. The result: **40–60 seconds per move** despite the search tree only containing ~100k unique nodes.

The root cause: control flow is inverted. Python drives the search; C++ is called repeatedly for small batches. We need to invert this so C++ drives the search and yields to Python only for massive, GPU-saturating batches.

---

## 2. Goals

| Goal | Metric |
|------|--------|
| Reduce Python crossings per turn | From ~200 to ≤4 |
| Increase average batch size to GPU | From ~512 to 16,000–32,000 |
| Maintain search correctness | Identical move decisions vs. old code |
| Leverage persistent TT | Leaf evaluations survive across turns |
| Keep architecture simple | No threading, no streaming, no async |

---

## 3. Architecture

### 3.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│  C++ ExpectimaxSearcher                                     │
│                                                             │
│  Step 0: Pre-Order CNN Evaluation (4 boards)                │
│          → establish move ordering for root moves           │
│                                                             │
│  Step 1: Multi-Pass Control Loop                            │
│          while (any_unresolved) {                           │
│            Pass A: Discovery DFS                            │
│              - TT hit  → return cached score                │
│              - depth 0 → canonicalize, queue, return INF    │
│              - child INF → propagate INF upward             │
│              - queue full → stop exploring, yield           │
│                                                             │
│            Pass B: Batch GPU Inference (if queue not empty) │
│              - deduplicate canonical uint64_t queue         │
│              - batch_eval_func(canonical_boards)            │
│              - store results in TT at depth 0               │
│          }                                                  │
│                                                             │
│  Step 2: Alpha-Beta Pruning (final pass only)               │
│          → extract best_move and move_scores                │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1 Python crossing per pass
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Python / PyTorch GPU                                       │
│  batch_eval_func(boards: List[np.ndarray]) → List[float]    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Why This Works

- **C++ traversal is free.** A 100k-node tree walk with TT hits takes <1ms. Re-traversing the tree 3–4 times is negligible compared to a single Python crossing.
- **GPU saturation.** Batches of 16k–32k tensors fully utilize CUDA cores and amortize kernel launch overhead.
- **Persistent TT compounds.** Leaves evaluated in move 50, pass 2 are cached for move 51. Over a full game, the number of *new* leaves per turn shrinks dramatically.
- **Correctness by construction.** No intermediate pruning. All nodes resolve to exact expectimax values.

---

## 4. Key Design Decisions

### 4.1 UNRESOLVED Sentinel

```cpp
constexpr float UNRESOLVED = -std::numeric_limits<float>::infinity();
```

**Rationale:**  
- Collision-proof. Real expectimax scores are bounded well within `[-1000, +1000]`.
- Detected via `std::isinf(score)` rather than `==` comparison.
- Cannot be confused with any valid score, reward, or value estimate.

### 4.2 8-Way Canonical Board Representation

**Rationale:**  
2048 has 8 symmetries (4 rotations × 2 reflections). Two boards that are rotations or mirrors of each other have identical game-theoretic value. Canonicalizing to a single representative before TT lookup and batch queue insertion:
- **8× fewer TT entries** for symmetric states
- **Instant deduplication** in the batch queue (just sort `uint64_t`)
- **Fewer GPU evaluations** per turn

**Implementation:**  
The board is stored as a 64-bit integer: 16 tiles × 4 bits each. For a given board, generate all 8 symmetry transforms, compute their 64-bit encodings, and pick the minimum as the canonical form.

> **Note on TT keys:** The canonical packed board *replaces* Zobrist hashing for TT keys. A 64-bit packed board is a perfect hash for 4×4 2048 (16 tiles × 4 bits = 64 bits). Zobrist hashing may still be used elsewhere (e.g., incremental hash updates in `RandomUtil`) but is no longer needed for TT lookups.

```cpp
uint64_t canonicalize(uint64_t board);
```

The `batch_queue` stores **only** `uint64_t` canonical representations. Deduplication is:
```cpp
std::sort(queue.begin(), queue.end());
queue.erase(std::unique(queue.begin(), queue.end()), queue.end());
```

### 4.3 Dynamic Target Batch Size

The `TARGET_BATCH_SIZE` is a **constructor parameter** passed from Python:

```cpp
ExpectimaxSearcher(size_t target_batch_size = 32768);
```

**Rationale:**
- Different GPUs have different optimal batch sizes (RTX 3070 Ti: ~32k, T4: ~16k, CPU: ~1k).
- Tuning without recompilation.
- Default of 32768 is chosen for modern desktop GPUs.

### 4.4 No Intermediate Pruning

Alpha-beta pruning is **only enabled in the final pass** when all nodes are resolved.

**Rationale:**
- Expectimax chance nodes require **all** children to compute an exact expectation.
- With `UNRESOLVED` children, bounds tracking becomes dangerously complex.
- The RTX 3070 Ti can easily absorb the slightly larger batches from an unpruned intermediate pass.
- Correctness by construction > marginal performance gain.

---

## 5. Component Breakdown

### 5.1 TranspositionTable (existing, extended)

No changes needed. The existing `TranspositionTable` already supports:
- `(key, depth, type)` tuple lookup
- Depth-preferred replacement
- Persistent across turns

It will now also store depth-0 leaf evaluations (both `MAX` and `CHANCE` types).

### 5.2 BoardEncoder (new)

Responsible for:
- Converting `Board` (4×4 array) to 64-bit packed integer
- Computing 8-way canonical representation
- Converting canonical 64-bit back to `Board` for Python callback

```cpp
class BoardEncoder {
public:
    static uint64_t pack(const Board& board);
    static Board unpack(uint64_t packed);
    static uint64_t canonicalize(uint64_t packed);
    static uint64_t canonicalize(const Board& board);
};
```

### 5.3 ExpectimaxSearcher (refactored)

**Removed:**
- `gather_leaves()` — no longer needed; search IS the gather
- `leaf_cache` parameter — replaced by persistent TT
- Per-root-move loop — unified into multi-pass control loop
- `BATCH_SIZE` constant — replaced by `target_batch_size_`

**Modified signatures:**
```cpp
float max_node_substitute(const Board& board, int depth, uint64_t board_hash,
                          std::vector<uint64_t>& batch_queue,
                          float alpha, float beta);

float chance_node_substitute(const Board& board, int depth, uint64_t board_hash,
                             std::vector<uint64_t>& batch_queue,
                             float alpha, float beta);
```

**New constructor:**
```cpp
explicit ExpectimaxSearcher(size_t target_batch_size = 32768);
```

**New method:**
```cpp
void resolve_batch_queue(const BatchEvalFunc& batch_eval_func);
```

### 5.4 Python Wrapper (minor update)

Expose `target_batch_size` parameter in Python:

```python
class ExpectimaxSearcher:
    def __init__(self, target_batch_size: int = 32768) -> None:
        self._impl = _impl.ExpectimaxSearcher(target_batch_size)
```

---

## 6. Control Flow Detail

### 6.1 Discovery Pass (DFS)

```cpp
float max_node_substitute(board, depth, hash, queue, alpha, beta) {
    // 1. Canonicalize and TT lookup
    uint64_t canon = BoardEncoder::canonicalize(board);
    float cached;
    if (tt.probe(canon, depth, MAX, cached)) return cached;

    // 2. Leaf node
    if (depth == 0) {
        queue.push_back(canon);
        return UNRESOLVED;
    }

    // 3. Check queue capacity
    if (queue.size() >= target_batch_size_) {
        return UNRESOLVED;
    }

    // 4. Explore children
    float best = -INF;
    bool any_unresolved = false;
    for (move in 0..3) {
        auto [child_board, child_hash] = apply_move(board, hash, move);
        float child_score = chance_node_substitute(child_board, depth, child_hash, queue, alpha, beta);
        if (std::isinf(child_score) && child_score < 0) {
            any_unresolved = true;
            continue;
        }
        best = std::max(best, child_score);
        // Alpha-beta: only active when all children are resolved (final pass)
        if (!any_unresolved && best >= beta) break;
        alpha = std::max(alpha, best);
    }

    if (any_unresolved) return UNRESOLVED;

    // 5. Store and return
    tt.store(canon, depth, MAX, best);
    return best;
}
```

### 6.2 Chance Node

```cpp
float chance_node_substitute(board, depth, hash, queue, alpha, beta) {
    uint64_t canon = BoardEncoder::canonicalize(board);
    float cached;
    if (tt.probe(canon, depth, CHANCE, cached)) return cached;

    auto empty_cells = find_empty(board);
    if (empty_cells.empty()) {
        return max_node_substitute(board, depth - 1, hash, queue, alpha, beta);
    }

    float total = 0;
    bool any_unresolved = false;
    for (cell in empty_cells) {
        if (queue.size() >= target_batch_size_) {
            return UNRESOLVED;
        }
        auto [child_board, child_hash] = spawn_tile(board, hash, cell, value);
        float child_score = max_node_substitute(child_board, depth - 1, child_hash, queue, alpha, beta);
        if (std::isinf(child_score) && child_score < 0) {
            any_unresolved = true;
            continue;
        }
        total += probability * child_score;
    }

    if (any_unresolved) return UNRESOLVED;

    float result = total / (2 * empty_cells.size());
    tt.store(canon, depth, CHANCE, result);
    return result;
}
```

### 6.3 Multi-Pass Loop in `find_best_move`

```cpp
SearchStats find_best_move(board, depth, batch_eval_func) {
    // Step 0: Pre-order CNN evaluation for move ordering
    auto root_moves = generate_and_order_root_moves(board, batch_eval_func);

    // Step 1: Multi-pass loop
    std::vector<uint64_t> batch_queue;
    batch_queue.reserve(target_batch_size_);

    float move_scores[4] = {UNRESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED};
    int resolved_count = 0;

    while (resolved_count < root_moves.size()) {
        batch_queue.clear();
        resolved_count = 0;

        for (auto& rm : root_moves) {
            if (!std::isinf(move_scores[rm.move_id])) {
                resolved_count++;
                continue;
            }

            float score = chance_node_substitute(
                rm.post_board, depth, rm.post_hash,
                batch_queue, global_alpha, INF
            );

            if (!std::isinf(score)) {
                move_scores[rm.move_id] = rm.immediate_reward + score;
                resolved_count++;
                global_alpha = std::max(global_alpha, move_scores[rm.move_id]);
            }
        }

        if (resolved_count < root_moves.size() && !batch_queue.empty()) {
            // Deduplicate
            std::sort(batch_queue.begin(), batch_queue.end());
            auto last = std::unique(batch_queue.begin(), batch_queue.end());
            batch_queue.erase(last, batch_queue.end());

            // Convert uint64_t → Board for Python callback
            std::vector<Board> boards_for_python;
            for (auto canon : batch_queue) {
                boards_for_python.push_back(BoardEncoder::unpack(canon));
            }

            // Single Python crossing
            auto values = batch_eval_func(boards_for_python);

            // Store in TT at depth 0
            for (size_t i = 0; i < batch_queue.size(); ++i) {
                tt.store(batch_queue[i], 0, MAX, values[i]);
                tt.store(batch_queue[i], 0, CHANCE, values[i]);
            }
        }
    }

    // Extract best move from move_scores
    // ...
}
```

---

## 7. Error Handling & Edge Cases

| Scenario | Behavior |
|----------|----------|
| `batch_queue` exceeds `target_batch_size_` mid-traversal | Stop exploring, return `UNRESOLVED` from current node. Queue will be evaluated in next step. |
| All root moves resolve in first pass | No Python crossing needed. Fast path for warm TT. |
| `batch_eval_func` returns fewer values than boards | Assert / throw. This is a programming error in the Python callback. |
| Board with no valid moves | `max_node_substitute` returns 0.0 (terminal state). Stored in TT. |
| Queue empty but some moves unresolved | This should not happen. If it does, assert — indicates a logic bug. |
| Canonicalization collision | Impossible for 4×4 2048 boards. 64 bits uniquely encode all states. |

---

## 8. Testing Strategy

### 8.1 Correctness Tests
- **Deterministic evaluator:** Use a fake `batch_eval_func` that returns `sum(board)`. Verify that move decisions are identical to the old two-phase implementation on 100 random boards.
- **UNRESOLVED sentinel isolation:** Confirm no `inf` values leak into final `move_scores` or `best_move`.
- **Canonicalization round-trip:** `unpack(pack(board)) == board` for all 8 symmetries.

### 8.2 Performance Tests
- **Batch size distribution:** Log the size of each `batch_queue` before deduplication and after. Assert average post-dedup batch ≥ 8000 for depth 4.
- **Python crossing count:** Assert ≤ 4 crossings per turn at depth 4.
- **TT hit rate progression:** Run a full game. Assert hit rate increases monotonically across moves.

### 8.3 Edge Case Tests
- **Empty queue fast path:** Search a board where the TT is already fully warm. Assert zero Python crossings.
- **Maximum queue size:** Force `target_batch_size_ = 1`. Verify the search still completes (many tiny batches).
- **All moves invalid:** Board with no valid moves. Assert search returns immediately with score 0.

---

## 9. Migration Path

1. **Phase 1:** Implement `BoardEncoder` with canonicalization.
2. **Phase 2:** Refactor `ExpectimaxSearcher` with new signatures and multi-pass loop.
3. **Phase 3:** Update pybind11 bindings and Python wrapper.
4. **Phase 4:** Build, run correctness tests against old implementation.
5. **Phase 5:** Run performance benchmark, tune `target_batch_size`.
6. **Phase 6:** Remove old `gather_leaves`, `leaf_cache`, and per-root-move loop code.

---

## 10. Success Criteria

| Criterion | Target |
|-----------|--------|
| Depth-4 move time | < 5 seconds (down from 40–60s) |
| Python crossings per turn | ≤ 4 |
| Average batch size | ≥ 8,000 boards |
| Move decision parity | 100% match with old implementation on same model |
| Test coverage | All new code paths covered by tests |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Canonical board** | The minimum 64-bit representation among all 8 symmetry transforms of a board. |
| **Deferred batching** | Delaying neural network evaluation until a large batch of leaves has been discovered, rather than evaluating immediately. |
| **Discovery pass** | A tree traversal that identifies missing leaf evaluations without computing them. |
| **UNRESOLVED** | Sentinel value (`-inf`) indicating a node cannot be computed yet because at least one child lacks a cached evaluation. |
| **Target batch size** | The maximum number of leaf boards to accumulate before triggering a GPU inference batch. |
