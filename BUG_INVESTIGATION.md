# ExpectimaxSearcher Infinite Loop Bug - Investigation Log

## The Problem

The `ExpectimaxSearcher::find_best_move()` enters an infinite loop on certain board states (e.g., `[[0,1,5,1],[0,1,2,6],[0,0,0,1],[0,0,0,1]]` at depth 2). The searcher visits ~55 million nodes per second but makes zero forward progress — resolved moves stays at 2/3 forever, `batch_queue` is always size 1 after pass 3, and the TT saturates at ~1450 entries.

## Reproduction

```bash
cd /home/ale/.superset/worktrees/AI2048/searcher-improvements/cpp_src
g++ -std=c++17 -O2 -I/usr/include/python3.12 -I/home/ale/.superset/worktrees/AI2048/searcher-improvements/.venv/lib/python3.12/site-packages/pybind11/include test_slow.cpp ExpectimaxSearcher.cpp Fast2048.cpp BoardEncoder.cpp RandomUtil.cpp -o test_slow
./test_slow  # times out after 30s
```

The test board:
```cpp
Board board = {{
    {{0, 1, 5, 1}},
    {{0, 1, 2, 6}},
    {{0, 0, 0, 1}},
    {{0, 0, 0, 1}}
}};
```

Uses `fake_eval` which returns `sum(board)` for all boards.

## Symptom Timeline

### Pass 1 (GOOD)
```
queue=1326 nodes=4004 tt_size=0 resolved=0/3 collisions=0
```
- 1326 leaves discovered in one DFS pass
- TT is empty (nothing stored yet)

### Pass 2 (PARTIAL)
```
queue=2 nodes=1652 tt_size=2765 resolved=1/3 collisions=11
```
- 1 move resolved (1/3)
- 2 new leaves discovered (they're not in TT yet — collision with existing entries)

### Pass 3+ (BROKEN)
```
queue=1 nodes=99 tt_size=2768 resolved=2/3 collisions=13
queue=1 nodes=53 tt_size=2768 resolved=2/3 collisions=14
... (infinite, queue stays at 1, nodes increase by 53 per pass)
```
- 2/3 moves resolved — the third never resolves
- Queue always has exactly 1 board per pass
- TT has 2768 entries (~1326 leaves stored twice — once MAX, once CHANCE)
- Collisions increase slowly, all from same two keys cycling

## Key Evidence Observed

### DEDUP Before/After (from logs)
```
[DEDUP] before: 3742 entries  after: 1326 entries  (pass 1 — many duplicates from different search paths)
[DEDUP] before: 3 entries     after: 2 entries     (pass 2 — 3 leaves, 2 unique)
[DEDUP] before: 2 entries     after: 1 entries     (pass 3+ — same 2 keys cycling)
```

Pass 3+ always has the same 2 keys in the queue: `281617081893201` and `587140587065617`. These two keys ping-pong — one gets stored, the other wasn't stored because it collided, next pass the other one...

### TT_MISS Log (key evidence)
All misses show `entry_depth=0` and `entry_key=0` (empty bucket) — which means at the time of probing, the TT bucket is **empty**. But we stored 2768 entries. The only way an empty bucket gets probed is if the **same canonical key** maps to **different hash indices** in store vs probe.

### Collision Pattern
```
[COLLISION] idx=9754243 old_key=281617081893201 new_key=587140587065617 old_type=0 new_type=0
```
Two very different keys colliding at the same bucket index `9754243`. This is a **genuine hash collision** (different keys same bucket), not the type-collision issue from earlier.

### PROBE_MISS on depth=1 nodes
```
[PROBE_MISS] depth=1 canon=8592429649
[PROBE_MISS] depth=1 canon=8592433745
...
```
Depth-1 probe misses — but we stored leaves at depth=255, and 255 >= 1. So why do these miss?

### batch_queue size after deduplication
Pass 1: 3742 → 1326 (many duplicates)
Pass 2: 3 → 2 (2 unique new boards)
Pass 3+: 2 → 1 (same 2 keys, ping-ponging)

## What Was Tested / Fixes Applied

### 1. TT Hash Index Formula (WRONG APPROACH)
**Hypothesis**: Adding type (0/1) after `hash_key()` creates adjacent bucket indices for MAX/CHANCE.

**Fix applied**:
```cpp
// Before
uint32_t idx = (hash_key(key) + static_cast<uint64_t>(type)) & TT_MASK;

// After
uint32_t idx = static_cast<uint32_t>(hash_key(key ^ (static_cast<uint64_t>(type) << 1)) & TT_MASK);
```

**Result**: Reduced initial collisions but did NOT fix the infinite loop. The remaining collisions were type=0 vs type=0 (genuine hash collisions, not type collisions).

### 2. TT Replacement Policy — Same-Key Overwrites
**Hypothesis**: The store replacement logic was incorrectly rejecting same-key entries.

**Fix applied**:
```cpp
// Before: entry.key == key && depth >= entry.depth → replace
// After:  entry.key == key && entry.type == type → always replace
```

**Result**: `same_key_overwrite_count` increased (confirming the old policy was wrong), but infinite loop persisted.

### 3. LEAF_DEPTH = 255 (Secondary Fix)
**Hypothesis**: Leaves stored at depth=0 can't satisfy probes from depth > 0.

**Fix applied**:
```cpp
constexpr uint8_t LEAF_DEPTH = 255;
for (size_t i = 0; i < batch_queue.size(); ++i) {
    transposition_table.store(batch_queue[i], LEAF_DEPTH, NodeType::MAX, values[i]);
    transposition_table.store(batch_queue[i], LEAF_DEPTH, NodeType::CHANCE, values[i]);
}
```

**Result**: Used but not independently tested (combined with other changes). The TT still can't help because the two problematic boards keep colliding.

### 4. Removed `continue` in chance_node_substitute
**Hypothesis**: When a child returns UNRESOLVED, the loop should keep iterating to discover siblings, not skip them.

**Fix applied**:
```cpp
// Before
if (std::isinf(val_2) && val_2 < 0) { any_unresolved = true; continue; }

// After
if (std::isinf(val_2) && val_2 < 0) { any_unresolved = true; }
else { total_value += 0.9f * val_2; }
```

**Result**: Applied but didn't help — the infinite loop isn't about sibling discovery.

### 5. Removed `continue` in max_node_substitute
Same pattern as above.

### 6. Batch Queue Size Check Before Recursion
**Hypothesis**: Should yield when queue is nearly full, not when full.

**Fix applied**: Added queue size check at the START of each loop iteration, before recursing.

**Result**: Applied but didn't help the infinite loop.

## Current State of Code

### TranspositionTable.h (modified)
- `hash_key(key, type)` — type XOR'd before hash diffusion
- Simplified replacement: same key+type always overwrites
- TT_MISS logging added for debugging

### ExpectimaxSearcher.cpp (modified)
- `continue` removed from both `chance_node_substitute` and `max_node_substitute`
- Queue size checks before each child recursion
- `LEAF_DEPTH = 255` for storing batch results
- Extensive debug logging: `[ENQUEUE]`, `[PASS]`, `[DEDUP]`, `[STORE]`, `[PY_RET]`, `[PROBE_MISS]`, `[TT_MISS]`

## The Real Bug (Suspected)

The infinite loop is caused by **hash collisions on the two specific boards** `281617081893201` and `587140587065617`. These two keys keep colliding at index `9754243` on every pass:

1. Pass N: board A is enqueued, stored (overwrites previous content at idx 9754243)
2. Pass N+1: board B is enqueued, probes idx 9754243 — finds board A's key (wrong), misses
3. Board B is enqueued, stored (overwrites board A's entry with board B's value)
4. Pass N+2: board A is enqueued, probes idx 9754243 — finds board B's key (wrong), misses
5. ... repeat forever

Both boards are depth-0 leaves (or near-leaves) that should have been resolved in earlier passes. The TT is 2768 entries (~1326 unique boards × 2 types) but the hash function keeps mapping these two specific boards to the same bucket that other boards also want.

### Why Pass 1 Works (1326 leaves)
Pass 1 discovers 1326 leaves because the TT is empty — no collisions yet. All 1326 are stored and deduplicate down to ~726 unique boards.

### Why Pass 2 Only Adds 2
After storing 1326 leaves at depth 255, the remaining tree is partially explored. Only 2 new boards are discovered that don't collide.

### Why Pass 3+ Never Resolves the 3rd Move
The 3rd root move's subtree contains one of the two colliding boards. Every time that board is visited, it misses the TT, gets enqueued, and overwrites the other board's entry. The other board then misses on the next pass. This cycle prevents the subtree from ever completing.

## Diagnostic Evidence (Actual Log Output)

```
[TT_MISS] key=8592360033 depth=2 type=1 entry_depth=0 entry_key=0 entry_type=0 idx=6861256
[TT_MISS] key=8592429649 depth=1 type=0 entry_depth=0 entry_key=0 entry_type=0 idx=9767767
```

Entry_key=0 means the bucket is EMPTY. We have 2768 entries stored but these buckets are empty. This can only happen if:
- `hash_key(key, type) & TT_MASK` produces different indices in store vs probe
- The store is hitting a different bucket than the probe

But `hash_key` is a pure function — same inputs should give same outputs. Unless there's uninitialized memory, memory corruption, or the TT pointer is being corrupted...

Wait — actually, looking more carefully at the TT_MISS output: `entry_depth=0` and `entry_key=0` consistently. This pattern suggests ALL the buckets being probed in passes 3+ are empty, not that there's key mismatch. If the entry_key were the other colliding key, we'd see that. But it's always 0.

This means the two cycling keys (`281617081893201` and `587140587065617`) have already overwritten ALL the entries in the paths being probed. After ~1450 passes, every bucket in the "relevant portion" of the TT has been claimed by these two keys or the 1326 initial leaves that have been pushed out.

The real fix might need to be either:
1. Increase TT size to reduce collision rate
2. Use a better hash function (not splitmix64, which has known weaknesses for certain input patterns)
3. Add a proper eviction policy instead of always-overwrite
4. Investigate why `hash_key` seems to produce same bucket for two very different keys at idx=9754243

## Files Involved

- `cpp_src/ExpectimaxSearcher.cpp` — main searcher (modified with debug logging)
- `cpp_src/TranspositionTable.h` — TT implementation (modified with hash fix and debug)
- `cpp_src/test_slow.cpp` — standalone C++ test with fake_eval
- `cpp_src/BoardEncoder.cpp/h` — board canonicalization
- `cpp_src/Fast2048.cpp/h` — game logic
- `tests/test_persistent_tt.py` — existing TT tests (all pass)

## Next Steps (Recommended)

1. **Verify hash consistency**: Add a test that calls `hash_key(key, type)` and `hash_key(key, type)` twice on the same inputs and confirms the same output
2. **Log the actual hash index** for the two colliding keys to confirm they're hitting the same bucket
3. **Check for memory corruption**: Run with valgrind or AddressSanitizer
4. **Consider ccache': Use a proper collision-aware eviction policy (e.g., always keep the higher-depth entry)
5. **Alternative**: Add a "poisoned" list of known-colliding boards that bypass the TT entirely