# C++ Searcher Development Guidelines

This document outlines the architectural requirements and pitfalls discovered during the development of the Expectimax Searcher for 2048.

## Transposition Table (TT) Architecture

The TT is critical for search progress in the multi-pass deferred-batching model.

1.  **Associativity**: Always use at least **4-way associativity**.
    *   **Rationale**: 16-byte `TTEntry` structs allow a 4-way `TTBucket` to fit in exactly 64 bytes (one L1 cache line). This provides high collision resistance without extra cache misses.
    *   **Pitfall**: 1-way or 2-way tables with "always overwrite" policies can lead to infinite loops if two "hot" keys collide, as the searcher will lose subtree progress on every pass.

2.  **Replacement Policy**: Use **Depth-Preferred Replacement**.
    *   **Rule**: Always keep the entry with the higher `depth` value. Deeper entries represent more computational work and are more valuable.
    *   **Tie-breaking**: Use a cyclical index (`collision_count % 4`) to decide which entry to evict if multiple entries have the same minimum depth. This prevents stable oscillation cycles.

3.  **Leaf Depth**: Store batch-evaluated leaves with **Depth 0**.
    *   **Rationale**: Leaves are cheap to re-evaluate (one CNN pass) compared to internal nodes which may represent millions of visited states. Never let a leaf displace an internal search node.

4.  **Key Canonicalization**: Always use `BoardEncoder::canonicalize()` before probing or storing in the TT.
    *   This reduces the state space by 8x (handling symmetries) but increases "hotness" per bucket.

## Search Loop Design

1.  **Multi-Pass Failsafe**: Any loop that depends on the TT for termination (like the multi-pass batching loop) **MUST** have a hard iteration limit.
    *   Example: `while (unresolved && pass < 100)`.
    *   This prevents a logic bug or unexpected hash collision pattern from hanging the entire application.

2.  **Silent Hot Paths**: No logging (`std::cerr`, `printf`) is allowed in `max_node_substitute` or `chance_node_substitute`.
    *   **Pitfall**: High-frequency searchers can swamp the OS pipe buffer, causing the process to block on write. This looks like a hang in the parent process.

3.  **Memory Management**: The TT is a single large allocation (`~256 MiB` for 2^24 entries). Avoid per-move or per-pass allocations.
