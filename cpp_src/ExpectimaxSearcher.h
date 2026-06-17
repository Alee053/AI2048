#pragma once
#include "Fast2048.h"
#include "TranspositionTable.h"
#include "BoardEncoder.h"
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <fstream>

using BatchEvalFunc = std::function<std::vector<float>(const std::vector<std::array<std::array<int, 4>, 4>>&)>;
using Board = std::array<std::array<int, 4>, 4>;

struct SearchStats {
    int best_move;
    double think_ms;
    size_t nodes_visited;
    size_t batches_eval;
    float move_scores[4];
    size_t tt_size;
    size_t tt_lookups;
    size_t tt_hits;
    size_t tt_collisions;
    size_t tt_same_key_overwrites;
    int moves_resolved;
    int moves_unresolved;
    int cap_hits;
    // Diagnostics for the chance-divisor fix and the alpha-beta correctness investigation
    // (see the alpha_beta_cuts caveat in ExpectimaxSearcher.cpp — at a max node, the
    //  `max_value >= beta` cut can prune children whose contribution would have brought
    //  the parent chance node's average back into bounds).
    size_t alpha_beta_cuts = 0;
    size_t chance_nodes_evaluated = 0;
    size_t max_nodes_evaluated = 0;
    double chance_value_sum = 0.0;   // sum of chance-node return values (sanity check the divisor)
    size_t chance_value_count = 0;   // number of chance-node returns
    // Diagnostic: unique leaves evaluated (canonical packed-board key count).
    // Lets us compare the OLD's "all reachable leaves" with the NEW's "leaves
    // the search actually evaluated" without dumping every individual leaf.
    size_t unique_leaves_evaluated = 0;
};

class ExpectimaxSearcher {
public:
    explicit ExpectimaxSearcher(size_t target_batch_size = 32768);

    SearchStats find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

    void clear_tt() { transposition_table.clear(); }

    // Diagnostic: enable search-tree trace logging. When enabled, every
    // chance_node and max_node call writes one line to the trace file:
    //   "kind=<chance|max> depth=<N> board=0x<hex> value=<f> src=<computed|tt_hit|tt_miss|unresolved>\n"
    // Use to compare against a pure-Python reimplementation of the OLD
    // algorithm to pinpoint where the aggregation diverges.
    void set_trace_log(const std::string& path) {
        trace_log_.open(path, std::ios::out | std::ios::trunc);
        trace_enabled_ = trace_log_.is_open();
    }
    void close_trace_log() {
        if (trace_log_.is_open()) trace_log_.close();
        trace_enabled_ = false;
    }
    bool trace_enabled() const { return trace_enabled_; }

    // Diagnostic: dump the (canonical_key, value) pairs of all unique leaves the
    // last find_best_move evaluated. Keys are 64-bit packed canonical boards
    // (BoardEncoder::canonicalize output); values are the model outputs.
    // One line per leaf, format: "<hex_key> <value>\n". Enables direct comparison
    // with the OLD's gather_leaves() output (after canonicalizing each board).
    // Resets internal state, so call AFTER find_best_move if you want this run's
    // data; the next find_best_move will start a fresh capture.
    std::string dump_leaves() const;

    // Diagnostic: number of unique leaves the last find_best_move evaluated.
    // Same number as dump_leaves().size() but cheap to query.
    size_t last_unique_leaves() const { return leaves_.size(); }

private:
    static constexpr float UNRESOLVED = -std::numeric_limits<float>::infinity();

    Fast2048 game_instance;
    TranspositionTable transposition_table;
    size_t target_batch_size_;

    // Counters (reset every find_best_move)
    size_t tt_lookups = 0;
    size_t tt_hits = 0;
    size_t batches_eval = 0;
    size_t nodes_visited = 0;
    size_t alpha_beta_cuts_ = 0;
    size_t chance_nodes_evaluated_ = 0;
    size_t max_nodes_evaluated_ = 0;
    double chance_value_sum_ = 0.0;
    size_t chance_value_count_ = 0;
    // Captured leaves (canonical key -> value) for the current/last find_best_move.
    // Reset at the start of each find_best_move.
    std::unordered_map<uint64_t, float> leaves_;
    std::chrono::high_resolution_clock::time_point search_start;

    // Diagnostic: search-tree trace logging. Off by default. Open with
    // set_trace_log(path), close with close_trace_log(). Lines are written
    // for every chance_node and max_node call.
    std::ofstream trace_log_;
    bool trace_enabled_ = false;

    float chance_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                 std::vector<uint64_t>& batch_queue);
    float max_node_substitute(const Board& board, int depth, uint64_t board_hash,
                              std::vector<uint64_t>& batch_queue,
                              float alpha, float beta);
};