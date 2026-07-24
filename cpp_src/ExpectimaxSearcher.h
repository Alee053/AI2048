#pragma once
#include "Fast2048.h"
#include "TranspositionTable.h"
#include "BoardEncoder.h"
#include <vector>
#include <map>
#include <functional>
#include <chrono>
#include <unordered_map>

using BatchEvalFunc = std::function<std::vector<float>(const std::vector<std::array<std::array<int, 4>, 4>>&)>;
using Board = std::array<std::array<int, 4>, 4>;
using LeafCache = std::unordered_map<uint64_t, float>;

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
};

class ExpectimaxSearcher {
public:
    explicit ExpectimaxSearcher(size_t target_batch_size = 32768);

    SearchStats find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

    void clear_tt() { transposition_table.clear(); }

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
    std::chrono::high_resolution_clock::time_point search_start;

    float chance_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                  std::vector<uint64_t>& batch_queue,
                                  const LeafCache& leaf_cache);
    float max_node_substitute(const Board& board, int depth, uint64_t board_hash,
                               std::vector<uint64_t>& batch_queue,
                               const LeafCache& leaf_cache,
                               float alpha, float beta);
};
