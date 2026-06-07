#pragma once
#include "Fast2048.h"
#include "TranspositionTable.h"
#include "BoardEncoder.h"
#include <vector>
#include <map>
#include <functional>
#include <chrono>

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
    std::chrono::high_resolution_clock::time_point search_start;

    float chance_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                 std::vector<uint64_t>& batch_queue);
    float max_node_substitute(const Board& board, int depth, uint64_t board_hash,
                              std::vector<uint64_t>& batch_queue,
                              float alpha, float beta);
};