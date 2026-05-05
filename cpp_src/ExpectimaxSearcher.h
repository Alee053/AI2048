#pragma once
#include "Fast2048.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
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
};

struct BoardHash {
    size_t operator()(const Board& b) const {
        size_t h = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                h = h * 31 + std::hash<int>{}(b[i][j]);
        return h;
    }
};

class ExpectimaxSearcher {
public:
    ExpectimaxSearcher();

    SearchStats find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

private:
    static constexpr size_t BATCH_SIZE = 512;
    Fast2048 game_instance;

    // Key: uint64_t Zobrist hash of the board
    std::unordered_map<uint64_t, float> transposition_table;

    // Counters
    size_t tt_lookups = 0;
    size_t tt_hits = 0;
    size_t batches_eval = 0;
    std::chrono::high_resolution_clock::time_point search_start;

    void gather_leaves(const Board& board, int depth, uint64_t board_hash,
                       std::vector<Board>& leaves_queue,
                       std::map<int, std::unordered_set<Board, BoardHash>>& visited);

    float chance_node_substitute(const Board& board, int depth,
                                 const std::unordered_map<Board, float, BoardHash>& leaf_cache,
                                 float alpha, float beta);
    float max_node_substitute(const Board& board, int depth, uint64_t board_hash,
                              const std::unordered_map<Board, float, BoardHash>& leaf_cache,
                              float alpha, float beta);
};
