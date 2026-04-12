#pragma once
#include "Fast2048.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <functional>

using BatchEvalFunc = std::function<std::vector<float>(const std::vector<std::array<std::array<int, 4>, 4>>&)>;
using Board = std::array<std::array<int, 4>, 4>;

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

    int find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

private:
    static constexpr size_t BATCH_SIZE = 512;
    Fast2048 game_instance;

    // Key: uint64_t Zobrist hash of the board
    std::unordered_map<uint64_t, float> transposition_table;

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
