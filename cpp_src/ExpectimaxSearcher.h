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

struct TranspositionKeyHash {
    size_t operator()(const std::pair<Board, int>& k) const {
        BoardHash bh;
        return bh(k.first) * 31 + std::hash<int>{}(k.second);
    }
};

class ExpectimaxSearcher {
public:
    ExpectimaxSearcher();

    int find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

private:
    Fast2048 game_instance;

    // Key: (Board, depth) pair. Depth is included because a board at depth 1
    // has a different value than at depth 3 (more future moves possible).
    std::unordered_map<std::pair<Board, int>, float, TranspositionKeyHash> transposition_table;

    void gather_leaves(const Board& board, int depth, std::vector<Board>& leaves_queue,
                       std::map<int, std::unordered_set<Board, BoardHash>>& visited);

    float chance_node_substitute(const Board& board, int depth,
                                 const std::unordered_map<Board, float, BoardHash>& leaf_cache);
    float max_node_substitute(const Board& board, int depth,
                              const std::unordered_map<Board, float, BoardHash>& leaf_cache);
};
