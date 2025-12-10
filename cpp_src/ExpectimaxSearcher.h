#pragma once
#include "Fast2048.h"
#include <vector>
#include <map>
#include <functional>

using BatchEvalFunc = std::function<std::vector<float>(const std::vector<std::array<std::array<int, 4>, 4>>&)>;
using Board = std::array<std::array<int, 4>, 4>;

class ExpectimaxSearcher {
public:
    ExpectimaxSearcher();

    int find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

private:
    Fast2048 game_instance;

    // Key: Board State + Depth, Value: Score
    // We include depth because a board at depth 1 has a different value than at depth 3
    struct TranspositionKey {
        Board board;
        int depth;

        bool operator<(const TranspositionKey& other) const {
            if (depth != other.depth) return depth < other.depth;
            return board < other.board;
        }
    };

    std::map<TranspositionKey, float> transposition_table;

    void gather_leaves(const Board& board, int depth, std::vector<Board>& leaves_queue, std::map<Board, bool>& visited);

    float chance_node_substitute(const Board& board, int depth, const std::map<Board, float>& leaf_cache);
    float max_node_substitute(const Board& board, int depth, const std::map<Board, float>& leaf_cache);
};