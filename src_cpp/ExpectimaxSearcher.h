#pragma once

#include "Fast2048.h"
#include <vector>
#include <functional>
#include <map>

using Board = std::array<std::array<int, 4>, 4>;

using BatchEvalFunc = std::function<std::vector<float>(const std::vector<Board>&)>;


class ExpectimaxSearcher {
public:
    ExpectimaxSearcher();

    int find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

private:
    void gather_leaves(const Board& board, int depth, std::vector<Board>& leaves_queue, std::map<Board, bool>& visited);

    float max_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache);
    float chance_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache);

    Fast2048 game_instance;
};