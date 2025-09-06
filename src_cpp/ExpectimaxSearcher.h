#pragma once

#include "Fast2048.h"
#include <vector>
#include <functional> // Required for std::function (the callback)

class ExpectimaxSearcher {
public:
    ExpectimaxSearcher();

    int find_best_move_with_eval(
        const std::array<std::array<int, 4>, 4>& board,
        int depth,
        const std::function<float(const std::array<std::array<int, 4>, 4>&)>& eval_func
    );

private:
    float chance_node(Fast2048& game, int depth, const std::function<float(const std::array<std::array<int, 4>, 4>&)>& eval_func);
    float max_node(Fast2048& game, int depth, const std::function<float(const std::array<std::array<int, 4>, 4>&)>& eval_func);
};
