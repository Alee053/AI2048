#include "ExpectimaxSearcher.h"
#include <iostream>
#include <vector>

std::vector<float> fake_eval(const std::vector<Board>& boards) {
    std::vector<float> results;
    for (const auto& b : boards) {
        float sum = 0;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                sum += b[r][c];
        results.push_back(sum);
    }
    return results;
}

int main() {
    ExpectimaxSearcher searcher(32768);
    Board board = {{
        {{1, 1, 3, 5}},
        {{0, 0, 0, 3}},
        {{1, 0, 0, 2}},
        {{0, 0, 0, 0}}
    }};

    std::cout << "Starting search for problematic board at depth 3...\n";
    auto stats = searcher.find_best_move(board, 3, fake_eval);
    std::cout << "best_move=" << stats.best_move << " nodes=" << stats.nodes_visited << "\n";

    return 0;
}
