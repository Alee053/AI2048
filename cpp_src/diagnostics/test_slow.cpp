#include <iostream>
#include <chrono>
#include <set>
#include "../ExpectimaxSearcher.h"
#include "../BoardEncoder.h"

std::vector<float> fake_eval(const std::vector<Board>& boards) {
    std::vector<float> result;
    for (const auto& b : boards) {
        float sum = 0;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                sum += b[r][c];
        result.push_back(sum);
    }
    return result;
}

int main() {
    Board board = {{
        {{0, 1, 5, 1}},
        {{0, 1, 2, 6}},
        {{0, 0, 0, 1}},
        {{0, 0, 0, 1}}
    }};

    ExpectimaxSearcher searcher;
    std::cout << "Starting search..." << std::endl;

    try {
        auto stats = searcher.find_best_move(board, 2, fake_eval);
        std::cout << "best_move=" << stats.best_move << " nodes=" << stats.nodes_visited << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
