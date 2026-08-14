#include <iostream>
#include <stdexcept>
#include "Fast2048.h"
#include "TranspositionTable.h"

int main() {
    TranspositionTable tt;
    float score = 0.0f;

    tt.store(0, 0, NodeType::MAX, 42.0f);
    if (tt.occupancy() != 1 || !tt.probe(0, 0, NodeType::MAX, score) || score != 42.0f) {
        std::cerr << "zero key was not stored and probed correctly\n";
        return 1;
    }

    tt.clear();
    if (tt.occupancy() != 0 || tt.probe(0, 0, NodeType::MAX, score)) {
        std::cerr << "zero key survived clear\n";
        return 1;
    }

    Fast2048 game;
    const std::array<std::array<int, 4>, 4> board{{
        {{15, 15, 0, 0}},
        {{0, 0, 0, 0}},
        {{0, 0, 0, 0}},
        {{0, 0, 0, 0}},
    }};
    game.set_board(board);
    const int score_before = game.get_score();
    if (game.is_move_valid(3)) {
        std::cerr << "simulated overflow was reported as valid\n";
        return 1;
    }
    auto [simulated_score, simulated_moved] = game.move_simulated(3);

    if (simulated_score != 0 || simulated_moved || game.get_board() != board ||
        game.get_score() != score_before || game.is_move_valid(3)) {
        std::cerr << "simulated overflow mutated game state\n";
        return 1;
    }

    return 0;
}
