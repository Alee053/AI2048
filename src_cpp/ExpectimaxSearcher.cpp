#include "ExpectimaxSearcher.h"
#include <vector>
#include <iostream>

ExpectimaxSearcher::ExpectimaxSearcher() = default;

int ExpectimaxSearcher::find_best_move_with_eval(
    const std::array<std::array<int, 4>, 4>& board,
    int depth,
    const std::function<float(const std::array<std::array<int, 4>, 4>&)>& eval_func) {
    float best_score = -1e9;
    int best_move = -1;

    for (int move = 0; move < 4; ++move) {
        Fast2048 temp_game;
        temp_game.set_board(board);

        if (!temp_game.is_move_valid(move)) {
            continue;
        }

        auto [merge_score, done, moved] = temp_game.move(move);

        float score = static_cast<float>(merge_score) + chance_node(temp_game, depth, eval_func);
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }

    if (best_move == -1) {
        Fast2048 temp_game;
        temp_game.set_board(board);
        for(int m = 0; m < 4; ++m) {
            if (temp_game.is_move_valid(m)) return m;
        }
    }

    return best_move;
}

float ExpectimaxSearcher::chance_node(Fast2048& game, int depth, const std::function<float(const std::array<std::array<int, 4>, 4>&)>& eval_func) {
    std::vector<std::pair<int, int>> empty_cells;
    auto board = game.get_board();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (board[r][c] == 0) {
                empty_cells.emplace_back(r, c);
            }
        }
    }

    if (empty_cells.empty()) {
        return 0.0f;
    }

    float total_value = 0.0f;
    for (const auto& cell : empty_cells) {
        Fast2048 game_with_2 = game;
        auto board_with_2 = game_with_2.get_board();
        board_with_2[cell.first][cell.second] = 1;
        game_with_2.set_board(board_with_2);
        total_value += 0.9f * max_node(game_with_2, depth - 1, eval_func);

        Fast2048 game_with_4 = game;
        auto board_with_4 = game_with_4.get_board();
        board_with_4[cell.first][cell.second] = 2;
        game_with_4.set_board(board_with_4);
        total_value += 0.1f * max_node(game_with_4, depth - 1, eval_func);
    }

    // Return the average expected value
    return total_value / empty_cells.size();
}

float ExpectimaxSearcher::max_node(Fast2048& game, int depth, const std::function<float(const std::array<std::array<int, 4>, 4>&)>& eval_func) {
    if (depth <= 0) {
        return eval_func(game.get_board());
    }

    float max_value = -1e9;
    bool any_move_possible = false;

    for (int move = 0; move < 4; ++move) {
        Fast2048 temp_game = game;

        if (!temp_game.is_move_valid(move)) {
            continue;
        }

        auto [merge_score, done, moved] = temp_game.move(move);

        if (!moved) {
            continue;
        }
        
        any_move_possible = true;
        float value = static_cast<float>(merge_score) + chance_node(temp_game, depth, eval_func);
        if (value > max_value) {
            max_value = value;
        }
    }

    if (!any_move_possible) {
        return 0.0f;
    }

    return max_value;
}
