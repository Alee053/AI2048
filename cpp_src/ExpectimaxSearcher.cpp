#include "ExpectimaxSearcher.h"
#include <vector>

ExpectimaxSearcher::ExpectimaxSearcher() = default;

float get_log_reward(int merge_score) {
    if (merge_score <= 0) return 0.0f;
    return std::log2(static_cast<float>(merge_score));
}

void ExpectimaxSearcher::gather_leaves(const Board& board, int depth, std::vector<Board>& leaves_queue, std::map<Board, bool>& visited) {
    if (depth == 0) {
        if (visited.find(board) == visited.end()) {
            leaves_queue.push_back(board);
            visited[board] = true;
        }
        return;
    }

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);

        auto [ms, moved] = game_instance.move_simulated(move);
        if (!moved) continue;

        Board post_move_board = game_instance.get_board();

        std::vector<std::pair<int, int>> empty_cells;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if (post_move_board[r][c] == 0) empty_cells.push_back({r, c});
            }
        }

        if (empty_cells.empty()) {
             if (visited.find(post_move_board) == visited.end()) {
                leaves_queue.push_back(post_move_board);
                visited[post_move_board] = true;
            }
            continue;
        }

        for (const auto& cell : empty_cells) {
            Board next_board_2 = post_move_board;
            next_board_2[cell.first][cell.second] = 1; 
            gather_leaves(next_board_2, depth - 1, leaves_queue, visited);

            Board next_board_4 = post_move_board;
            next_board_4[cell.first][cell.second] = 2; 
            gather_leaves(next_board_4, depth - 1, leaves_queue, visited);
        }
    }
}

float ExpectimaxSearcher::chance_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache) {
    std::vector<std::pair<int, int>> empty_cells;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) if (board[r][c] == 0) empty_cells.push_back({r, c});

    if (empty_cells.empty()) {
        return max_node_substitute(board, depth - 1, eval_cache);
    }

    float total_value = 0;
    for (const auto& cell : empty_cells) {
        Board next_board2 = board;
        next_board2[cell.first][cell.second] = 1;
        total_value += 0.9f * max_node_substitute(next_board2, depth - 1, eval_cache);

        Board next_board4 = board;
        next_board4[cell.first][cell.second] = 2;
        total_value += 0.1f * max_node_substitute(next_board4, depth - 1, eval_cache);
    }
    return total_value / empty_cells.size();
}

float ExpectimaxSearcher::max_node_substitute(const Board& board, int depth, const std::map<Board, float>& leaf_cache) {
    if (depth == 0) {
        return leaf_cache.count(board) ? leaf_cache.at(board) : -100.0f;
    }

    TranspositionKey key = {board, depth};
    if (transposition_table.count(key)) {
        return transposition_table[key];
    }

    float max_value = -1e9;
    bool any_move_possible = false;

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        auto [merge_score, moved] = game_instance.move_simulated(move);

        if (!moved) continue;
        any_move_possible = true;

        float immediate_reward = get_log_reward(merge_score);
        float future_value = chance_node_substitute(game_instance.get_board(), depth, leaf_cache);

        float total_value = immediate_reward + future_value;
        if (total_value > max_value) max_value = total_value;
    }

    float result = any_move_possible ? max_value : 0.0f;

    transposition_table[key] = result;

    return result;
}

int ExpectimaxSearcher::find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func) {
    transposition_table.clear();

    std::vector<Board> leaves_to_evaluate;
    std::map<Board, bool> visited_leaves;

    gather_leaves(board, depth, leaves_to_evaluate, visited_leaves);

    std::vector<float> evaluations;
    if (!leaves_to_evaluate.empty()) {
        evaluations = batch_eval_func(leaves_to_evaluate);
    }

    std::map<Board, float> leaf_cache;
    for (size_t i = 0; i < leaves_to_evaluate.size(); ++i) {
        leaf_cache[leaves_to_evaluate[i]] = evaluations[i];
    }

    float best_score = -1e9;
    int best_move = -1;

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        auto [merge_score, moved] = game_instance.move_simulated(move);
        if (!moved) continue;

        float immediate_reward = get_log_reward(merge_score);

        float future_value = chance_node_substitute(game_instance.get_board(), depth, leaf_cache);

        float score = immediate_reward + future_value;

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }
    return best_move != -1 ? best_move : 0;
}