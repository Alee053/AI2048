#include "ExpectimaxSearcher.h"
#include <vector>

ExpectimaxSearcher::ExpectimaxSearcher() = default;

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
        if (!game_instance.is_move_valid(move)) continue;

        auto [ms, done, moved] = game_instance.move(move);
        if (!moved) continue;

        Board post_move_board = game_instance.get_board();

        // Explore chance nodes
        std::vector<std::pair<int, int>> empty_cells;
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) if (post_move_board[r][c] == 0) empty_cells.push_back({r, c});

        if (empty_cells.empty()) continue;

        for (const auto& cell : empty_cells) {
            Board next_board = post_move_board;
            // We only need to check one tile type (e.g., a '2') to gather the unique board states.
            next_board[cell.first][cell.second] = 1;
            gather_leaves(next_board, depth - 1, leaves_queue, visited);
        }
    }
}

float ExpectimaxSearcher::chance_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache) {
    std::vector<std::pair<int, int>> empty_cells;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) if (board[r][c] == 0) empty_cells.push_back({r, c});
    if (empty_cells.empty()) return 0;

    float total_value = 0;
    for (const auto& cell : empty_cells) {
        Board next_board2 = board; next_board2[cell.first][cell.second] = 1;
        total_value += 0.9f * max_node_substitute(next_board2, depth - 1, eval_cache);
        Board next_board4 = board; next_board4[cell.first][cell.second] = 2;
        total_value += 0.1f * max_node_substitute(next_board4, depth - 1, eval_cache);
    }
    return total_value / empty_cells.size();
}

float ExpectimaxSearcher::max_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache) {
    if (depth == 0) {
        return eval_cache.count(board) ? eval_cache.at(board) : 0.0f;
    }

    float max_value = -1e9;
    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        if (!game_instance.is_move_valid(move)) continue;

        auto [merge_score, done, moved] = game_instance.move(move);
        if (!moved) continue;

        float value = merge_score + chance_node_substitute(game_instance.get_board(), depth, eval_cache);
        if (value > max_value) max_value = value;
    }
    return (max_value == -1e9) ? 0.0f : max_value;
}


int ExpectimaxSearcher::find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func) {
    std::vector<Board> leaves_to_evaluate;
    std::map<Board, bool> visited_leaves;
    gather_leaves(board, depth, leaves_to_evaluate, visited_leaves);

    std::vector<float> evaluations;
    if (!leaves_to_evaluate.empty()) {
        evaluations = batch_eval_func(leaves_to_evaluate);
    }

    std::map<Board, float> eval_cache;
    for (size_t i = 0; i < leaves_to_evaluate.size(); ++i) {
        eval_cache[leaves_to_evaluate[i]] = evaluations[i];
    }

    float best_score = -1e9;
    int best_move = -1;

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        if (!game_instance.is_move_valid(move)) continue;

        auto [merge_score, done, moved] = game_instance.move(move);
        if (!moved) continue;

        float score = merge_score + chance_node_substitute(game_instance.get_board(), depth, eval_cache);
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }
    return best_move != -1 ? best_move : 0;
}