/**
 * @file ExpectimaxSearcher.cpp
 * @brief Implements the ExpectimaxSearcher class methods.
 */
﻿#include "ExpectimaxSearcher.h"
#include <vector>

/**
 * @brief Default constructor for the ExpectimaxSearcher.
 */
ExpectimaxSearcher::ExpectimaxSearcher() = default;

/**
 * @brief Recursively explores the game tree to find all unique board states at a given depth.
 *
 * This function performs a depth-first search. At each "max" node (where the agent moves),
 * it tries all valid moves. At each "chance" node (after a move), it considers all
 * possible locations for the next random tile. To avoid re-exploring identical subtrees,
 * it only explores one type of random tile spawn (e.g., a '2' tile) since the goal is
 * only to identify the unique board layouts that need to be evaluated by the critic.
 *
 * @param board The current board state to explore from.
 * @param depth The remaining depth to search.
 * @param leaves_queue A reference to a vector where the found leaf boards will be stored.
 * @param visited A reference to a map used to track visited board states to prevent duplicates.
 */
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

/**
 * @brief Calculates the expected value of a "chance" node in the game tree.
 *
 * After a move is made, the game randomly places a new tile (either a '2' or a '4')
 * in an empty cell. This function calculates the weighted average value over all
 * possible outcomes of this random event.
 *
 * @param board The board state after a move, for which to calculate the expected value.
 * @param depth The remaining search depth.
 * @param eval_cache A map containing the pre-computed values of leaf nodes.
 * @return The expected value of the board state.
 */
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

/**
 * @brief Calculates the value of a "max" node in the game tree.
 *
 * This function simulates the agent's turn, where it chooses the move that leads
 * to the state with the highest expected value. If the current node is a leaf node
 * (depth is 0), it returns the pre-computed value from the evaluation cache.
 *
 * @param board The board state for which to find the best move's value.
 * @param depth The remaining search depth.
 * @param eval_cache A map containing the pre-computed values of leaf nodes.
 * @return The value of the best possible move from this state. Returns 0 if no moves are possible.
 */
float ExpectimaxSearcher::max_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache) {
    if (depth == 0) {
        return eval_cache.count(board) ? eval_cache.at(board) : 0.0f;
    }

    float max_value = -1e9;
    bool move_found = false;
    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        if (!game_instance.is_move_valid(move)) continue;

        auto [merge_score, done, moved] = game_instance.move(move);
        if (!moved) continue;

        move_found = true;
        float value = merge_score + chance_node_substitute(game_instance.get_board(), depth, eval_cache);
        if (value > max_value) max_value = value;
    }
    return move_found ? max_value : 0.0f;
}

/**
 * @brief Orchestrates the entire Expectimax search to find the best move.
 *
 * This is the main public method of the class. It performs the search in three phases:
 * 1.  Calls `gather_leaves` to explore the game tree down to the specified depth and
 *     collect all unique leaf board states.
 * 2.  Calls the `batch_eval_func` (a callback to the Python-side PPO critic) to
 *     evaluate all collected leaf nodes in a single batch.
 * 3.  Propagates these values back up the tree by calling `chance_node_substitute`
 *     to determine the score for each initial move.
 *
 * @param board The current 4x4 game board state.
 * @param depth The maximum depth to search.
 * @param batch_eval_func The callback function for batch evaluation of boards.
 * @return An integer representing the best move (0-3). Returns a default move (0) if no valid move is found.
 */
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