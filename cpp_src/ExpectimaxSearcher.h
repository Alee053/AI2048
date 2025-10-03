/**
 * @file ExpectimaxSearcher.h
 * @brief Defines the ExpectimaxSearcher class for finding the optimal move in 2048.
 */

﻿#pragma once

#include "Fast2048.h"
#include <vector>
#include <functional>
#include <map>

/**
 * @brief Type alias for the 4x4 game board.
 */
using Board = std::array<std::array<int, 4>, 4>;

/**
 * @brief Type alias for the batch evaluation function callback.
 * This function takes a vector of boards and returns a vector of their values,
 * typically evaluated by a neural network critic.
 */
using BatchEvalFunc = std::function<std::vector<float>(const std::vector<Board>&)>;

/**
 * @class ExpectimaxSearcher
 * @brief Implements an Expectimax algorithm to determine the best move in the game 2048.
 *
 * This class uses a combination of the Expectimax search algorithm and a neural
 * network-based value function (critic) to explore the game tree. The search
 * is batched for efficiency: it first gathers all unique leaf nodes at a certain
 * depth, evaluates them in a single batch call to the Python-side critic, and then
 * propagates the values back up the tree to find the move with the highest
 * expected value.
 */
class ExpectimaxSearcher {
public:
    /**
     * @brief Constructs a new ExpectimaxSearcher object.
     */
    ExpectimaxSearcher();

    /**
     * @brief Finds the best move for a given board state using the Expectimax search.
     * @param board The current 4x4 game board.
     * @param depth The maximum depth to search in the game tree.
     * @param batch_eval_func A callback function to evaluate a batch of boards.
     * @return The best move to take (0: Up, 1: Right, 2: Down, 3: Left).
     */
    int find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func);

private:
    /**
     * @brief Recursively gathers all unique leaf nodes at a specified depth.
     * @param board The current board state to explore from.
     * @param depth The remaining depth to search.
     * @param leaves_queue A vector to store the unique leaf boards found.
     * @param visited A map to keep track of visited boards to avoid redundant exploration.
     */
    void gather_leaves(const Board& board, int depth, std::vector<Board>& leaves_queue, std::map<Board, bool>& visited);

    /**
     * @brief Represents a "max" node in the Expectimax tree, where the agent chooses a move.
     *
     * This function is called after the leaf nodes have been evaluated. It calculates
     * the value of a state by choosing the move that leads to the chance node with
     * the highest expected value.
     *
     * @param board The board state for this max node.
     * @param depth The remaining depth of the search.
     * @param eval_cache A cache containing the pre-computed values of leaf nodes.
     * @return The maximum expected value achievable from this state.
     */
    float max_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache);

    /**
     * @brief Represents a "chance" node in the Expectimax tree, where a new tile appears randomly.
     *
     * This function calculates the expected value of a state by averaging the values
     * of all possible next states that can result from a random tile spawn.
     *
     * @param board The board state for this chance node.
     * @param depth The remaining depth of the search.
     * @param eval_cache A cache containing the pre-computed values of leaf nodes.
     * @return The average or "expected" value of this state.
     */
    float chance_node_substitute(const Board& board, int depth, const std::map<Board, float>& eval_cache);

    /// @brief An instance of the game logic used for simulating moves.
    Fast2048 game_instance;
};