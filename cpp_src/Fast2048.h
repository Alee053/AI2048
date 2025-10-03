/**
 * @file Fast2048.h
 * @brief Defines the Fast2048 class, a C++ implementation of the 2048 game logic.
 */

﻿#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include "RandomUtil.h"
#include <iostream>
#include <tuple>

/**
 * @class Fast2048
 * @brief A highly optimized, Look-Up Table (LUT) based C++ implementation of the 2048 game.
 *
 * This class manages the game state, including the board, score, and game-over status.
 * It uses pre-computed look-up tables for all 65,536 possible row states to perform
 * move calculations with exceptional speed, making it suitable for high-throughput
 * simulations and search algorithms.
 */
class Fast2048 {
public:
    /**
     * @brief Constructs a new Fast2048 game instance.
     * Initializes the look-up tables if they haven't been created yet.
     */
    Fast2048();

    /**
     * @brief Resets the game to a starting state.
     * Clears the board, resets the score, and places two new random tiles.
     */
    void reset();

    /**
     * @brief Performs a move in a given direction and updates the game state.
     * @param direction The direction to move (0: Up, 1: Right, 2: Down, 3: Left).
     * @return A tuple containing:
     *         - int: The score obtained from merges in this move.
     *         - bool: A flag indicating if the game is over after the move.
     *         - bool: A flag indicating if the move resulted in a change to the board.
     */
    std::tuple<int, bool, bool> move(int direction);

    /**
     * @brief Checks if a move is possible in the given direction.
     * @param action The direction of the move to check.
     * @return True if the move is valid, false otherwise.
     */
    bool is_move_valid(int action) const;

    /**
     * @brief Gets a copy of the current game board.
     * @return A 4x4 array representing the board with log2-encoded tile values.
     */
    std::array<std::array<int, 4>, 4> get_board() const;

    /**
     * @brief Sets the game board to a specific state.
     * @param new_board The 4x4 array to set as the current board.
     */
    void set_board(const std::array<std::array<int, 4>, 4>& new_board);

    /**
     * @brief Gets the current score.
     * @return The integer score.
     */
    int get_score() const;

    /**
     * @brief Gets the value of the highest tile on the board.
     * @return The log2-encoded value of the max tile.
     */
    int get_max_tile() const;

private:
    /**
     * @brief Initializes the Look-Up Tables (LUTs) for moves, rewards, and validity.
     * This is a one-time operation that pre-computes the result of a left-move
     * for all 65,536 possible row configurations.
     */
    void init_LUT();

    /**
     * @brief Places a new random tile ('2' or '4') on an empty cell of the board.
     */
    void generate_random();

    /**
     * @brief Checks if there are any valid moves left on the board.
     * @return True if the game is still playable, false if it's over.
     */
    bool is_playable() const;

    /**
     * @brief Updates the score and max_tile attributes based on the current board state.
     */
    void update_values();

    /**
     * @brief Converts a 4-element row into a 16-bit integer for LUT indexing.
     * @param row The row to convert.
     * @return The integer representation of the row.
     */
    int row_to_number(const std::array<int, 4>& row) const;

    /// @brief The 4x4 game board, storing log2 of tile values (e.g., 3 represents the 8 tile).
    std::array<std::array<int, 4>, 4> board;
    /// @brief The current game score.
    int score;
    /// @brief Flag indicating if the game is over.
    bool done;
    /// @brief The log2 value of the highest tile on the board.
    int max_tile;


    /// @brief Static LUT for the resulting row after a left-move.
    static std::vector<std::array<int, 4>> move_row_LUT;
    /// @brief Static LUT for the reward obtained from a left-move.
    static std::vector<int> move_reward_LUT;
    /// @brief Static LUT for checking if a left-move is valid for a row.
    static std::vector<bool> move_valid_LUT;
};