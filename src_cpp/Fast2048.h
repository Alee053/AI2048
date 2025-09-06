#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include "RandomUtil.h"
#include <iostream>
#include <tuple>

class Fast2048 {
public:
    Fast2048();

    void reset();
    std::tuple<int, bool, bool> move(int direction);
    bool is_move_valid(int action) const;
    std::array<std::array<int, 4>, 4> get_board() const;

    void set_board(const std::array<std::array<int, 4>, 4>& new_board);
    int get_score() const;
    int get_max_tile() const;

private:
    void init_LUT();
    void generate_random();
    bool check_done() const;
    void update_values();
    int row_to_number(const std::array<int, 4>& row) const;

    std::array<std::array<int, 4>, 4> board;
    int score;
    bool done;
    int max_tile;


    static std::vector<std::array<int, 4>> move_row_LUT;
    static std::vector<int> move_reward_LUT;
    static std::vector<bool> move_valid_LUT;
};