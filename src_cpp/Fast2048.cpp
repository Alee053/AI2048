#pragma once
#include "Fast2048.h"

Fast2048::Fast2048() {
    if (move_row_LUT.empty())
        init_LUT();
    reset();
}

void Fast2048::reset() {
    for (auto &row : board)
        row.fill(0);
    score = 0;
    done = false;
    max_tile = 0;
    generate_random();
    generate_random();
    update_values();
}

std::tuple<int, bool, bool> Fast2048::move(int direction) {
    int merge_score = 0;

    bool moved=is_move_valid(direction);
    if (!moved)
        return {0, done, false};

    if (direction == 3) {
        for (auto &row : board) {
            int index=row_to_number(row);
            merge_score += move_reward_LUT[index];
            row = move_row_LUT[index];
        }
    }
    else if (direction == 1)
    {
        for (auto &row : board) {
            std::reverse(row.begin(), row.end());
            int index=row_to_number(row);
            merge_score += move_reward_LUT[index];
            row = move_row_LUT[index];
            std::reverse(row.begin(), row.end());
        }
    }
    else if (direction == 0)
    {
        for (int col=0;col<4;col++) {
            std::array<int, 4> column;
            for (int row=0;row<4;row++)
                column[row] = board[row][col];
            int index=row_to_number(column);
            merge_score += move_reward_LUT[index];
            column = move_row_LUT[index];
            for (int row=0;row<4;row++)
                board[row][col] = column[row];
        }
    }
    else if (direction == 2)
    {
        for (int col=0;col<4;col++) {
            std::array<int, 4> column;
            for (int row=0;row<4;row++)
                column[row] = board[3-row][col];
            int index=row_to_number(column);
            merge_score += move_reward_LUT[index];
            column = move_row_LUT[index];
            for (int row=0;row<4;row++)
                board[3-row][col] = column[row];
        }
    }
    score += merge_score;


    generate_random();
    update_values();
    done = check_done();

    return {merge_score, done, moved};
}

bool Fast2048::is_move_valid(int direction) const {
    if (direction == 3) {
        for (auto &row : board) {
            if (move_valid_LUT[row_to_number(row)])return true;
        }
    }
    else if (direction == 1)
    {
        for (auto &row : board) {
            std::array<int, 4> reversed_row = row;
            std::reverse(reversed_row.begin(), reversed_row.end());
            if (move_valid_LUT[row_to_number(reversed_row)])return true;
        }
    }
    else if (direction == 0)
    {
        for (int col=0;col<4;col++) {
            std::array<int, 4> column;
            for (int row=0;row<4;row++)
                column[row] = board[row][col];
            if (move_valid_LUT[row_to_number(column)])return true;
        }
    }
    else if (direction == 2)
    {
        for (int col=0;col<4;col++) {
            std::array<int, 4> column;
            for (int row=0;row<4;row++)
                column[row] = board[3-row][col];
            if (move_valid_LUT[row_to_number(column)])return true;
        }
    }
    return false;
}

std::array<std::array<int, 4>, 4> Fast2048::get_board() const {
    return std::array<std::array<int, 4>, 4>(board);
}

void Fast2048::set_board(const std::vector<std::vector<int>> &new_board) {
    board=std::array<std::array<int, 4>, 4>();
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++) {
            board[i][j] = new_board[i][j];
        }
    }
    score = 0;
    done = check_done();
    update_values();
}

int Fast2048::get_score() const {
    return score;
}

int Fast2048::get_max_tile() const {
    return max_tile;
}

void Fast2048::init_LUT() {
    for (int i=0;i<65536;i++) {
        std::array<int, 4> original_row,row;
        for (int j=0;j<4;j++) {
            original_row[j] = (i >> (j * 4)) & 0xF;
            row[j] = original_row[j];
        }

        // Stack
        for (int j=0;j<4;j++) {
            for (int k=1;k<4;k++) {
                if (row[k]!=0 && row[k-1] == 0) {
                    std::swap(row[k-1], row[k]);
                }
            }
        }
        // Merge
        int reward = 0;
        for (int j=1;j<4;j++) {
            if (row[j-1]==row[j] && row[j]!=0) {
                row[j-1]++;
                row[j] = 0;
                reward += (1 << row[j-1]);
            }
        }
        // Stack
        for (int j=0;j<4;j++) {
            for (int k=1;k<4;k++) {
                if (row[k]!=0 && row[k-1] == 0) {
                    std::swap(row[k-1], row[k]);
                }
            }
        }

        move_row_LUT.push_back(row);
        move_reward_LUT.push_back(reward);
        move_valid_LUT.push_back(original_row != row);
    }
}

void Fast2048::generate_random() {
    std::vector<std::pair<int,int>> empty_positions;
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++) {
            if (board[i][j]==0)
                empty_positions.emplace_back(i,j);
        }
    }

    if (empty_positions.empty())
        return;

    int cell_index = RandomUtil::get().getRandom<int>(0, empty_positions.size() - 1);
    std::pair<int, int> chosen_cell = empty_positions[cell_index];

    double probability = RandomUtil::get().getRandom<double>(0.0, 1.0);

    int new_tile_value = (probability < 0.9) ? 1 : 2;

    board[chosen_cell.first][chosen_cell.second] = new_tile_value;
}

bool Fast2048::check_done() const {
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++) {
            if (board[i][j]==0)
                return false;
            if (j<3 && board[i][j]==board[i][j+1])
                return false;
            if (i<3 && board[i][j]==board[i+1][j])
                return false;
        }
    }
    return true;
}

void Fast2048::update_values() {
    for (const auto &row : board) {
        for (const auto &tile : row) {
            if (tile > max_tile)
                max_tile = tile;
        }
    }
}

int Fast2048::row_to_number(const std::array<int, 4> &row) const {
    return row[0] | (row[1] << 4) | (row[2] << 8) | (row[3] << 12);
}

std::vector<std::array<int, 4>> Fast2048::move_row_LUT;
std::vector<int> Fast2048::move_reward_LUT;
std::vector<bool> Fast2048::move_valid_LUT;
