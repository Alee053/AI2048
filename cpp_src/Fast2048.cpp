/**
 * @file Fast2048.cpp
 * @brief Implements the methods of the Fast2048 class.
 */
#include "Fast2048.h"

/**
 * @brief Constructs a Fast2048 object, initializing LUTs if necessary and resetting the board.
 */
Fast2048::Fast2048() {
    if (move_row_LUT.empty())
        init_LUT();
    reset();
}

/**
 * @brief Resets the game to a clean state with two random tiles.
 */
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

/**
 * @brief Executes a move, updates the board, and adds a new random tile.
 *
 * This function uses the pre-computed LUTs to perform moves. For moves other than
 * LEFT, it transforms the board (reversing for RIGHT, transposing for UP/DOWN) to
 * effectively reuse the same LEFT-move LUT, then transforms it back.
 */
std::tuple<int, bool, bool> Fast2048::move(int direction) {
    int merge_score = 0;

    bool moved=is_move_valid(direction);
    if (!moved)
        return {0, done, false};

    // Note: Directions are mapped differently here than in the Python version.
    // 3: LEFT, 1: RIGHT, 0: UP, 2: DOWN
    if (direction == 3) { // LEFT
        for (auto &row : board) {
            int index=row_to_number(row);
            merge_score += move_reward_LUT[index];
            row = move_row_LUT[index];
        }
    }
    else if (direction == 1) // RIGHT
    {
        for (auto &row : board) {
            std::reverse(row.begin(), row.end());
            int index=row_to_number(row);
            merge_score += move_reward_LUT[index];
            row = move_row_LUT[index];
            std::reverse(row.begin(), row.end());
        }
    }
    else if (direction == 0) // UP
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
    else if (direction == 2) // DOWN
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
    done = !is_playable();

    return {merge_score, done, moved};
}

/**
 * @brief Checks if a move in any direction is valid by checking the corresponding LUT.
 */
bool Fast2048::is_move_valid(int direction) const {
    if (direction == 3) { // LEFT
        for (auto &row : board) {
            if (move_valid_LUT[row_to_number(row)])return true;
        }
    }
    else if (direction == 1) // RIGHT
    {
        for (auto &row : board) {
            std::array<int, 4> reversed_row = row;
            std::reverse(reversed_row.begin(), reversed_row.end());
            if (move_valid_LUT[row_to_number(reversed_row)])return true;
        }
    }
    else if (direction == 0) // UP
    {
        for (int col=0;col<4;col++) {
            std::array<int, 4> column;
            for (int row=0;row<4;row++)
                column[row] = board[row][col];
            if (move_valid_LUT[row_to_number(column)])return true;
        }
    }
    else if (direction == 2) // DOWN
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

/**
 * @brief Returns a copy of the current board state.
 */
std::array<std::array<int, 4>, 4> Fast2048::get_board() const {
    return std::array<std::array<int, 4>, 4>(board);
}

/**
 * @brief Manually sets the board to a given state and updates game values.
 */
void Fast2048::set_board(const std::array<std::array<int, 4>, 4> &new_board) {
    board = new_board;

    score = 0;
    done = !is_playable();
    update_values();
}

/**
 * @brief Returns the current score.
 */
int Fast2048::get_score() const {
    return score;
}

/**
 * @brief Returns the log2 value of the highest tile.
 */
int Fast2048::get_max_tile() const {
    return max_tile;
}

/**
 * @brief Populates the static Look-Up Tables (LUTs).
 *
 * This function iterates through all 65,536 (2^16) possible 4-tile rows. For each
 * row, it simulates a left-move by stacking tiles, merging them, and stacking again.
 * The resulting row, the reward from the merge, and whether the row changed are
 * stored in the corresponding LUTs.
 */
void Fast2048::init_LUT() {
    for (int i=0;i<65536;i++) {
        std::array<int, 4> original_row,row;
        for (int j=0;j<4;j++) {
            original_row[j] = (i >> (j * 4)) & 0xF;
            row[j] = original_row[j];
        }

        // Stack tiles to the left
        for (int j=0;j<4;j++) {
            for (int k=1;k<4;k++) {
                if (row[k]!=0 && row[k-1] == 0) {
                    std::swap(row[k-1], row[k]);
                }
            }
        }
        // Merge identical adjacent tiles
        int reward = 0;
        for (int j=1;j<4;j++) {
            if (row[j-1]==row[j] && row[j]!=0) {
                row[j-1]++;
                row[j] = 0;
                reward += (1 << row[j-1]);
            }
        }
        // Stack again after merging
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

/**
 * @brief Finds all empty cells and places a new tile (90% '2', 10% '4') in a random one.
 */
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

    int new_tile_value = (probability < 0.9) ? 1 : 2; // 1 represents 2, 2 represents 4

    board[chosen_cell.first][chosen_cell.second] = new_tile_value;
}

/**
 * @brief Determines if the game is over by checking if any move is valid.
 */
bool Fast2048::is_playable() const {
    bool res=false;
    for (int i=0;i<4;i++) {
        res|=is_move_valid(i);
        if (res) break;
    }
    return res;
}

/**
 * @brief Updates the `max_tile` member variable by scanning the board.
 */
void Fast2048::update_values() {
    for (const auto &row : board) {
        for (const auto &tile : row) {
            if (tile > max_tile)
                max_tile = tile;
        }
    }
}

/**
 * @brief Encodes a row into a 16-bit integer for efficient LUT indexing.
 */
int Fast2048::row_to_number(const std::array<int, 4> &row) const {
    return row[0] | (row[1] << 4) | (row[2] << 8) | (row[3] << 12);
}

// --- Static Member Initialization ---
std::vector<std::array<int, 4>> Fast2048::move_row_LUT;
std::vector<int> Fast2048::move_reward_LUT;
std::vector<bool> Fast2048::move_valid_LUT;
