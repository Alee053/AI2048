#include <iostream>
#include <iomanip> // For std::setw
#include <string>
#include <cmath>   // For std::pow

#include "Fast2048.h" // Your game class

// Helper function to print the board to the console
void print_board(const std::array<std::array<int, 4>, 4>& board) {
    std::cout << "-----------------------------" << std::endl;
    for (int r = 0; r < 4; ++r) {
        std::cout << "|";
        for (int c = 0; c < 4; ++c) {
            if (board[r][c] == 0) {
                std::cout << std::setw(6) << " .";
            } else {
                // Convert log2 value back to the actual tile value (2, 4, 8...)
                int tile_value = static_cast<int>(std::pow(2, board[r][c]));
                std::cout << std::setw(6) << tile_value;
            }
        }
        std::cout << " |" << std::endl;
    }
    std::cout << "-----------------------------" << std::endl;
}

int main() {
    Fast2048 game; // Create an instance of your game
    int score = 0;

    std::cout << "2048 C++ Console Tester" << std::endl;
    std::cout << "Use W(up), A(left), S(down), D(right) to move. Type 'q' to quit." << std::endl;

    while (true) {
        print_board(game.get_board());
        std::cout << "Current Score: " << score << std::endl;
        std::cout << "Enter move (w/a/s/d): ";

        char input;
        std::cin >> input;

        if (input == 'q') {
            break;
        }

        int direction = -1;
        switch (input) {
            case 'w': direction = 0; break; // Up
            case 'd': direction = 1; break; // Right
            case 's': direction = 2; break; // Down
            case 'a': direction = 3; break; // Left
            default:
                std::cout << "Invalid input. Please use w, a, s, or d." << std::endl;
                continue; // Skip the rest of the loop
        }

        // Use C++17 structured bindings to unpack the returned tuple
        auto [merge_score, done, moved] = game.move(direction);
        score += merge_score;

        if (done) {
            std::cout << "\n--- GAME OVER ---" << std::endl;
            print_board(game.get_board());
            std::cout << "Final Score: " << score << std::endl;
            break;
        }
    }

    return 0;
}