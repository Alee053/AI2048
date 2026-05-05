#include <iostream>
#include <set>
#include <random>
#include "BoardEncoder.h"

uint64_t hash_key(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

int main() {
    std::set<uint32_t> seen;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, 6);
    
    for (int i = 0; i < 100000; ++i) {
        Board board;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                board[r][c] = dist(rng);
        uint64_t canon = BoardEncoder::canonicalize(board);
        uint32_t idx_max = static_cast<uint32_t>(hash_key(canon + 0) & 0xFFFFFF);
        uint32_t idx_chance = static_cast<uint32_t>(hash_key(canon + 1) & 0xFFFFFF);
        seen.insert(idx_max);
        seen.insert(idx_chance);
    }
    
    std::cout << "Unique buckets for 100k random boards (both types): " << seen.size() << std::endl;
    
    // Now test with boards that have the same number of empty cells as the slow board
    std::set<uint32_t> slow_seen;
    Board slow = {{{{0, 1, 5, 1}}, {{0, 1, 2, 6}}, {{0, 0, 0, 1}}, {{0, 0, 0, 1}}}};
    
    // Generate boards similar to the slow board (sparse, specific values)
    std::uniform_int_distribution<int> cell_dist(0, 10);
    for (int i = 0; i < 100000; ++i) {
        Board board{};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if (cell_dist(rng) < 3) {
                    board[r][c] = dist(rng);
                }
            }
        }
        uint64_t canon = BoardEncoder::canonicalize(board);
        uint32_t idx = static_cast<uint32_t>(hash_key(canon + 0) & 0xFFFFFF);
        slow_seen.insert(idx);
    }
    std::cout << "Unique buckets for 100k sparse boards: " << slow_seen.size() << std::endl;
    return 0;
}
