#pragma once
#include <random>
#include <array>
#include <cstdint>

class RandomUtil {
public:
    static RandomUtil& get();

    template<typename T>
    T getRandom(T min, T max);

    // Zobrist hashing for 2048 boards
    static void init_zobrist_table();
    static uint64_t compute_board_hash(const std::array<std::array<int, 4>, 4>& board);
    static uint64_t zobrist_reseed(uint64_t hash, int row, int col, int tile_value);

private:
    RandomUtil();

    RandomUtil(const RandomUtil&) = delete;
    void operator=(const RandomUtil&) = delete;

    std::mt19937 m_engine;

    static std::array<std::array<std::array<uint64_t, 17>, 4>, 4> zobrist_table;  // [row][col][tile_value]
    static bool zobrist_initialized;
};

