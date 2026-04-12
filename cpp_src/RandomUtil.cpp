#include "RandomUtil.h"
#include <type_traits>
#include <algorithm>

RandomUtil& RandomUtil::get() {
    static RandomUtil instance;
    return instance;
}

RandomUtil::RandomUtil() {
    std::random_device rd;
    m_engine.seed(rd());
}

template<typename T>
T RandomUtil::getRandom(T min, T max) {
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(m_engine);
    }
    else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(m_engine);
    }
}

// Template instantiations
template int RandomUtil::getRandom<int>(int min, int max);
template double RandomUtil::getRandom<double>(double min, double max);
template float RandomUtil::getRandom<float>(float min, float max);

// Zobrist hashing implementation
std::array<std::array<std::array<uint64_t, 17>, 4>, 4> RandomUtil::zobrist_table;
bool RandomUtil::zobrist_initialized = false;

void RandomUtil::init_zobrist_table() {
    if (zobrist_initialized) return;
    std::mt19937_64 rng(0xdeadbeef);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            for (int v = 0; v <= 16; ++v) {
                zobrist_table[r][c][v] = rng();
            }
        }
    }
    zobrist_initialized = true;
}

uint64_t RandomUtil::compute_board_hash(const std::array<std::array<int, 4>, 4>& board) {
    uint64_t h = 0;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int v = board[r][c];
            if (v > 16) v = 16;
            h ^= zobrist_table[r][c][v];
        }
    }
    return h;
}

uint64_t RandomUtil::zobrist_reseed(uint64_t hash, int row, int col, int tile_value) {
    int v = std::max(0, std::min(tile_value, 16));
    return hash ^ zobrist_table[row][col][v];
}
