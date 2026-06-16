#pragma once
#include <cstdint>
#include <array>
#include <algorithm>

using Board = std::array<std::array<int, 4>, 4>;

class BoardEncoder {
public:
    // Pack a 4x4 board into a 64-bit integer (16 tiles × 4 bits each)
    static uint64_t pack(const Board& board);

    // Unpack a 64-bit integer back into a 4x4 board
    static Board unpack(uint64_t packed);

    // Compute the canonical representation among all 8 symmetries
    static uint64_t canonicalize(const Board& board);
    static uint64_t canonicalize(uint64_t packed);

    // Generate all 8 symmetry transforms of a packed board
    static void generate_symmetries(uint64_t packed, uint64_t out[8]);

private:
    // Rotate 90 degrees clockwise
    static uint64_t rotate90(uint64_t packed);

    // Reflect horizontally (mirror left-right)
    static uint64_t reflect_h(uint64_t packed);
};