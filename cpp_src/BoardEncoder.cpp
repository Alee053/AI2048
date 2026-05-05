#include "BoardEncoder.h"

uint64_t BoardEncoder::pack(const Board& board) {
    uint64_t packed = 0;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int shift = (r * 4 + c) * 4;
            packed |= (static_cast<uint64_t>(board[r][c] & 0xF) << shift);
        }
    }
    return packed;
}

Board BoardEncoder::unpack(uint64_t packed) {
    Board board{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int shift = (r * 4 + c) * 4;
            board[r][c] = static_cast<int>((packed >> shift) & 0xF);
        }
    }
    return board;
}

uint64_t BoardEncoder::rotate90(uint64_t packed) {
    // Rotate 90 degrees clockwise
    // new[r][c] = old[3-c][r]
    uint64_t result = 0;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int old_r = 3 - c;
            int old_c = r;
            int old_shift = (old_r * 4 + old_c) * 4;
            int new_shift = (r * 4 + c) * 4;
            result |= ((packed >> old_shift) & 0xF) << new_shift;
        }
    }
    return result;
}

uint64_t BoardEncoder::reflect_h(uint64_t packed) {
    // Reflect horizontally: new[r][c] = old[r][3-c]
    uint64_t result = 0;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int old_c = 3 - c;
            int old_shift = (r * 4 + old_c) * 4;
            int new_shift = (r * 4 + c) * 4;
            result |= ((packed >> old_shift) & 0xF) << new_shift;
        }
    }
    return result;
}

void BoardEncoder::generate_symmetries(uint64_t packed, uint64_t out[8]) {
    // Generate 4 rotations
    uint64_t rot0 = packed;
    uint64_t rot1 = rotate90(rot0);
    uint64_t rot2 = rotate90(rot1);
    uint64_t rot3 = rotate90(rot2);

    // Generate 4 reflected rotations
    uint64_t refl0 = reflect_h(rot0);
    uint64_t refl1 = reflect_h(rot1);
    uint64_t refl2 = reflect_h(rot2);
    uint64_t refl3 = reflect_h(rot3);

    out[0] = rot0;
    out[1] = rot1;
    out[2] = rot2;
    out[3] = rot3;
    out[4] = refl0;
    out[5] = refl1;
    out[6] = refl2;
    out[7] = refl3;
}

uint64_t BoardEncoder::canonicalize(const Board& board) {
    return canonicalize(pack(board));
}

uint64_t BoardEncoder::canonicalize(uint64_t packed) {
    uint64_t symmetries[8];
    generate_symmetries(packed, symmetries);
    uint64_t min_val = symmetries[0];
    for (int i = 1; i < 8; ++i) {
        if (symmetries[i] < min_val) {
            min_val = symmetries[i];
        }
    }
    return min_val;
}