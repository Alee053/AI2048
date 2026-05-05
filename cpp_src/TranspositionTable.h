#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

enum class NodeType : uint8_t { MAX = 0, CHANCE = 1 };

// Keeping the struct small maximizes CPU cache hits.
// 16 bytes per entry => 256 MiB for 2^24 slots.
struct TTEntry {
    uint64_t key;   // Zobrist hash of the board state
    float    score; // Evaluated Expectimax score
    uint8_t  depth; // Search depth remaining (uint8_t is plenty for 2048)
    uint8_t  type;  // 0 = MAX, 1 = CHANCE
};

class TranspositionTable {
public:
    // 2^24 entries ≈ 16.7 million slots (~256 MiB)
    static constexpr uint32_t TT_SIZE = 16777216;
    static constexpr uint32_t TT_MASK = TT_SIZE - 1;

    TranspositionTable() {
        // Allocate once. Zero-initialise so key==0 means "empty slot".
        table = new TTEntry[TT_SIZE]();
        if (!table) {
            throw std::bad_alloc();
        }
    }

    ~TranspositionTable() {
        delete[] table;
    }

    // Non-copyable / non-movable
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    // Scramble 64-bit key into a well-distributed 64-bit hash.
    // Uses splitmix64 to avoid clustering when low bits of canonical boards collide.
static uint64_t hash_key(uint64_t key, uint8_t type) {
        uint64_t x = key ^ (static_cast<uint64_t>(type) << 1);
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    bool probe(uint64_t key, uint8_t depth, NodeType type, float& out_score) const {
        uint32_t idx = static_cast<uint32_t>(hash_key(key, static_cast<uint8_t>(type)) & TT_MASK);
        const TTEntry& entry = table[idx];
        if (entry.key == key && entry.type == static_cast<uint8_t>(type) && entry.depth >= depth) {
            out_score = entry.score;
            return true;
        }
        static int miss_log = 20;
        if (miss_log > 0 && depth <= 2) {
            std::cerr << "[TT_MISS] key=" << key << " depth=" << (int)depth << " type=" << (int)type
                      << " entry_depth=" << (int)entry.depth << " entry_key=" << entry.key
                      << " entry_type=" << (int)entry.type << " idx=" << idx << "\n";
            miss_log--;
        }
        return false;
    }

    void store(uint64_t key, uint8_t depth, NodeType type, float score) {
        uint32_t idx = static_cast<uint32_t>(hash_key(key, static_cast<uint8_t>(type)) & TT_MASK);
        TTEntry& entry = table[idx];

        bool replace = false;
        if (entry.key == 0) {
            // Empty sentinel
            replace = true;
            num_entries_++;
        } else if (entry.key == key && entry.type == static_cast<uint8_t>(type)) {
            // Same board+type: always replace (higher depth entries can't overwrite lower ones)
            replace = true;
            same_key_overwrite_count_++;
        } else {
            replace = true;
            collision_count_++;
            if (collision_count_ <= 10) {
                std::cerr << "[COLLISION] idx=" << idx << " old_key=" << entry.key
                          << " new_key=" << key << " old_type=" << (int)entry.type
                          << " new_type=" << (int)type << "\n";
            }
        }

        if (replace) {
            entry.key   = key;
            entry.score = score;
            entry.depth = depth;
            entry.type  = static_cast<uint8_t>(type);
        }
    }

    // Wipe the table (rarely needed; mainly for explicit resets between models).
    void clear() {
        std::memset(table, 0, TT_SIZE * sizeof(TTEntry));
        num_entries_ = 0;
    }

    size_t occupancy() const { return num_entries_; }
    size_t collision_count() const { return collision_count_; }
    size_t same_key_overwrite_count() const { return same_key_overwrite_count_; }

private:
    TTEntry* table;
    size_t num_entries_ = 0;
    size_t collision_count_ = 0;
    size_t same_key_overwrite_count_ = 0;
};
