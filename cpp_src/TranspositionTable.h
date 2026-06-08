#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <iostream>

enum class NodeType : uint8_t { MAX = 0, CHANCE = 1 };

// Keeping the struct small maximizes CPU cache hits.
// 16 bytes per entry.
struct TTEntry {
    uint64_t key;   // Zobrist hash of the board state
    float    score; // Evaluated Expectimax score
    uint8_t  depth; // Search depth remaining (uint8_t is plenty for 2048)
    uint8_t  type;  // 0 = MAX, 1 = CHANCE
    uint8_t  generation; // Age tag (5 bits used, 0..31, & 0x1F wrap)
};

// 4-way associative bucket (64 bytes total, fits perfectly in a typical CPU cache line)
struct TTBucket {
    TTEntry entries[4];
};

class TranspositionTable {
public:
    // 2^22 buckets * 4 entries/bucket = 2^24 entries total (~256 MiB)
    static constexpr uint32_t NUM_BUCKETS = 4194304;
    static constexpr uint32_t BUCKET_MASK = NUM_BUCKETS - 1;

    TranspositionTable() {
        // Allocate once. Zero-initialise so key==0 means "empty slot".
        table = new TTBucket[NUM_BUCKETS]();
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
    static uint64_t hash_key(uint64_t key, uint8_t type) {
        uint64_t x = key ^ (static_cast<uint64_t>(type) << 1);
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    bool probe(uint64_t key, uint8_t depth, NodeType type, float& out_score) const {
        uint32_t idx = static_cast<uint32_t>(hash_key(key, static_cast<uint8_t>(type)) & BUCKET_MASK);
        const TTBucket& bucket = table[idx];
        
        for (int i = 0; i < 4; ++i) {
            const TTEntry& entry = bucket.entries[i];
            if (entry.key == key && entry.type == static_cast<uint8_t>(type)) {
                if (entry.depth >= depth) {
                    out_score = entry.score;
                    return true;
                }
                // Key matches but stored depth is shallower than requested search.
                return false;
            }
        }
        return false;
    }

    void store(uint64_t key, uint8_t depth, NodeType type, float score) {
        uint32_t idx = static_cast<uint32_t>(hash_key(key, static_cast<uint8_t>(type)) & BUCKET_MASK);
        TTBucket& bucket = table[idx];

        // 1. If key already exists in any slot, overwrite it.
        for (int i = 0; i < 4; ++i) {
            if (bucket.entries[i].key == key && bucket.entries[i].type == static_cast<uint8_t>(type)) {
                bucket.entries[i].score = score;
                bucket.entries[i].depth = depth;
                same_key_overwrite_count_++;
                return;
            }
        }

        // 2. Otherwise, find an empty slot.
        for (int i = 0; i < 4; ++i) {
            if (bucket.entries[i].key == 0) {
                bucket.entries[i].key   = key;
                bucket.entries[i].score = score;
                bucket.entries[i].depth = depth;
                bucket.entries[i].type  = static_cast<uint8_t>(type);
                num_entries_++;
                return;
            }
        }

        // 3. All slots full: replace the one with the smallest depth.
        // Use collision_count to cycle through slots on equal depth to avoid ping-ponging.
        collision_count_++;
        int replace_idx = collision_count_ % 4;
        uint8_t min_depth = bucket.entries[replace_idx].depth;

        for (int i = 0; i < 4; ++i) {
            if (bucket.entries[i].depth < min_depth) {
                min_depth = bucket.entries[i].depth;
                replace_idx = i;
            }
        }
        
        bucket.entries[replace_idx].key   = key;
        bucket.entries[replace_idx].score = score;
        bucket.entries[replace_idx].depth = depth;
        bucket.entries[replace_idx].type  = static_cast<uint8_t>(type);
    }

    void clear() {
        std::memset(table, 0, NUM_BUCKETS * sizeof(TTBucket));
        num_entries_ = 0;
        collision_count_ = 0;
        same_key_overwrite_count_ = 0;
    }

    void reset_counters() {
        collision_count_ = 0;
        same_key_overwrite_count_ = 0;
    }

    void begin_new_search() {
        current_generation_ = (current_generation_ + 1) & 0x1F;
    }

    size_t occupancy() const { return num_entries_; }
    size_t collision_count() const { return collision_count_; }
    size_t same_key_overwrite_count() const { return same_key_overwrite_count_; }

private:
    TTBucket* table;
    size_t num_entries_ = 0;
    size_t collision_count_ = 0;
    size_t same_key_overwrite_count_ = 0;
    uint8_t current_generation_ = 0;
};
