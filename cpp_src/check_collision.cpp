#include <iostream>
#include <cstdint>

enum class NodeType : uint8_t { MAX = 0, CHANCE = 1 };

static constexpr uint32_t TT_SIZE = 16777216;
static constexpr uint32_t TT_MASK = TT_SIZE - 1;

static uint64_t hash_key(uint64_t key, uint8_t type) {
    uint64_t x = key ^ (static_cast<uint64_t>(type) << 1);
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int main() {
    uint64_t key1 = 281617081893201ULL;
    uint64_t key2 = 587140587065617ULL;

    for (int type = 0; type <= 1; ++type) {
        uint32_t idx1 = hash_key(key1, type) & TT_MASK;
        uint32_t idx2 = hash_key(key2, type) & TT_MASK;
        std::cout << "Type " << type << ":\n";
        std::cout << "  Key1: " << key1 << " -> Idx: " << idx1 << "\n";
        std::cout << "  Key2: " << key2 << " -> Idx: " << idx2 << "\n";
        if (idx1 == idx2) {
            std::cout << "  COLLISION DETECTED!\n";
        }
    }
    return 0;
}
