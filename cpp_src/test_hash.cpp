#include <iostream>
#include <set>
#include <cstdint>

uint64_t hash_key(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

int main() {
    std::set<uint32_t> seen;
    for (uint64_t i = 0; i < 10000; ++i) {
        seen.insert(static_cast<uint32_t>(hash_key(i) & 0xFFFFFF));
    }
    std::cout << "Unique hashes for 0-9999: " << seen.size() << std::endl;
    
    // Test with some typical board values
    uint64_t boards[] = {
        0x0000000000000000ULL,
        0x0000000000000001ULL,
        0x0000000000000010ULL,
        0x0000000000000100ULL,
        0x0000000000001000ULL,
        0x0000000000010000ULL,
        0x0000000000100000ULL,
        0x0000000001000000ULL,
        0x0000000010000000ULL,
        0x0000000100000000ULL,
    };
    for (auto b : boards) {
        std::cout << "hash(" << std::hex << b << ") = " << hash_key(b) << std::dec << std::endl;
    }
    return 0;
}
