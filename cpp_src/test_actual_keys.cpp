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
    uint64_t keys[] = {
        299068857262625,
        299075758063970,
        281483840455010,
        281479277318690,
        17596486652450,
        580543833973281,
        316667944108386,
        281484108890466,
        281483572027746,
        17602470486561,
        8592429905,
        8860861265,
        8609203025,
        17600778470225,
        1108104053585,
        8592434001,
        9129296721,
        8625980241,
        35192964514641,
        2207615681361,
    };
    
    std::set<uint32_t> seen_max;
    std::set<uint32_t> seen_chance;
    
    for (auto k : keys) {
        uint32_t idx_max = static_cast<uint32_t>(hash_key(k + 0) & 0xFFFFFF);
        uint32_t idx_chance = static_cast<uint32_t>(hash_key(k + 1) & 0xFFFFFF);
        seen_max.insert(idx_max);
        seen_chance.insert(idx_chance);
        std::cout << "key=" << k << " max_idx=" << idx_max << " chance_idx=" << idx_chance << std::endl;
    }
    
    std::cout << "Unique max indices: " << seen_max.size() << "/20" << std::endl;
    std::cout << "Unique chance indices: " << seen_chance.size() << "/20" << std::endl;
    return 0;
}
