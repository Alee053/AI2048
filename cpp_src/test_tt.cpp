#include <iostream>
#include <random>
#include "TranspositionTable.h"

int main() {
    TranspositionTable tt;
    std::mt19937_64 rng(42);
    
    for (int i = 0; i < 5000; ++i) {
        uint64_t key = rng();
        tt.store(key, 0, NodeType::MAX, float(i));
    }
    
    std::cout << "Occupancy: " << tt.occupancy() << std::endl;
    std::cout << "Collisions: " << tt.collision_count() << std::endl;
    std::cout << "Same key overwrites: " << tt.same_key_overwrite_count() << std::endl;
    return 0;
}
