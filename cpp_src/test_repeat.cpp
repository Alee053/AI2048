#include <iostream>
#include <set>
#include <vector>
#include "TranspositionTable.h"
#include "BoardEncoder.h"

int main() {
    TranspositionTable tt;
    
    // These are actual keys from the search output
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
    
    // Store each key multiple times with different depths and types
    for (int i = 0; i < 100; ++i) {
        for (auto k : keys) {
            tt.store(k, 0, NodeType::MAX, 1.0f);
            tt.store(k, 1, NodeType::MAX, 2.0f);
            tt.store(k, 2, NodeType::MAX, 3.0f);
            tt.store(k, 0, NodeType::CHANCE, 4.0f);
            tt.store(k, 1, NodeType::CHANCE, 5.0f);
            tt.store(k, 2, NodeType::CHANCE, 6.0f);
        }
    }
    
    std::cout << "Occupancy: " << tt.occupancy() << std::endl;
    std::cout << "Collisions: " << tt.collision_count() << std::endl;
    std::cout << "Same key overwrites: " << tt.same_key_overwrite_count() << std::endl;
    
    return 0;
}
