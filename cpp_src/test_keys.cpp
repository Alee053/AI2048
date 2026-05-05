#include <iostream>
#include <set>
#include <vector>
#include "TranspositionTable.h"
#include "BoardEncoder.h"

uint64_t hash_key(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

int main() {
    // Simulate storing the keys from the search
    Board board = {{
        {{0, 1, 5, 1}},
        {{0, 1, 2, 6}},
        {{0, 0, 0, 1}},
        {{0, 0, 0, 1}}
    }};
    
    TranspositionTable tt;
    std::set<uint64_t> keys;
    std::set<uint32_t> indices;
    
    // We'll just store a bunch of canonical boards from this board's search tree
    // by doing a simple DFS
    
    std::vector<Board> stack;
    stack.push_back(board);
    
    int stored = 0;
    int collisions = 0;
    
    while (!stack.empty() && stored < 5000) {
        Board b = stack.back();
        stack.pop_back();
        
        uint64_t canon = BoardEncoder::canonicalize(b);
        uint32_t idx = static_cast<uint32_t>(hash_key(canon) & 0xFFFFFF);
        
        if (keys.insert(canon).second) {
            // New key
            bool idx_new = indices.insert(idx).second;
            bool idx_chance_new = indices.insert(static_cast<uint32_t>(hash_key(canon + 1) & 0xFFFFFF)).second;
            
            tt.store(canon, 0, NodeType::MAX, 1.0f);
            tt.store(canon, 1, NodeType::MAX, 2.0f);
            tt.store(canon, 2, NodeType::MAX, 3.0f);
            tt.store(canon, 0, NodeType::CHANCE, 4.0f);
            tt.store(canon, 1, NodeType::CHANCE, 5.0f);
            tt.store(canon, 2, NodeType::CHANCE, 6.0f);
            stored += 6;
            
            if (!idx_new || !idx_chance_new) {
                collisions++;
            }
        }
        
        // Generate some child boards (moves + spawns)
        // This is a simplified version just to generate many boards
        for (int r = 0; r < 4 && stack.size() < 10000; ++r) {
            for (int c = 0; c < 4 && stack.size() < 10000; ++c) {
                if (b[r][c] == 0) {
                    Board child = b;
                    child[r][c] = 1; // place a 2
                    stack.push_back(child);
                }
            }
        }
    }
    
    std::cout << "Stored: " << stored << std::endl;
    std::cout << "Collisions: " << collisions << std::endl;
    std::cout << "TT occupancy: " << tt.occupancy() << std::endl;
    std::cout << "TT collision_count: " << tt.collision_count() << std::endl;
    std::cout << "TT same_key_overwrite_count: " << tt.same_key_overwrite_count() << std::endl;
    std::cout << "sizeof(TTEntry): " << sizeof(TTEntry) << std::endl;
    std::cout << "TT_SIZE: " << TranspositionTable::TT_SIZE << std::endl;
    std::cout << "Total bytes: " << (TranspositionTable::TT_SIZE * sizeof(TTEntry)) << std::endl;
    
    // Verify zero initialization by probing random empty slots
    float dummy;
    int empty_hits = 0;
    for (int i = 0; i < 1000; ++i) {
        if (tt.probe(0xDEADBEEF00000000ULL + i, 0, NodeType::MAX, dummy)) empty_hits++;
    }
    std::cout << "Empty hits (should be 0): " << empty_hits << std::endl;
    return 0;
}
