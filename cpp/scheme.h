// scheme.h
#pragma once 

#include <vector>
#include <unordered_map>
#include <random>
#include <cstdint>

using U64 = std::uint64_t;

class Scheme {
public:
    std::vector<U64> data;  // [u0,v0,w0, u1,v1,w1, ...]
    int n_orbits;
    
    // Core data structures
    std::unordered_map<U64, std::vector<int>> value_to_indices;  // value -> positions
    std::vector<U64> twoplusl;                                   // values with 2+ occurrences
    std::unordered_map<U64, int> twoplus_index;                  // value -> index in twoplusl
    
    // Random generator
    std::mt19937 rng;
    
    // Constructor
    Scheme(const std::vector<U64>& initial_data, int seed = 42);
    
    // Core operations
    bool flip();
    
    // Statistics
    void print() const;
    void print_matches() const;
    int count_nonzero() const;
    
private:
    // Helper to maintain twoplusl
    void update_twoplus(U64 value, size_t new_count);
};