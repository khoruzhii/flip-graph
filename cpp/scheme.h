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
    std::unordered_map<U64, std::vector<int>> value_map;  // value -> list of positions
    std::vector<U64> flippable;  // values that appear 2+ times in different orbits
    
    // Lookup tables for cyclic permutations
    std::vector<int> next;  // next[i] = orbit(i)*3 + (pos(i)+1)%3
    std::vector<int> prev;  // prev[i] = orbit(i)*3 + (pos(i)+2)%3
    
    // Random generator
    std::mt19937 rng;
    
    // Constructor
    Scheme(const std::vector<U64>& initial_data, int seed = 42);
    
    // Perform one flip operation
    bool flip();
    
    // Debug output
    void print() const;
    
private:
    // Find two indices from different orbits
    bool find_different_orbits(const std::vector<int>& indices, U64 val, int& idx1, int& idx2);
    
    // Check if value should be in flippable list
    bool is_flippable(U64 value) const;
    
    // Update flippable list after value count changes
    void upd_flippable(U64 value);
};