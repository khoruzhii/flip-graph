// scheme.h
#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <random>
#include <cstdint>

using U64 = std::uint64_t;

class Scheme {
public:
    // Constructor
    Scheme(const std::vector<U64>& initial_data, int seed = 42);
    
    // One flip operation
    bool flip();

    // Plus transition
    bool plus();
    
    // Get helpers
    const std::vector<U64>& get_data() const { return data; }
    int get_orank() { return orank; }
    
private:
    std::vector<U64> data;  // [u0,v0,w0, u1,v1,w1, ...]
    int n_orbits;
    int orank;
    
    // Core data structures
    std::unordered_map<U64, std::vector<int>> positions;  // value -> list of positions
    std::vector<U64> flippable;  // values that appear 2+ times in different orbits
    std::set<U64> affected;  // values affected by current flip
    
    // Lookup tables for cyclic permutations
    std::vector<int> next;  // next[i] = orbit(i)*3 + (pos(i)+1)%3
    std::vector<int> prev;  // prev[i] = orbit(i)*3 + (pos(i)+2)%3
    
    // Random generator
    std::mt19937 rng;
    
    // Sample two indices from different orbits
    bool sample_orbits_flip(int& idx1, int& idx2);
    bool sample_orbits_plus(int& u1, int& u2);
    
    // Zero out entire orbit
    void zero_orbit(int orbit);
    
    // Helper methods for positions map
    void positions_del(U64 val, int idx);
    void positions_add(U64 val, int idx);
    
    // Check if value should be in flippable list
    bool is_flippable(U64 value) const;
    
    // Update flippable list after value count changes
    void upd_flippable(U64 value);

    // Returns first empty orbit index 
    int get_empty_orbit() const;
};