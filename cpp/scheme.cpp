// scheme.cpp
#include "scheme.h"
#include <iostream>
#include <algorithm>
#include <set>

Scheme::Scheme(const std::vector<U64>& initial_data, int seed) : rng(seed) {
    data = initial_data;
    n_orbits = data.size() / 3;
    orank = 0;
    
    // Initialize lookup tables for cyclic permutations
    next.resize(n_orbits * 3);
    prev.resize(n_orbits * 3);
    for (int i = 0; i < n_orbits * 3; i++) {
        int orb = i / 3;
        int pos = i % 3;
        next[i] = orb * 3 + (pos + 1) % 3;
        prev[i] = orb * 3 + (pos + 2) % 3;
    }
    
    // Build initial value_map and count active orbits
    for (int i = 0; i < n_orbits * 3; i++) {
        if (data[i] != 0) {
            value_map[data[i]].push_back(i);
        }
    }
    
    // Count active orbits (orbit is active if ALL components are non-zero)
    for (int i = 0; i < n_orbits; i++) {
        if (data[i*3] != 0 && data[i*3+1] != 0 && data[i*3+2] != 0) {
            orank++;
        }
    }
    
    // Initialize flippable with values that appear 2+ times in different orbits
    for (const auto& [value, indices] : value_map) {
        if (is_flippable(value)) {
            flippable.push_back(value);
        }
    }
}

bool Scheme::is_flippable(U64 value) const {
    auto it = value_map.find(value);
    if (it == value_map.end() || it->second.size() < 2) {
        return false;
    }
    
    // Check if indices are from different orbits
    const auto& indices = it->second;
    int first_orbit = indices[0] / 3;
    for (size_t i = 1; i < indices.size(); i++) {
        if (indices[i] / 3 != first_orbit) {
            return true;  // Found different orbit
        }
    }
    return false;
}

void Scheme::upd_flippable(U64 value) {
    bool should_be = is_flippable(value);
    
    // Check if already in flippable
    auto pos = std::find(flippable.begin(), flippable.end(), value);
    bool is_in = (pos != flippable.end());
    
    if (should_be && !is_in) {
        // Add to flippable
        flippable.push_back(value);
    } else if (!should_be && is_in) {
        // Remove from flippable
        flippable.erase(pos);
    }
}

bool Scheme::find_different_orbits(const std::vector<int>& indices, U64 val, int& idx1, int& idx2) {
    for (int attempts = 0; attempts < 100; attempts++) {
        int i1 = rng() % indices.size();
        int i2 = rng() % indices.size();
        if (i1 != i2 && indices[i1] / 3 != indices[i2] / 3) {
            idx1 = indices[i1];
            idx2 = indices[i2];
            return true;
        }
    }
    
    std::cerr << "Error: couldn't find indices from different orbits for value " << val << "\n";
    return false;
}

bool Scheme::flip() {
    if (flippable.empty()) return false;
    
    // Select random value from flippable
    U64 val = flippable[rng() % flippable.size()];
    const auto& indices = value_map[val];
    
    // Find two indices from different orbits
    int idx1, idx2;
    if (!find_different_orbits(indices, val, idx1, idx2)) {
        return false;
    }
    
    // Use lookup tables for cyclic indices
    int j1 = next[idx1];
    int j2 = prev[idx2];
    int orb1 = j1 / 3;
    int orb2 = j2 / 3;
    
    // Check if orbits were active before modification
    bool orb1_was_active = (data[orb1*3] != 0 && data[orb1*3+1] != 0 && data[orb1*3+2] != 0);
    bool orb2_was_active = (data[orb2*3] != 0 && data[orb2*3+1] != 0 && data[orb2*3+2] != 0);
    
    // Store old values
    U64 old1 = data[j1];
    U64 old2 = data[j2];
    
    // XOR modifications using lookup tables
    data[j1] ^= data[next[idx2]];
    data[j2] ^= data[prev[idx1]];
    
    // Get new values
    U64 new1 = data[j1];
    U64 new2 = data[j2];
    
    // First, update value_map for the changed positions
    // Remove old indices
    if (old1 != 0) {
        auto& vec1 = value_map[old1];
        vec1.erase(std::remove(vec1.begin(), vec1.end(), j1), vec1.end());
        if (vec1.empty()) value_map.erase(old1);
    }
    
    if (old2 != 0 && j1 != j2) {  // Avoid double removal if same position
        auto& vec2 = value_map[old2];
        vec2.erase(std::remove(vec2.begin(), vec2.end(), j2), vec2.end());
        if (vec2.empty()) value_map.erase(old2);
    }
    
    // Add new indices
    if (new1 != 0) {
        value_map[new1].push_back(j1);
    }
    
    if (new2 != 0 && j1 != j2) {  // Avoid double addition if same position
        value_map[new2].push_back(j2);
    }
    
    // Collect all affected values for flippable update
    std::set<U64> affected_values;
    if (old1 != 0) affected_values.insert(old1);
    if (old2 != 0) affected_values.insert(old2);
    if (new1 != 0) affected_values.insert(new1);
    if (new2 != 0) affected_values.insert(new2);
    
    // Check if any orbit needs to be zeroed (if any component became 0)
    std::set<int> orbits_to_zero;
    if (new1 == 0) {
        orbits_to_zero.insert(orb1);
    }
    if (new2 == 0) {
        orbits_to_zero.insert(orb2);
    }
    
    // Zero out complete orbits
    for (int orbit : orbits_to_zero) {
        // Determine if this orbit was active before
        bool was_active = false;
        if (orbit == orb1) {
            was_active = orb1_was_active;
        } else if (orbit == orb2) {
            was_active = orb2_was_active;
        }
        
        // Collect values from orbit before zeroing (for flippable update)
        for (int i = 0; i < 3; i++) {
            int idx = orbit * 3 + i;
            if (data[idx] != 0) {
                affected_values.insert(data[idx]);
            }
        }
        
        // Remove all components from value_map and zero them
        for (int i = 0; i < 3; i++) {
            int idx = orbit * 3 + i;
            U64 val = data[idx];
            if (val != 0) {
                auto& vec = value_map[val];
                vec.erase(std::remove(vec.begin(), vec.end(), idx), vec.end());
                if (vec.empty()) {
                    value_map.erase(val);
                }
                data[idx] = 0;
            }
        }
        
        // Decrement orank only if orbit was active
        if (was_active) {
            orank--;
        }
    }
    
    // Update flippable for all affected values
    for (U64 v : affected_values) {
        upd_flippable(v);
    }
    
    return true;
}

void Scheme::print() const {
    std::cout << "Active orbits: " << orank << "/" << n_orbits << "\n";
    for (int i = 0; i < n_orbits; i++) {
        if (data[i*3] != 0 && data[i*3+1] != 0 && data[i*3+2] != 0) {
            std::cout << i << ": " 
                      << data[i*3] << " " 
                      << data[i*3+1] << " " 
                      << data[i*3+2] << "\n";
        }
    }
}