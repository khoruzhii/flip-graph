// scheme.cpp
#include "scheme.h"
#include <iostream>
#include <algorithm>
#include <set>

Scheme::Scheme(const std::vector<U64>& initial_data, int seed) : rng(seed) {
    data = initial_data;
    n_orbits = data.size() / 3;
    
    // Build initial structures
    for (int i = 0; i < n_orbits * 3; i++) {
        if (data[i] != 0) {
            value_to_indices[data[i]].push_back(i);
        }
    }
    
    // Initialize twoplusl
    for (const auto& [value, indices] : value_to_indices) {
        if (indices.size() >= 2) {
            // Check if indices are from different orbits
            std::set<int> orbits;
            for (int idx : indices) {
                orbits.insert(idx / 3);
                if (orbits.size() >= 2) break;
            }
            
            if (orbits.size() >= 2) {
                twoplus_index[value] = twoplusl.size();
                twoplusl.push_back(value);
            }
        }
    }
}

void Scheme::update_twoplus(U64 value, size_t new_count) {
    if (new_count < 2) {
        // Remove from twoplusl if present
        auto it = twoplus_index.find(value);
        if (it != twoplus_index.end()) {
            int idx = it->second;
            twoplus_index.erase(it);
            
            if (idx != twoplusl.size() - 1) {
                twoplusl[idx] = twoplusl.back();
                twoplus_index[twoplusl[idx]] = idx;
            }
            twoplusl.pop_back();
        }
        return;
    }
    
    // Check if indices are from different orbits
    const auto& indices = value_to_indices[value];
    std::set<int> orbits;
    for (int idx : indices) {
        orbits.insert(idx / 3);
        if (orbits.size() >= 2) break;  // Found 2+ different orbits
    }
    
    bool has_different_orbits = (orbits.size() >= 2);
    auto it = twoplus_index.find(value);
    bool in_twoplus = (it != twoplus_index.end());
    
    if (has_different_orbits && !in_twoplus) {
        // Add to twoplusl
        twoplus_index[value] = twoplusl.size();
        twoplusl.push_back(value);
    } else if (!has_different_orbits && in_twoplus) {
        // Remove from twoplusl
        int idx = it->second;
        twoplus_index.erase(it);
        
        if (idx != twoplusl.size() - 1) {
            twoplusl[idx] = twoplusl.back();
            twoplus_index[twoplusl[idx]] = idx;
        }
        twoplusl.pop_back();
    }
}

bool Scheme::flip() {
    if (twoplusl.empty()) return false;
    
    // 1. Select random value (guaranteed to have indices from different orbits)
    U64 selected_value = twoplusl[rng() % twoplusl.size()];
    const auto& indices = value_to_indices[selected_value];
    
    // 2. Find two indices from different orbits
    int idx1 = -1, idx2 = -1;
    
    for (int attempts = 0; attempts < 100; attempts++) {
        int i1 = rng() % indices.size();
        int i2 = rng() % indices.size();
        if (i1 != i2 && indices[i1] / 3 != indices[i2] / 3) {
            idx1 = indices[i1];
            idx2 = indices[i2];
            break;
        }
    }
    
    if (idx1 == -1) return false;  // Should rarely happen now
    
    // 3. Calculate indices to modify
    int orb1 = idx1 / 3, sym1 = idx1 % 3;
    int orb2 = idx2 / 3, sym2 = idx2 % 3;
    int idx1_next = orb1 * 3 + (sym1 + 1) % 3;
    int idx2_prev = orb2 * 3 + (sym2 + 2) % 3;
    
    // 4. Store old values BEFORE modifying data
    U64 old_val1 = data[idx1_next];
    U64 old_val2 = data[idx2_prev];
    
    // 5. Perform XOR
    data[idx1_next] ^= data[orb2 * 3 + (sym2 + 1) % 3];
    data[idx2_prev] ^= data[orb1 * 3 + (sym1 + 2) % 3];
    
    // 6. Get new values AFTER XOR
    U64 new_val1 = data[idx1_next];
    U64 new_val2 = data[idx2_prev];
    
    // 7. Collect all affected values for twoplus update
    std::set<U64> affected_values;
    if (old_val1 != 0) affected_values.insert(old_val1);
    if (old_val2 != 0) affected_values.insert(old_val2);
    if (new_val1 != 0) affected_values.insert(new_val1);
    if (new_val2 != 0) affected_values.insert(new_val2);
    
    // 8. Remove old indices from value_to_indices
    // IMPORTANT: Handle both indices even if they have the same value
    if (old_val1 != 0) {
        auto& vec = value_to_indices[old_val1];
        vec.erase(std::remove(vec.begin(), vec.end(), idx1_next), vec.end());
        if (vec.empty()) {
            value_to_indices.erase(old_val1);
        }
    }
    
    if (old_val2 != 0) {
        auto& vec = value_to_indices[old_val2];
        vec.erase(std::remove(vec.begin(), vec.end(), idx2_prev), vec.end());
        if (vec.empty()) {
            value_to_indices.erase(old_val2);
        }
    }
    
    // 9. Add new indices to value_to_indices
    // IMPORTANT: Add both indices even if they have the same value
    if (new_val1 != 0) {
        value_to_indices[new_val1].push_back(idx1_next);
    }
    
    if (new_val2 != 0) {
        value_to_indices[new_val2].push_back(idx2_prev);
    }
    
    // 10. Update twoplusl for all affected values
    for (U64 val : affected_values) {
        if (value_to_indices.count(val)) {
            update_twoplus(val, value_to_indices[val].size());
        } else {
            update_twoplus(val, 0);
        }
    }
    
    return true;
}

void Scheme::print() const {
    for (int i = 0; i < n_orbits; i++) {
        std::cout << i << ": " 
                  << data[i*3] << " " 
                  << data[i*3+1] << " " 
                  << data[i*3+2] << "\n";
    }
}

void Scheme::print_matches() const {
    std::cout << "Unique values: " << value_to_indices.size() << "\n";
    std::cout << "Values with 2+ occurrences: " << twoplusl.size() << "\n";
    
    // Show some statistics
    int total_occurrences = 0;
    int max_occurrences = 0;
    for (const auto& [val, indices] : value_to_indices) {
        total_occurrences += indices.size();
        max_occurrences = std::max(max_occurrences, (int)indices.size());
    }
    
    std::cout << "Total non-zero positions: " << total_occurrences << "\n";
    std::cout << "Max occurrences of single value: " << max_occurrences << "\n";
}
