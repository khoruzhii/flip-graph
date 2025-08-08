// scheme.cpp
#include "scheme.h"
#include <iostream>
#include <algorithm>

Scheme::Scheme(const std::vector<U64>& initial_data, int seed) : rng(seed) {
    data = initial_data;
    n_orbits = data.size() / 3;
    
    // Initialize lookup tables for cyclic permutations
    next.resize(n_orbits * 3);
    prev.resize(n_orbits * 3);
    for (int i = 0; i < n_orbits * 3; i++) {
        int orb = i / 3;
        int pos = i % 3;
        next[i] = orb * 3 + (pos + 1) % 3;
        prev[i] = orb * 3 + (pos + 2) % 3;
    }
    
    // Build initial positions map
    for (int i = 0; i < n_orbits * 3; i++) {
        if (data[i] != 0) {
            positions[data[i]].push_back(i);
        }
    }
    
    // Initialize flippable with values that appear 2+ times in different orbits
    for (const auto& [value, indices] : positions) {
        if (is_flippable(value)) {
            flippable.push_back(value);
        }
    }

    // Initialize orank value
    orank = 0;
    for (int i = 0; i < n_orbits; i++) {
        if (data[i*3] != 0) orank++;
    }
}

// ========================================================
// ======================== HELPERS =======================
// ========================================================

void Scheme::positions_del(U64 val, int idx) {
    auto& vec = positions[val];
    auto it = std::find(vec.begin(), vec.end(), idx);
    if (it != vec.end()) {
        *it = vec.back();  // Replace with last element
        vec.pop_back();    // Remove last
    }
    if (vec.empty())
        positions.erase(val);
}

void Scheme::positions_add(U64 val, int idx) {
    positions[val].push_back(idx);
}

bool Scheme::is_flippable(U64 value) const {
    auto it = positions.find(value);
    if (it == positions.end() || it->second.size() < 2) {
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
    bool flag = is_flippable(value);
    
    // Check if already in flippable
    auto pos = std::find(flippable.begin(), flippable.end(), value);
    bool is_in = (pos != flippable.end());
    
    if (flag && !is_in) {
        // Add to flippable
        flippable.push_back(value);
    } else if (!flag && is_in) {
        // Remove from flippable
        flippable.erase(pos);
    }
}

// ========================================================
// ========================= FLIP =========================
// ========================================================

bool Scheme::sample_orbits_flip(int& idx1, int& idx2) {
    if (flippable.empty()) return false;
    
    // Select random value from flippable
    U64 val = flippable[rng() % flippable.size()];
    const auto& indices = positions[val];
    
    // Find two indices from different orbits
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
    // Clear affected values from previous flip
    affected.clear();
    
    // Sample two indices from different orbits
    int idx1, idx2;
    if (!sample_orbits_flip(idx1, idx2)) {
        return false;
    }
    
    // Use lookup tables for cyclic indices
    int j1 = next[idx1];
    int j2 = prev[idx2];
    
    // Store old values
    U64 old1 = data[j1];
    U64 old2 = data[j2];
    
    // XOR modifications using lookup tables
    data[j1] ^= data[next[idx2]];
    data[j2] ^= data[prev[idx1]];
    
    // Get new values
    U64 new1 = data[j1];
    U64 new2 = data[j2];
    
    // Update positions for the changed positions and zero out orbits if any component became 0
    positions_del(old1, j1);
    positions_del(old2, j2);
    affected.insert(old1);
    affected.insert(old2);
    
    if (new1 != 0) {
        positions_add(new1, j1);
        affected.insert(new1);
    } else {
        zero_orbit(j1 / 3);
    }
    
    if (new2 != 0) {
        positions_add(new2, j2);
        affected.insert(new2);
    } else {
        zero_orbit(j2 / 3);
    }
    
    // Update flippable for all affected values
    for (U64 v : affected) {
        upd_flippable(v);
    }
    
    return true;
}

void Scheme::zero_orbit(int orbit) {
    // Collect values from orbit before zeroing
    for (int i = 0; i < 3; i++) {
        int idx = orbit * 3 + i;
        if (data[idx] != 0) {
            affected.insert(data[idx]);
        }
    }
    
    // Remove all components from positions and zero them
    for (int i = 0; i < 3; i++) {
        int idx = orbit * 3 + i;
        U64 val = data[idx];
        if (val != 0) {
            positions_del(val, idx);
            data[idx] = 0;
        }
    }

    // Update orank value
    orank--;
}

// ========================================================
// ========================= PLUS =========================
// ========================================================

int Scheme::get_empty_orbit() const {
    for (int i = 0; i < n_orbits; i++) {
        if (data[i*3] == 0) {
            return i;
        }
    }
    return -1;
}

bool Scheme::sample_orbits_plus(int& u1, int& u2) {
    for (int attempts = 0; attempts < 1000; attempts++) {
        int o1 = rng() % n_orbits;
        int o2 = rng() % n_orbits;
        if (o1 != o2 && data[o1*3] && data[o2*3]) {
            u1 = o1 * 3;
            u2 = o2 * 3;
            return true;
        }
    }
    return false;
}

bool Scheme::plus() {
    // Find empty orbit
    int empty_orbit = get_empty_orbit();
    if (empty_orbit == -1) return false;
    
    // Sample two non-empty orbits
    int u1, u2;
    if (!sample_orbits_plus(u1, u2)) return false;

    // Component indices
    int v1 = next[u1], w1 = prev[u1];
    int v2 = next[u2], w2 = prev[u2];
    int u3 = empty_orbit * 3;
    int v3 = next[u3], w3 = prev[u3];
    
    // Save old values
    U64 old[3] = {data[v1], data[u2], data[w2]};
    U64 val_u1 = data[u1], val_v2 = data[v2], val_w1 = data[w1];
    
    // Clear affected
    affected.clear();
    
    // Update positions for changed values
    positions_del(old[0], v1);
    positions_del(old[1], u2);
    positions_del(old[2], w2);
    
    // Apply plus transition
    U64 new_vals[6] = {
        old[0] ^ val_v2,      // v1 ^= v2
        val_u1,               // u2 = u1
        val_w1 ^ old[2],      // w2 = w1 ^ w2
        val_u1 ^ old[1],      // u3 = u1 ^ u2
        val_v2,               // v3 = v2
        old[2]                // w3 = w2
    };
    
    int indices[6] = {v1, u2, w2, u3, v3, w3};
    
    // Set new values and update positions
    for (int i = 0; i < 6; i++) {
        data[indices[i]] = new_vals[i];
        if (new_vals[i]) {
            positions_add(new_vals[i], indices[i]);
            affected.insert(new_vals[i]);
        }
    }
    
    // Add old values to affected
    for (U64 v : old)
        if (v) affected.insert(v);
    
    // Update flippable for affected values
    for (U64 v : affected)
        upd_flippable(v);
    
    orank++;
    return true;
}
