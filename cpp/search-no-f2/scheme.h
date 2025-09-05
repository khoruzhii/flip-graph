// scheme.h
#pragma once

#include <vector>
#include <random>
#include <array>
#include <cstdint>
#include "hash_dict.h"

using U64 = std::uint64_t;
using U32 = std::uint32_t;

class Scheme {
private:
    std::vector<U64> data;
    
    // For each component (0,1,2), maintain:
    // - unique[comp]: HashDict mapping value -> block index in pos
    // - flippable_idx[comp]: HashDict mapping value -> index in flippable[comp]
    // - flippable[comp]: vector of values with multiplicity >= 2
    std::array<HashDict, 3> unique;
    std::array<HashDict, 3> flippable_idx;
    std::array<std::vector<U64>, 3> flippable;
    
    // Position blocks: for each distinct value, store [len, idx1, idx2, ...]
    // Each component has its own pos array
    std::array<std::vector<int>, 3> pos;
    std::array<std::vector<int>, 3> free_slots;
    
    std::mt19937 rng;
    int reduce_cnt;
    int type, j1, j2;
    
    // Pair selection tables (for efficient random pair selection)
    std::vector<int> pair_starts;
    std::vector<int> pair_i;
    std::vector<int> pair_j;
    
    static constexpr int BLOCK_SIZE = 128; // Max terms per block
    
    void build_pair_tables() {
        pair_starts.clear();
        pair_i.clear();
        pair_j.clear();
        pair_starts.reserve(100);
        pair_i.reserve(6400);
        pair_j.reserve(6400);
        
        pair_starts.push_back(0);
        pair_starts.push_back(0);
        
        for (int x = 1; x < 80; ++x) {
            for (int y = 0; y < x; ++y) {
                pair_i.push_back(x);
                pair_j.push_back(y);
                pair_i.push_back(y);
                pair_j.push_back(x);
            }
            pair_starts.push_back(static_cast<int>(pair_i.size()));
        }
    }
    
    void add(int term_idx, int comp, U64 val) {
        if (val == 0) return;
        
        int present = unique[comp].contains(val);
        if (present) {
            int b = unique[comp].getvaluex(val);
            int l = pos[comp][b];
            
            if (l == 1) {
                // Becoming flippable
                flippable_idx[comp].lasthash = unique[comp].lasthash;
                flippable_idx[comp].addx(val, static_cast<int>(flippable[comp].size()));
                flippable[comp].push_back(val);
            }
            
            ++l;
            pos[comp][b + l] = term_idx;
            pos[comp][b] = l;
        } else {
            if (free_slots[comp].empty()) {
                // Allocate new block
                int b = static_cast<int>(pos[comp].size());
                pos[comp].resize(b + BLOCK_SIZE + 1);
                free_slots[comp].push_back(b);
            }
            
            int b = free_slots[comp].back();
            free_slots[comp].pop_back();
            
            unique[comp].addx(val, b);
            pos[comp][b] = 1;
            pos[comp][b + 1] = term_idx;
        }
    }
    
    void del(int term_idx, int comp, U64 val) {
        if (val == 0) return;
        
        if (!unique[comp].contains(val)) return;
        
        int b = unique[comp].getvaluex(val);
        int l = pos[comp][b];
        
        if (l == 2) {
            // No longer flippable
            flippable_idx[comp].lasthash = unique[comp].lasthash;
            if (flippable_idx[comp].contains(val)) {
                int idx = flippable_idx[comp].getvaluex(val);
                U64 last_val = flippable[comp].back();
                if (idx < flippable[comp].size() - 1) {
                    flippable_idx[comp].replace(last_val, idx);
                    flippable[comp][idx] = last_val;
                }
                flippable[comp].pop_back();
                flippable_idx[comp].lasthash = unique[comp].lasthash;
                flippable_idx[comp].removex(val);
            }
        }
        
        if (l == 1) {
            // Remove from unique
            free_slots[comp].push_back(b);
            unique[comp].removex(val);
        } else {
            // Remove term_idx from block
            int i = b + l;
            int x = pos[comp][i];
            while (x != term_idx) {
                --i;
                int y = x;
                x = pos[comp][i];
                pos[comp][i] = y;
            }
            pos[comp][b] = l - 1;
        }
    }
    
    // Sample random pair of terms with duplicate component
    bool sample_pair() {
        int s0 = flippable[0].size();
        int s1 = flippable[1].size();
        int s2 = flippable[2].size();
        
        if (s0 + s1 + s2 == 0) 
            return false;
        
        // Choose component weighted by number of flippable values
        unsigned int sample = rng();
        int x = sample % (s0 + s1 + s2);
        
        if (x < s0) {
            type = 0;
        } else if (x < s0 + s1) {
            type = 1;
            x -= s0;
        } else {
            type = 2;
            x -= s0 + s1;
        }
        
        U64 val = flippable[type][x];
        int b = unique[type].getvalue(val);
        int l = pos[type][b];
        
        if (l < 2) return false; // Shouldn't happen if data is consistent
        
        // Select two distinct indices from the block
        ++b; // Point to first index
        
        if (l == 2) {
            if (sample & 65536U) {
                j1 = pos[type][b];
                j2 = pos[type][b + 1];
            } else {
                j1 = pos[type][b + 1];
                j2 = pos[type][b];
            }
        } else {
            // Use pair tables for efficient random selection
            if (l < pair_starts.size()) {
                int pair_idx = (sample >> 16) % pair_starts[l];
                j1 = pos[type][b + pair_i[pair_idx]];
                j2 = pos[type][b + pair_j[pair_idx]];
            } else {
                // Fallback for large l
                int i1 = (sample >> 16) % l;
                int i2 = (sample >> 24) % (l - 1);
                if (i2 >= i1) ++i2;
                j1 = pos[type][b + i1];
                j2 = pos[type][b + i2];
            }
        }
        
        return true;
    }
    
    // Sample any random pair of distinct terms (for plus-transition)
    bool sample_any_pair(int& idx1, int& idx2) {
        int n_terms = data.size() / 3;
        std::vector<int> non_zero;
        
        // Collect all non-zero term indices
        for (int i = 0; i < n_terms; ++i) {
            if (data[i*3] != 0) {
                non_zero.push_back(i);
            }
        }
        
        if (non_zero.size() < 2) return false;
        
        // Select two random distinct indices
        unsigned int sample = rng();
        int i1 = sample % non_zero.size();
        int i2 = (sample >> 16) % (non_zero.size() - 1);
        if (i2 >= i1) ++i2;
        
        idx1 = non_zero[i1];
        idx2 = non_zero[i2];
        
        return true;
    }

public:
    explicit Scheme(std::vector<U64> data_in, U64 seed = 42) 
        : data(std::move(data_in)), rng(seed), reduce_cnt(0) {
        
        build_pair_tables();
        
        int n_terms = data.size() / 3;
        
        // Initialize position arrays and free slots
        for (int c = 0; c < 3; ++c) {
            // Pre-allocate some blocks
            int initial_blocks = std::min(100, n_terms / 10 + 1);
            pos[c].reserve(initial_blocks * (BLOCK_SIZE + 1));
            free_slots[c].reserve(initial_blocks);
            
            // Create initial free blocks
            for (int i = 0; i < initial_blocks; ++i) {
                int b = i * (BLOCK_SIZE + 1);
                pos[c].resize(b + BLOCK_SIZE + 1);
                free_slots[c].push_back(b);
            }
        }
        
        // Add all non-zero terms
        for (int i = 0; i < n_terms; ++i) {
            if (data[i*3] == 0) continue;  // Skip zero terms
            for (int j = 0; j < 3; ++j) {
                add(i, j, data[i*3 + j]);
            }
        }
    }
    
    // Add new term (u, v, w)
    void add_term(U64 u, U64 v, U64 w) {
        int n_terms = data.size() / 3;
        int term_idx = -1;
        
        // Find first zero triple or append
        for (int i = 0; i < n_terms; ++i) {
            if (data[i*3] == 0) {
                term_idx = i;
                break;
            }
        }
        
        if (term_idx == -1) {
            term_idx = n_terms;
            data.push_back(u);
            data.push_back(v);
            data.push_back(w);
        } else {
            data[term_idx*3] = u;
            data[term_idx*3 + 1] = v;
            data[term_idx*3 + 2] = w;
        }
        
        // Update structures
        add(term_idx, 0, u);
        add(term_idx, 1, v);
        add(term_idx, 2, w);
    }
    
    // Set component value (term_idx = term index, comp = 0/1/2)
    void set(int term_idx, int comp, U64 new_val) {
        int idx = term_idx * 3 + comp;
        U64 old_val = data[idx];
        
        if (old_val == new_val) return;
        
        // Handle zeroing out entire term
        if (new_val == 0) {
            for (int j = 0; j < 3; ++j) {
                del(term_idx, j, data[term_idx*3 + j]);
                data[term_idx*3 + j] = 0;
            }
            return;
        }
        
        // Update single component
        del(term_idx, comp, old_val);
        data[idx] = new_val;
        add(term_idx, comp, new_val);
    }
    
    // Perform flip operation
    bool flip() {
        if (!sample_pair())
            return false;
        
        // Get components
        int tn = (type + 1) % 3;
        int tp = (type + 2) % 3;
        
        U64 n1 = data[3*j1 + tn];
        U64 n2 = data[3*j2 + tn];
        U64 p1 = data[3*j1 + tp];
        U64 p2 = data[3*j2 + tp];
        
        // Flip
        set(j1, tn, n1 ^ n2);
        set(j2, tp, p1 ^ p2);

        // Try reduction
        // reduce(j1, tn);
        // reduce(j2, tp);

        return true;
    }
    
    // Perform plus-transition operation
    // Replaces {A⊗B⊗C, A'⊗B'⊗C'} with {(A-A')⊗B⊗C, A'⊗B⊗(C+C'), A'⊗(B'-B)⊗C'}
    bool plus() {
        int idx1, idx2;
        
        // Sample any two distinct non-zero terms
        if (!sample_any_pair(idx1, idx2))
            return false;
        
        // Get the components of the two terms
        U64 a1 = data[3*idx1];
        U64 b1 = data[3*idx1 + 1];
        U64 c1 = data[3*idx1 + 2];
        
        U64 a2 = data[3*idx2];
        U64 b2 = data[3*idx2 + 1];
        U64 c2 = data[3*idx2 + 2];
        
        // Check that terms don't share identical components (would cause issues)
        if (a1 == a2 || b1 == b2 || c1 == c2) {
            // Try again with different pair
            return plus();
        }
        
        // Compute new values for the three resulting terms
        // Term 1 (modify idx1): A⊗(B^B')⊗C
        U64 new_b1 = b1 ^ b2;
        
        // Term 2 (modify idx2): A⊗B'⊗(C^C')  
        U64 new_a2 = a1;
        U64 new_c2 = c1 ^ c2;
        
        // Term 3 (new term): (A^A')⊗B'⊗C'
        U64 new_a3 = a1 ^ a2;
        U64 new_b3 = b2;
        U64 new_c3 = c2;
        
        // Update idx1: only change b component
        del(idx1, 1, b1);
        data[3*idx1 + 1] = new_b1;
        add(idx1, 1, new_b1);
        
        // Update idx2: change a and c components
        del(idx2, 0, a2);
        del(idx2, 2, c2);
        data[3*idx2] = new_a2;
        data[3*idx2 + 2] = new_c2;
        add(idx2, 0, new_a2);
        add(idx2, 2, new_c2);
        
        // Add the new third term
        add_term(new_a3, new_b3, new_c3);
        
        return true;
    }


    // Try to reduce term at term_idx after component comp was modified
    bool reduce(int term_idx, int comp) {
        // Skip if term is already zero
        if (data[term_idx*3] == 0) return false;
        
        U64 u = data[term_idx*3];
        U64 v = data[term_idx*3 + 1];
        U64 w = data[term_idx*3 + 2];
        
        int n_terms = data.size() / 3;
        
        // Check against all other non-zero terms
        for (int i = 0; i < n_terms; ++i) {
            if (i == term_idx || data[i*3] == 0) continue;
            
            U64 u2 = data[i*3];
            U64 v2 = data[i*3 + 1];
            U64 w2 = data[i*3 + 2];
            
            // Count matching components
            int matches = (u == u2) + (v == v2) + (w == w2);
            
            // Need at least 2 matches for reduction
            if (matches < 2) continue;
            
            // Find which component differs (or any if all match)
            if (u != u2) {
                set(i, 0, u ^ u2);  // XOR differing component
            } else if (v != v2) {
                set(i, 1, v ^ v2);
            } else {
                set(i, 2, w ^ w2);
            }
            
            // Zero out term_idx
            set(term_idx, 0, 0);
            reduce_cnt++;
            
            return true;
        }
        
        return false;
    }
    
    // Get data reference
    const std::vector<U64>& get_data() const { return data; }
    
    // Get reduction count
    int get_reduce_count() const { return reduce_cnt; }
};