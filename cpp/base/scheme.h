#pragma once

#include <vector>
#include "unordered_dense.h"
#include <array>
#include <random>
#include <cstdint>
#include "indexed_set.h"

using U64 = std::uint64_t;

class Scheme {
private:
    std::vector<U64> data;
    std::array<indexed_set<U64>, 3> vecs;
    std::array<ankerl::unordered_dense::map<U64, indexed_set<int>>, 3> maps;
    std::mt19937 rng;
    int type, j1, j2;
    
    void add(int term_idx, int comp, U64 val) {
        auto& m = maps[comp][val];
        m.insert(term_idx);
        if (m.size() == 2) vecs[comp].insert(val);
    }
    
    void del(int term_idx, int comp, U64 val) {
        auto it = maps[comp].find(val);
        if (it == maps[comp].end()) return;
        
        auto& m = it->second;
        if (m.size() == 2) vecs[comp].erase(val);
        m.erase(term_idx);
        if (m.empty()) maps[comp].erase(it);
    }
    
    // Sample random pair of terms with duplicate component
    bool sample_pair() {
        int s0 = vecs[0].size();
        int s1 = vecs[1].size();
        int s2 = vecs[2].size();
        if (s0 + s1 + s2 == 0) 
            return false;
        
        int x = rng() % (s0 + s1 + s2);
        int idx = x - (x >= s0) * s0 - (x >= s0 + s1) * s1;
        type = (x >= s0) + (x >= s0 + s1);
        
        U64 val = vecs[type][idx];
        auto& v = maps[type][val];
        int sv = v.size();
        
        int i1 = rng() % sv;
        int i2 = rng() % (sv - 1);
        i2 += (i2 >= i1);
        
        j1 = v[i1];
        j2 = v[i2];
        return true;
    }

public:
    // Initialize from data, optionally with seed
    explicit Scheme(std::vector<U64> data_in, U64 seed = 42) 
        : data(std::move(data_in)), rng(seed) {
        
        int n_terms = data.size() / 3;
        for (int i = 0; i < n_terms; ++i) {
            if (data[i*3] == 0) continue;  // Skip zero terms
            for (int j = 0; j < 3; ++j)
                add(i, j, data[i*3 + j]);
        }
        
    }
    
    // Add new term (u, v, w)
    void add_term(U64 u, U64 v, U64 w) {
        // Find first zero triple or append
        int n_terms = data.size() / 3;
        int term_idx = -1;
        
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
        
        // Update maps and vecs
        add(term_idx, 0, u);
        add(term_idx, 1, v);
        add(term_idx, 2, w);
    }
    
    // Set component value (term_idx = term index, comp = 0/1/2)
    void set(int term_idx, int comp, U64 new_val) {
        int idx = term_idx * 3 + comp;
        U64 old_val = data[idx];
        
        if (old_val == new_val) return;
        
        // Handle zeroing out
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
        
        // Perform flip
        set(j1, tn, n1 ^ n2);
        set(j2, tp, p1 ^ p2);
        return true;
    }
    
    // Get data reference
    const std::vector<U64>& get_data() const { return data; }
};