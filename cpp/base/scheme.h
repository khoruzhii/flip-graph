// scheme.h
#pragma once

#include <vector>
#include "unordered_dense.h"
#include <array>
#include <random>
#include <cstdint>

using U64 = std::uint64_t;
using U32 = std::uint32_t;


template <typename T>
class indexed_set {
    std::vector<T> items_;
    ankerl::unordered_dense::map<T, int> pos_;

public:
    // Returns false if x already exists; O(1)
    bool insert(const T& x) {
        if (auto [it, ok] = pos_.try_emplace(x, items_.size()); ok) {
            items_.push_back(x);
            return true;
        }
        return false;
    }

    // Erase by value; O(1) via swap-and-pop
    bool erase(const T& x) {
        auto it = pos_.find(x);
        if (it == pos_.end()) return false;
        
        int i = it->second;
        if (i != items_.size() - 1) {
            pos_[items_[i] = std::move(items_.back())] = i;
        }
        items_.pop_back();
        pos_.erase(it);
        return true;
    }

    // O(1) queries
    int size() const noexcept { return items_.size(); }
    bool empty() const noexcept { return items_.empty(); }
    bool contains(const T& x) const { return pos_.count(x); }
    
    // O(1) access
    T& operator[](int i) { return items_[i]; }
    const T& operator[](int i) const { return items_[i]; }
};


class Scheme {
private:
    std::vector<U64> data;
    std::array<indexed_set<U64>, 3> vecs;
    std::array<ankerl::unordered_dense::map<U64, indexed_set<int>>, 3> maps;
    std::mt19937 rng;
    int reduce_cnt;
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
        
        reduce_cnt = 0;
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
        
        // Flip
        set(j1, tn, n1 ^ n2);
        set(j2, tp, p1 ^ p2);

        // Reduction
        reduce(j1, tn);
        reduce(j2, tp);

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
};