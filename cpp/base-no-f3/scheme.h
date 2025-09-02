// scheme.h - Mod 3 implementation
#pragma once

#include <vector>
#include "unordered_dense.h"
#include <array>
#include <random>
#include <cstdint>

using U32 = std::uint32_t;
using U64 = std::uint64_t;

// Two-bit packed ternary: per bit position, 0=00, 1=01 (lo=1), 2=10 (hi=1)
struct B3 {
    U64 lo;
    U64 hi;
    
    B3() : lo(0), hi(0) {}
    B3(U64 l, U64 h) : lo(l), hi(h) {}
    
    // Check if zero
    bool is_zero() const { return lo == 0 && hi == 0; }
    
    // Equality
    bool operator==(const B3& other) const {
        return lo == other.lo && hi == other.hi;
    }
    
    bool operator!=(const B3& other) const {
        return !(*this == other);
    }
};

// Mod 3 addition
static inline B3 operator+(const B3& a, const B3& b) {
    const U64 u = ((a.lo | b.lo) & (a.hi | b.hi));
    return B3{
        (a.lo ^ b.lo) ^ (a.hi & b.hi) ^ u,
        (a.hi ^ b.hi) ^ (a.lo & b.lo) ^ u
    };
}

// Mod 3 negation
static inline B3 operator-(const B3& x) { return B3{ x.hi, x.lo }; }

// Mod 3 subtraction
static inline B3 operator-(const B3& a, const B3& b) { return a + (-b); }

// Pack B3 into single U64 for use as map key (assumes n^2 <= 32)
static inline U64 pack_b3(const B3& b) {
    return (b.hi << 32) | (b.lo & 0xFFFFFFFF);
}

// Unpack U64 back to B3
static inline B3 unpack_b3(U64 packed) {
    return B3{ packed & 0xFFFFFFFF, packed >> 32 };
}

// Check if a = k*b (mod 3) for some k ∈ {0,1,2}
// Returns: -1 if not proportional, otherwise returns k
int get_prop_coeff(const B3& a, const B3& b) {
    // If b is zero, then a must be zero (any k works, return 1)
    if (b.is_zero()) {
        return a.is_zero() ? 1 : -1;
    }
    
    // If a is zero and b is not, then k = 0
    if (a.is_zero()) {
        return 0;
    }
    
    // Check k = 1: a == b
    if (a == b) {
        return 1;
    }
    
    // Check k = 2: a == 2*b (which is same as a == -b in mod 3)
    if (a == -b) {
        return 2;
    }
    
    // Not proportional
    return -1;
}

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
    std::vector<B3> data;  // Stores triplets: [u0,v0,w0, u1,v1,w1, ...]
    std::array<indexed_set<U64>, 3> vecs;  // Store packed B3 values
    std::array<ankerl::unordered_dense::map<U64, indexed_set<int>>, 3> maps;
    std::mt19937 rng;
    int type, j1, j2;
    static constexpr int next[3] = {1, 2, 0};
    static constexpr int prev[3] = {2, 0, 1};
    
    void add(int term_idx, int comp, const B3& val) {
        if (val.is_zero()) return;
        
        U64 packed = pack_b3(val);
        auto& m = maps[comp][packed];
        m.insert(term_idx);
        if (m.size() == 2) vecs[comp].insert(packed);
    }
    
    void del(int term_idx, int comp, const B3& val) {
        if (val.is_zero()) return;
        
        U64 packed = pack_b3(val);
        auto it = maps[comp].find(packed);
        if (it == maps[comp].end()) return;
        
        auto& m = it->second;
        if (m.size() == 2) vecs[comp].erase(packed);
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
        
        U64 packed_val = vecs[type][idx];
        auto& v = maps[type][packed_val];
        int sv = v.size();
        
        int i1 = rng() % sv;
        int i2 = rng() % (sv - 1);
        i2 += (i2 >= i1);
        
        j1 = v[i1];
        j2 = v[i2];
        return true;
    }

public:
    // Initialize from data (U64 triplets), optionally with seed
    // For compatibility, still accept U64 vector but convert to B3
    explicit Scheme(std::vector<U64> data_in, U64 seed = 42) 
        : rng(seed) {
        
        int n_terms = data_in.size() / 3;
        data.resize(n_terms * 3);  // Allocate space for all triplets
        
        for (int i = 0; i < n_terms; ++i) {
            // Convert each U64 component to B3 (assuming binary input, so hi=0)
            data[i*3]     = B3{ data_in[i*3],     0 };  // u component
            data[i*3 + 1] = B3{ data_in[i*3 + 1], 0 };  // v component  
            data[i*3 + 2] = B3{ data_in[i*3 + 2], 0 };  // w component
            
            // Add to maps if not zero
            if (!data[i*3].is_zero()) {
                for (int j = 0; j < 3; ++j) {
                    add(i, j, data[i*3 + j]);
                }
            }
        }
    }
    
    // Alternative constructor directly with B3 data
    explicit Scheme(std::vector<B3> data_b3, U64 seed = 42)
        : data(std::move(data_b3)), rng(seed) {
        
        int n_terms = data.size() / 3;
        
        for (int i = 0; i < n_terms; ++i) {
            if (!data[i*3].is_zero()) {
                for (int j = 0; j < 3; ++j)
                    add(i, j, data[i*3 + j]);
            }
        }
    }
    
    // Add new term (u, v, w) as B3 values
    void add_term(const B3& u, const B3& v, const B3& w) {
        // Find first zero triple or append
        int n_terms = data.size() / 3;
        int term_idx = -1;
        
        for (int i = 0; i < n_terms; ++i) {
            if (data[i*3].is_zero()) {
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
    void set(int term_idx, int comp, const B3& new_val) {
        int idx = term_idx * 3 + comp;
        B3 old_val = data[idx];
        
        if (old_val == new_val) return;
        
        // Handle zeroing out
        if (new_val.is_zero()) {
            for (int j = 0; j < 3; ++j) {
                del(term_idx, j, data[term_idx*3 + j]);
                data[term_idx*3 + j] = B3{0, 0};
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
        int tn = next[type];
        int tp = prev[type];
        
        B3 n1 = data[3*j1 + tn];
        B3 n2 = data[3*j2 + tn];
        B3 p1 = data[3*j1 + tp];
        B3 p2 = data[3*j2 + tp];
        
        // Flip using mod 3 arithmetic
        set(j1, tn, n1 - n2);
        set(j2, tp, p1 + p2);

        // Try reductions
        reduce(j1, tn);
        reduce(j2, tp);

        return true;
    }

    // Try to reduce term at term_idx after component comp was modified
    bool reduce(int term_idx, int comp) {
        // Skip if term is already zero
        if (data[term_idx*3].is_zero()) return false;
        
        B3 t1[3] = {data[term_idx*3], data[term_idx*3 + 1], data[term_idx*3 + 2]};
        
        int n_terms = data.size() / 3;
        
        // Check against all other non-zero terms
        for (int i = 0; i < n_terms; ++i) {
            if (i == term_idx || data[i*3].is_zero()) continue;
            
            B3 t2[3] = {data[i*3], data[i*3 + 1], data[i*3 + 2]};
            
            // Get proportionality coefficients for each component pair
            int k[3] = {get_prop_coeff(t1[0], t2[0]), 
                        get_prop_coeff(t1[1], t2[1]), 
                        get_prop_coeff(t1[2], t2[2])};
            

            // Need at least 2 proportional components for reduction
            if ((k[0] > 0) + (k[1] > 0) + (k[2] > 0) < 2) continue;
            
            // Find which component differs (first negative, or 2 if all positive)
            int diff_comp = 2;
            for (int j = 0; j < 3; ++j) {
                if (k[j] < 0) {
                    diff_comp = j;
                    break;
                }
            }
            
            // Calculate product of the two proportional coefficients
            int coeff_product = (k[next[diff_comp]] * k[prev[diff_comp]]) % 3;
            
            // Apply reduction based on the coefficient
            B3 new_val = (coeff_product == 1) ? t1[diff_comp] + t2[diff_comp] 
                                              : t2[diff_comp] - t1[diff_comp];
            set(i, diff_comp, new_val);
            
            // Zero out term_idx
            set(term_idx, 0, B3{0, 0});
            
            return true;
        }
        
        return false;
    }
    
    // Get data reference - return as B3 vector
    const std::vector<B3>& get_data_b3() const { return data; }
    
    // Get data as U64 vector for compatibility (returns lo and hi separately)
    std::vector<U64> get_data() const {
        std::vector<U64> result;
        result.reserve(data.size());
        for (const auto& b3 : data) {
            result.push_back(b3.lo);
        }
        return result;
    }
    
    // Get rank (number of non-zero terms)
    int get_rank() const {
        int n_terms = data.size() / 3;
        int rank = 0;
        for (int i = 0; i < n_terms; ++i) {
            if (!data[i*3].is_zero())
                rank++;
        }
        return rank;
    }
};


// {(u,v,w1),(2u,v,w2)} -> {(u,v,w1),(u,v,w2+w2)} -> (u,v,w1+w2+w2)

