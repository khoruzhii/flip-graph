// utils_gmm.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include "field.h"

// Generate trivial decomposition for n×n matrix multiplication
// Returns binary data as U64 (will be converted to Field in Scheme constructor)
std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    data.reserve(n * n * n * 3);
    
    // For each (i,j,k) triple, add rank-one tensor:
    // u[i,j] ⊗ v[j,k] ⊗ w[k,i]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                U64 u = 1ULL << (i * n + j); // u[i,j]
                U64 v = 1ULL << (j * n + k); // v[j,k]
                U64 w = 1ULL << (k * n + i); // w[k,i]
                
                data.push_back(u);
                data.push_back(v);
                data.push_back(w);
            }
        }
    }
    
    return data;
}

// Extract coefficient at given index from Field
template<typename Field>
inline int get_coefficient(const Field& f, int idx);

// Specialization for B2: returns 0 or 1
template<>
inline int get_coefficient<B2>(const B2& f, int idx) {
    return (f.val >> idx) & 1;
}

// Specialization for B3: returns 0, 1, or 2
template<>
inline int get_coefficient<B3>(const B3& f, int idx) {
    int lo_bit = (f.lo >> idx) & 1;
    int hi_bit = (f.hi >> idx) & 1;
    return lo_bit + 2 * hi_bit;  // 00->0, 01->1, 10->2
}

// Verify if the scheme correctly implements matrix multiplication
template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    // Number of rank-one tensors
    int r = data.size() / 3;
    
    // Determine modulo based on Field type
    constexpr int mod = field_traits<Field>::is_mod2 ? 2 : 3;
    
    // Check each Brent equation
    for (int i1 = 0; i1 < n; ++i1) {
        for (int i2 = 0; i2 < n; ++i2) {
            for (int j1 = 0; j1 < n; ++j1) {
                for (int j2 = 0; j2 < n; ++j2) {
                    for (int k1 = 0; k1 < n; ++k1) {
                        for (int k2 = 0; k2 < n; ++k2) {
                            // Compute sum over all rank-one tensors
                            int sum = 0;
                            
                            for (int l = 0; l < r; ++l) {
                                const Field& u = data[3 * l];
                                const Field& v = data[3 * l + 1]; 
                                const Field& w = data[3 * l + 2];
                                
                                // Skip if term is zero
                                if (u.is_zero()) continue;
                                
                                int a = get_coefficient<Field>(u, i1 * n + i2);
                                int b = get_coefficient<Field>(v, j1 * n + j2);
                                int c = get_coefficient<Field>(w, k1 * n + k2);
                                
                                // Add product modulo 2 or 3
                                sum = (sum + a * b * c) % mod;
                            }
                            
                            int expected = (i2 == j1 && j2 == k1 && k2 == i1) ? 1 : 0;
                            if (sum != expected) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return true;
}

// Print single Field element (for debugging)
template<typename Field>
void print_field(const Field& f, int n);

template<>
void print_field<B2>(const B2& f, int n) {
    for (int j = 0; j < n*n; ++j) {
        std::cout << ((f.val >> j) & 1);
    }
}

template<>
void print_field<B3>(const B3& f, int n) {
    for (int j = 0; j < n*n; ++j) {
        std::cout << get_coefficient<B3>(f, j);
    }
}

// Print scheme data in readable format
template<typename Field>
void print_scheme(const std::vector<Field>& data, int n) {
    int n_terms = data.size() / 3;
    for (int i = 0; i < n_terms; ++i) {
        if (data[i*3].is_zero()) continue;  // Skip zero terms
        
        std::cout << "Term " << i << ": ";
        print_field(data[i*3], n);
        std::cout << " ";
        print_field(data[i*3 + 1], n);
        std::cout << " ";
        print_field(data[i*3 + 2], n);
        std::cout << "\n";
    }
}

// Get rank (number of non-zero terms)
template<typename Field>
int get_rank(const std::vector<Field>& data) {
    int n_terms = data.size() / 3;
    int rank = 0;
    for (int i = 0; i < n_terms; ++i) {
        if (!data[i*3].is_zero())
            rank++;
    }
    return rank;
}

// // Simple timing utility
// class Timer {
//     using clock = std::chrono::steady_clock;
//     clock::time_point start;
    
// public:
//     Timer() : start(clock::now()) {}
    
//     double elapsed_ms() const {
//         auto end = clock::now();
//         return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     }
    
//     double elapsed_s() const {
//         return elapsed_ms() / 1000.0;
//     }
    
//     void reset() {
//         start = clock::now();
//     }
// };