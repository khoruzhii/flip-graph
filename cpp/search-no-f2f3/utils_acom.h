// utils_acom.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <string>
#include "field.h"

std::string suffix = "acom";

// Convert symmetric matrix index (i,j) where i<=j to packed index
inline int sym_to_packed(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    return i * (2 * n - i - 1) / 2 + j;
}

// Convert packed index back to (i,j) pair (i <= j)
inline std::pair<int,int> packed_to_sym(int idx, int n) {
    int i = 0;
    int len = n;
    int t = idx;
    while (t >= len) {
        t -= len;
        ++i;
        --len;
    }
    int j = i + t;
    return {i, j};
}

// Generate trivial decomposition for anticommutator C = AB + BA
// A, B, C are all symmetric (n(n+1)/2 elements each)
std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    int sym_dim = n * (n + 1) / 2;
    
    // Reserve space - we'll have 2n^3 terms (n^3 for AB and n^3 for BA)
    data.reserve(2 * n * n * n * 3);
    
    // For each element C[i][j] = sum_k (A[i][k]*B[k][j] + B[i][k]*A[k][j])
    // Since C is symmetric, only generate for i <= j
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            int idx_C = sym_to_packed(i, j, n);
            
            // Add terms from AB
            for (int k = 0; k < n; ++k) {
                int idx_A = sym_to_packed(std::min(i, k), std::max(i, k), n);
                int idx_B = sym_to_packed(std::min(k, j), std::max(k, j), n);
                
                U64 u = 1ULL << idx_A;  // u[idx_A] = 1
                U64 v = 1ULL << idx_B;  // v[idx_B] = 1
                U64 w = 1ULL << idx_C;  // w[idx_C] = 1
                
                data.push_back(u);
                data.push_back(v);
                data.push_back(w);
            }
            
            // Add terms from BA
            for (int k = 0; k < n; ++k) {
                int idx_B = sym_to_packed(std::min(i, k), std::max(i, k), n);
                int idx_A = sym_to_packed(std::min(k, j), std::max(k, j), n);
                
                U64 u = 1ULL << idx_A;  // u[idx_A] = 1 (note: A and B swapped)
                U64 v = 1ULL << idx_B;  // v[idx_B] = 1
                U64 w = 1ULL << idx_C;  // w[idx_C] = 1
                
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

// Verify if the scheme correctly implements anticommutator C = AB + BA
// A, B, C all live in Sym_n (dimension s = n(n+1)/2)
template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    const int r = static_cast<int>(data.size()) / 3;
    const int s = n * (n + 1) / 2;
    
    // Select modulus from Field traits (2 for B2, 3 for B3)
    constexpr int mod = field_traits<Field>::is_mod2 ? 2 : 3;
    
    // Precompute unpacked pairs for all symmetric indices
    std::vector<std::pair<int,int>> unpack(s);
    {
        int t = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                unpack[t++] = {i, j};
            }
        }
    }
    
    // Check Brent equations for C = AB + BA
    // For each triple (alpha, beta, gamma) in Sym_n^*
    for (int alpha = 0; alpha < s; ++alpha) {
        const auto [a, b] = unpack[alpha];
        
        for (int beta = 0; beta < s; ++beta) {
            const auto [c, d] = unpack[beta];
            
            for (int gamma = 0; gamma < s; ++gamma) {
                const auto [e, f] = unpack[gamma];
                
                // Compute expected value for C[e,f] from A[a,b] and B[c,d]
                // C[e,f] gets contribution if:
                // 1) From AB: a=e, b connects to c or d, and the other matches f
                // 2) From BA: c=e, d connects to a or b, and the other matches f
                int expected = 0;
                
                // Check AB contribution: A[a,b] * B[c,d] -> C[e,f]
                // Need: a or b = e, connecting index, other = f
                if (a == e) {
                    // b must connect to c or d, and the other must be f
                    if (b == c && d == f) expected = 1;
                    else if (b == d && c == f) expected = 1;
                } else if (b == e) {
                    // a must connect to c or d, and the other must be f
                    if (a == c && d == f) expected = 1;
                    else if (a == d && c == f) expected = 1;
                }
                
                // Check BA contribution: B[c,d] * A[a,b] -> C[e,f]
                if (c == e) {
                    // d must connect to a or b, and the other must be f
                    if (d == a && b == f) expected = (expected + 1) % mod;
                    else if (d == b && a == f) expected = (expected + 1) % mod;
                } else if (d == e) {
                    // c must connect to a or b, and the other must be f
                    if (c == a && b == f) expected = (expected + 1) % mod;
                    else if (c == b && a == f) expected = (expected + 1) % mod;
                }
                
                // Compute actual value from rank-one decomposition
                int sum = 0;
                for (int l = 0; l < r; ++l) {
                    const Field& u = data[3 * l];
                    if (u.is_zero()) continue;
                    
                    const Field& v = data[3 * l + 1];
                    const Field& w = data[3 * l + 2];
                    
                    const int uu = get_coefficient<Field>(u, alpha);
                    if (!uu) continue;
                    
                    const int vv = get_coefficient<Field>(v, beta);
                    if (!vv) continue;
                    
                    const int ww = get_coefficient<Field>(w, gamma);
                    if (!ww) continue;
                    
                    sum += uu * vv * ww;
                    if (mod == 2) sum &= 1; else sum %= 3;
                }
                
                if ((sum % mod) != expected) {
                    return false;
                }
            }
        }
    }
    
    return true;
}

// Print single Field element (for debugging)
template<typename Field>
void print_field(const Field& f, int n, bool is_symmetric = true);

template<>
void print_field<B2>(const B2& f, int n, bool is_symmetric) {
    int dim = is_symmetric ? n * (n + 1) / 2 : n * n;
    for (int j = 0; j < dim; ++j) {
        std::cout << ((f.val >> j) & 1);
    }
}

template<>
void print_field<B3>(const B3& f, int n, bool is_symmetric) {
    int dim = is_symmetric ? n * (n + 1) / 2 : n * n;
    for (int j = 0; j < dim; ++j) {
        std::cout << get_coefficient<B3>(f, j);
    }
}

// Print scheme data in readable format for anticommutator
template<typename Field>
void print_scheme(const std::vector<Field>& data, int n) {
    int n_terms = data.size() / 3;
    for (int i = 0; i < n_terms; ++i) {
        if (data[i*3].is_zero()) continue;  // Skip zero terms
        
        std::cout << "Term " << i << ": ";
        print_field(data[i*3], n, true);      // u - symmetric dim
        std::cout << " ";
        print_field(data[i*3 + 1], n, true);  // v - symmetric dim
        std::cout << " ";
        print_field(data[i*3 + 2], n, true);  // w - symmetric dim
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