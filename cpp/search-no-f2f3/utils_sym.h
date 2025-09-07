// utils_sym.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include "field.h"

// Convert symmetric matrix index (i,j) where i<=j to packed index
inline int sym_to_packed(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    // start(i) = i*n - i*(i-1)/2
    // pack(i,j) = start(i) + (j - i)
    return i * (2 * n - i - 1) / 2 + j;
}

// Convert packed index back to (i,j) pair (i <= j)
inline std::pair<int,int> packed_to_sym(int idx, int n) {
    // Each row i contributes (n - i) pairs: [i,i], [i,i+1], ..., [i,n-1]
    // Walk rows until idx falls into row i, then j = i + offset.
    int i = 0;
    int len = n;          // length of row 0
    int t = idx;
    while (t >= len) {
        t -= len;
        ++i;
        --len;
    }
    int j = i + t;
    return {i, j};
}

// Generate trivial decomposition for symmetric matrix multiplication
// A, B are symmetric (n(n+1)/2 elements each), C = AB is general (n^2 elements)
std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    int sym_dim = n * (n + 1) / 2;

    // Reserve space for n^3 rank-one tensors (u,v,w for each)
    data.reserve(n * n * n * 3);

    // For each (row, col, k) triple in C[row][col] = sum_k A[row][k] * B[k][col]
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int k = 0; k < n; ++k) {
                int idx_A = sym_to_packed(std::min(row, k), std::max(row, k), n);
                int idx_B = sym_to_packed(std::min(k, col), std::max(k, col), n);
                int idx_C = row * n + col;

                U64 u = 1ULL << idx_A;  // u[idx_A] = 1
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

// Verify if the scheme correctly implements symmetric matrix multiplication
// A,B live in Sym_n (dimension s = n(n+1)/2), C lives in M_n (dimension n^2).
template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    const int r = static_cast<int>(data.size()) / 3;
    const int s = n * (n + 1) / 2;

    // Select modulus from Field traits (2 for B2, 3 for B3)
    constexpr int mod = field_traits<Field>::is_mod2 ? 2 : 3;

    // Precompute unpacked pairs for all symmetric indices 0..s-1:
    // idx -> [p,q] with p <= q.
    std::vector<std::pair<int,int>> unpack(s);
    {
        int t = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                unpack[t++] = {i, j};
            }
        }
    }

    // Iterate over all coordinates of the target tensor:
    // alpha in Sym_n^*, beta in Sym_n^*, and (i,k) in M_n.
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const int idxC = i * n + k;

            for (int alpha = 0; alpha < s; ++alpha) {
                const auto [a, b] = unpack[alpha];

                // Determine "middle" index s_mid from alpha containing i.
                // If alpha does not contain i, expected = 0 for all beta.
                int s_mid = -1;
                if (a == i) s_mid = b;
                else if (b == i) s_mid = a;

                for (int beta = 0; beta < s; ++beta) {
                    int expected = 0;
                    if (s_mid != -1) {
                        const auto [c, d] = unpack[beta];
                        // beta must be [s_mid, k] in any order
                        if ((c == s_mid && d == k) || (d == s_mid && c == k)) {
                            expected = 1;
                        }
                    }

                    int sum = 0;

                    // Accumulate sum over all rank-one terms
                    for (int l = 0; l < r; ++l) {
                        const Field& u = data[3 * l];
                        if (u.is_zero()) continue; // skip zero tensors early

                        const Field& v = data[3 * l + 1];
                        const Field& w = data[3 * l + 2];

                        const int uu = get_coefficient<Field>(u, alpha);
                        if (!uu) continue;

                        const int vv = get_coefficient<Field>(v, beta);
                        if (!vv) continue;

                        const int ww = get_coefficient<Field>(w, idxC);
                        if (!ww) continue;

                        // Multiply and add modulo the field
                        sum += uu * vv * ww;
                        if (mod == 2) sum &= 1; else sum %= 3;
                    }

                    if ((sum % mod) != expected) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

// Print single Field element (for debugging) - adapted for symmetric dimensions
template<typename Field>
void print_field(const Field& f, int n, bool is_symmetric);

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

// Print scheme data in readable format for symmetric multiplication
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
        print_field(data[i*3 + 2], n, false); // w - full dim
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
