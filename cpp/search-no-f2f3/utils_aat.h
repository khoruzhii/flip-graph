// utils_AAt.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include "field.h"

// This header provides a trivial rank-1 decomposition for the bilinear map
//   (A, B) ↦ Up(A * B^T) ∈ Sym_n,
// where A, B ∈ M_n are general n×n matrices, and Up(·) denotes taking the
// upper-triangular (including diagonal) part stored in packed form.
// Substituting B = A yields Up(A * A^T), i.e., the desired C = A A^T in Sym_n.
//
// Dimensions:
//   U = M_n  (size n^2) — coordinates (i, k)
//   V = M_n  (size n^2) — coordinates (j, k)  [same k because of B^T]
//   W = Sym_n (size s = n(n+1)/2) — coordinates (i, j), i ≤ j, stored packed
//
// Trivial decomposition has r = s * n = n(n+1)/2 * n terms. Each term corresponds to
// a triple (i, j, k) with 0 ≤ i ≤ j < n and 0 ≤ k < n:
//   u has 1 at (i, k)      → linear index i*n + k
//   v has 1 at (j, k)      → linear index j*n + k
//   w has 1 at pack(i, j)  → symmetric packed index
//
// Verification checks Brent equations for the tensor of Up(A B^T).

// Convert symmetric matrix index (i,j) where i<=j to packed index in [0, n(n+1)/2)
inline int sym_to_packed(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    // Row i contributes (n - i) entries: (i,i), (i,i+1), ..., (i,n-1)
    // Start(i) = i*(2n - i - 1)/2, offset = (j - i)
    return i * (2 * n - i - 1) / 2 + j;
}

// Convert packed index back to (i,j) pair (i <= j)
inline std::pair<int,int> packed_to_sym(int idx, int n) {
    int i = 0;
    int len = n;   // number of entries in row 0
    int t = idx;
    while (t >= len) {
        t -= len;
        ++i;
        --len;
    }
    int j = i + t;
    return {i, j};
}

// Generate trivial decomposition for Up(A B^T).
// A, B live in M_n (n^2 each), C lives in Sym_n (n(n+1)/2).
// Returns binary data as U64 (converted to Field in the Scheme constructor).
inline std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    const int s = n * (n + 1) / 2;
    const int r = s * n;
    data.reserve(static_cast<size_t>(r) * 3);

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            const int idx_W = sym_to_packed(i, j, n);
            const U64 w = 1ULL << idx_W;

            for (int k = 0; k < n; ++k) {
                const int idx_U = i * n + k; // picks A[i,k]
                const int idx_V = j * n + k; // picks B[j,k] due to B^T
                const U64 u = 1ULL << idx_U;
                const U64 v = 1ULL << idx_V;

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

// Verify if the scheme correctly implements Up(A B^T).
// U and V are M_n (dimension n^2), W is Sym_n (dimension s = n(n+1)/2).
template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    const int r = static_cast<int>(data.size()) / 3;
    const int s = n * (n + 1) / 2;

    // Select modulus from Field traits (2 for B2, 3 for B3)
    constexpr int mod = field_traits<Field>::is_mod2 ? 2 : 3;

    // Precompute unpacked pairs for all symmetric indices 0..s-1
    std::vector<std::pair<int,int>> unpack(s);
    {
        int t = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                unpack[t++] = {i, j};
            }
        }
    }

    // Iterate over U* index alpha = (i1,k1), V* index beta = (j1,k2), and W index gamma = (i,j)
    for (int alpha = 0; alpha < n * n; ++alpha) {
        const int i1 = alpha / n;
        const int k1 = alpha % n;

        for (int beta = 0; beta < n * n; ++beta) {
            const int j1 = beta / n;
            const int k2 = beta % n;

            for (int gamma = 0; gamma < s; ++gamma) {
                const auto [i, j] = unpack[gamma];

                // Expected is 1 iff i == i1, j == j1, and k1 == k2
                const int expected = (i == i1 && j == j1 && k1 == k2) ? 1 : 0;

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

// Print single Field element (for debugging) with dimension control
template<typename Field>
void print_field(const Field& f, int n, bool is_symmetric);

template<>
inline void print_field<B2>(const B2& f, int n, bool is_symmetric) {
    const int dim = is_symmetric ? n * (n + 1) / 2 : n * n;
    for (int j = 0; j < dim; ++j) {
        std::cout << ((f.val >> j) & 1);
    }
}

template<>
inline void print_field<B3>(const B3& f, int n, bool is_symmetric) {
    const int dim = is_symmetric ? n * (n + 1) / 2 : n * n;
    for (int j = 0; j < dim; ++j) {
        std::cout << get_coefficient<B3>(f, j);
    }
}

// Print scheme data in readable format for Up(A B^T)
template<typename Field>
void print_scheme(const std::vector<Field>& data, int n) {
    const int n_terms = static_cast<int>(data.size()) / 3;
    for (int i = 0; i < n_terms; ++i) {
        if (data[i * 3].is_zero()) continue;  // Skip zero terms

        std::cout << "Term " << i << ": ";
        print_field(data[i * 3], n, false);     // u  — dimension n^2
        std::cout << " ";
        print_field(data[i * 3 + 1], n, false); // v  — dimension n^2
        std::cout << " ";
        print_field(data[i * 3 + 2], n, true);  // w  — dimension n(n+1)/2
        std::cout << "\n";
    }
}

// Get rank (number of non-zero terms)
template<typename Field>
int get_rank(const std::vector<Field>& data) {
    const int n_terms = static_cast<int>(data.size()) / 3;
    int rank = 0;
    for (int i = 0; i < n_terms; ++i) {
        if (!data[i * 3].is_zero())
            rank++;
    }
    return rank;
}
