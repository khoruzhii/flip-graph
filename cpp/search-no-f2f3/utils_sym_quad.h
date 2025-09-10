// utils_sym_quad.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <string>
#include "field.h"

std::string suffix = "sym_quad";

// Convert symmetric matrix index (i,j) where i<=j to packed index in [0, n(n+1)/2)
inline int sym_to_packed(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    // Row i contributes (n - i) entries: (i,i), (i,i+1), ..., (i,n-1)
    // start(i) = i*(2n - i - 1)/2, pack(i,j) = start(i) + (j - i)
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

// -----------------------------------------------------------------------------
// Trivial commutative-quadratic decomposition for C = A * B with A,B symmetric.
// Variables vector f has length 2*s, s = n(n+1)/2:
//   - A-block: indices [0 .. s-1], position sym_to_packed(i,s) encodes A[i,s]
//   - B-block: indices [s .. 2s-1], position s + sym_to_packed(s,k) encodes B[s,k]
// Output C has length n*n, index gamma = i*n + k.
inline std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    const int s = n * (n + 1) / 2;
    data.reserve(static_cast<size_t>(n) * n * n * 3);

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int mid = 0; mid < n; ++mid) {
                const int idxA_sym = sym_to_packed(i, mid, n);
                const int idxB_sym = sym_to_packed(mid, k, n);
                const int idxA = idxA_sym;          // in f, A-block
                const int idxB = s + idxB_sym;      // in f, B-block
                const int idxC = i * n + k;         // in output

                const U64 u = 1ULL << idxA;
                const U64 v = 1ULL << idxB;
                const U64 w = 1ULL << idxC;

                data.push_back(u);
                data.push_back(v);
                data.push_back(w);
            }
        }
    }
    return data;
}

// -----------------------------------------------------------------------------
// Coefficient extraction
template<typename Field>
inline int get_coefficient(const Field& f, int idx);

// B2: returns 0 or 1
template<>
inline int get_coefficient<B2>(const B2& f, int idx) {
    return (f.val >> idx) & 1;
}

// B3: returns 0, 1, or 2
template<>
inline int get_coefficient<B3>(const B3& f, int idx) {
    int lo_bit = (f.lo >> idx) & 1;
    int hi_bit = (f.hi >> idx) & 1;
    return lo_bit + 2 * hi_bit;  // 00->0, 01->1, 10->2
}

// -----------------------------------------------------------------------------
// Verify commutative-quadratic scheme for symmetric A,B.
// f has dimension fin = 2*s, output C has dimension n*n.
// For an unordered pair {alpha, beta} we use:
//   coeff({alpha,beta}) = u[alpha] v[beta] + u[beta] v[alpha]  if alpha != beta
//                       = u[alpha] v[alpha]                    if alpha == beta
// Expected coefficient at gamma=(i,k) equals 1 iff
//   {alpha,beta} = { A[i,mid], B[mid,k] } for some mid.
template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    const int r = static_cast<int>(data.size()) / 3;
    const int s = n * (n + 1) / 2;
    const int fin = 2 * s;
    constexpr int mod = field_traits<Field>::is_mod2 ? 2 : 3;

    // Precompute symmetric unpack for 0..s-1
    std::vector<std::pair<int,int>> unpack(s);
    {
        int t = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                unpack[t++] = {i, j};
            }
        }
    }

    for (int gamma = 0; gamma < n * n; ++gamma) {
        const int gi = gamma / n; // row
        const int gk = gamma % n; // col

        for (int alpha = 0; alpha < fin; ++alpha) {
            const bool a_is_A = (alpha < s);
            const int a_sym = a_is_A ? alpha : (alpha - s);
            const auto [a0, a1] = unpack[a_sym]; // sorted pair for A or B block

            for (int beta = alpha; beta < fin; ++beta) {
                const bool b_is_A = (beta < s);
                const int b_sym = b_is_A ? beta : (beta - s);
                const auto [b0, b1] = unpack[b_sym];

                int expected = 0;

                if (alpha != beta) {
                    // Case 1: alpha from A, beta from B
                    if (a_is_A && !b_is_A) {
                        // alpha must contain gi, beta must contain gk, and the "other" ends must match
                        // A pair {gi, mid}, B pair {mid, gk}
                        int midA = -1;
                        if (a0 == gi) midA = a1; else if (a1 == gi) midA = a0;
                        if (midA != -1) {
                            if ((b0 == midA && b1 == gk) || (b0 == gk && b1 == midA)) {
                                expected = 1;
                            }
                        }
                    }
                    // Case 2: alpha from B, beta from A
                    else if (!a_is_A && b_is_A) {
                        int midB = -1;
                        if (a0 == gk) midB = a1; else if (a1 == gk) midB = a0;
                        if (midB != -1) {
                            // beta must be {gi, midB}
                            if ((b0 == gi && b1 == midB) || (b0 == midB && b1 == gi)) {
                                expected = 1;
                            }
                        }
                    }
                } else {
                    // alpha == beta, both in same block -> no contribution to AB
                    expected = 0;
                }

                int sum = 0;
                for (int l = 0; l < r; ++l) {
                    const Field& u = data[3 * l + 0];
                    if (u.is_zero()) continue;
                    const Field& v = data[3 * l + 1];
                    const Field& w = data[3 * l + 2];

                    const int ww = get_coefficient<Field>(w, gamma);
                    if (!ww) continue;

                    const int u_a = get_coefficient<Field>(u, alpha);
                    const int v_a = get_coefficient<Field>(v, alpha);
                    const int u_b = get_coefficient<Field>(u, beta);
                    const int v_b = get_coefficient<Field>(v, beta);

                    int term = 0;
                    if (alpha == beta) {
                        term = u_a * v_a;
                    } else {
                        term = u_a * v_b + u_b * v_a;
                    }
                    if (!term) continue;

                    sum += term * ww;
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

// -----------------------------------------------------------------------------
// Printing helpers
template<typename Field>
void print_field_dim(const Field& f, int dim) {
    for (int j = 0; j < dim; ++j) {
        std::cout << get_coefficient<Field>(f, j);
    }
}

template<typename Field>
void print_scheme(const std::vector<Field>& data, int n) {
    const int n_terms = static_cast<int>(data.size()) / 3;
    const int s = n * (n + 1) / 2;
    const int fin = 2 * s;
    const int fout = n * n;
    for (int i = 0; i < n_terms; ++i) {
        if (data[i * 3].is_zero()) continue;
        std::cout << "Term " << i << ": ";
        print_field_dim<Field>(data[i * 3 + 0], fin); // u over f
        std::cout << " ";
        print_field_dim<Field>(data[i * 3 + 1], fin); // v over f
        std::cout << " ";
        print_field_dim<Field>(data[i * 3 + 2], fout); // w over C
        std::cout << "\n";
    }
}

// Rank: number of non-zero u's
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
