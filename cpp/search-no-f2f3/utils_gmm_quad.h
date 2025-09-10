// utils_gmm_quad.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include "field.h"

// Layout:
// - Input vector f has length 2*n*n:
//     A-block: indices 0 .. n*n-1   (alpha_A = i*n + j)
//     B-block: indices n*n .. 2*n*n-1 (alpha_B = n*n + j*n + k)
// - Output C has length n*n with linear index gamma = i*n + k.

// Generate trivial commutative-quadratic decomposition for C = A * B.
inline std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    data.reserve(static_cast<size_t>(n) * n * n * 3);

    const int nn = n * n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int idxA = i * n + j;        // A[i,j] in f
            const U64 u = 1ULL << idxA;

            for (int k = 0; k < n; ++k) {
                const int idxB = nn + j * n + k; // B[j,k] in f
                const int idxC = i * n + k;      // C[i,k] in output
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

// Extract coefficient at given index from Field
template<typename Field>
inline int get_coefficient(const Field& f, int idx);

// B2: 0/1
template<>
inline int get_coefficient<B2>(const B2& f, int idx) {
    return (f.val >> idx) & 1;
}

// B3: 0/1/2 (two-bit packed)
template<>
inline int get_coefficient<B3>(const B3& f, int idx) {
    int lo_bit = (f.lo >> idx) & 1;
    int hi_bit = (f.hi >> idx) & 1;
    return lo_bit + 2 * hi_bit;
}

// Verify commutative-quadratic scheme for C = A * B using f = (vec(A), vec(B)).
template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    const int r = static_cast<int>(data.size()) / 3;
    const int nn = n * n;
    const int fin = 2 * nn; // dimension of f
    constexpr int mod = field_traits<Field>::is_mod2 ? 2 : 3;

    // Iterate over unordered pairs {alpha, beta} with alpha <= beta
    for (int alpha = 0; alpha < fin; ++alpha) {
        const bool alpha_is_A = (alpha < nn);
        const int a_i = alpha_is_A ? (alpha / n) : ((alpha - nn) / n); // i or j (for B: j)
        const int a_j = alpha_is_A ? (alpha % n) : ((alpha - nn) % n); // j or k (for B: k)

        for (int beta = alpha; beta < fin; ++beta) {
            const bool beta_is_A = (beta < nn);
            const int b_i = beta_is_A ? (beta / n) : ((beta - nn) / n);
            const int b_j = beta_is_A ? (beta % n) : ((beta - nn) % n);

            for (int gamma = 0; gamma < nn; ++gamma) {
                const int gi = gamma / n; // i in C[i,k]
                const int gk = gamma % n; // k in C[i,k]

                // Expected coefficient for unordered monomial {alpha,beta} at output gamma:
                // It is 1 iff the pair corresponds to A[i,j] and B[j,k] (in any order) with j matching.
                int expected = 0;
                if (alpha != beta) {
                    if (alpha_is_A && !beta_is_A) {
                        // alpha = A[i,j], beta = B[j,k]
                        const int Ai = a_i, Aj = a_j;
                        const int Bj = b_i, Bk = b_j;
                        expected = (Aj == Bj && gi == Ai && gk == Bk) ? 1 : 0;
                    } else if (!alpha_is_A && beta_is_A) {
                        // alpha = B[j,k], beta = A[i,j]
                        const int Bj = a_i, Bk = a_j;
                        const int Ai = b_i, Aj = b_j;
                        expected = (Aj == Bj && gi == Ai && gk == Bk) ? 1 : 0;
                    }
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
                        term = u_a * v_a; // will be 0 in our trivial construction
                    } else {
                        term = u_a * v_b + u_b * v_a; // symmetric for unordered pair
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

// Print single Field element with explicit dimension
template<typename Field>
void print_field_dim(const Field& f, int dim) {
    for (int j = 0; j < dim; ++j) {
        std::cout << get_coefficient<Field>(f, j);
    }
}

// Print scheme data in readable form: u (2n^2) | v (2n^2) | w (n^2)
template<typename Field>
void print_scheme(const std::vector<Field>& data, int n) {
    const int n_terms = static_cast<int>(data.size()) / 3;
    const int fin = 2 * n * n;
    const int fout = n * n;
    for (int i = 0; i < n_terms; ++i) {
        if (data[i * 3].is_zero()) continue;
        std::cout << "Term " << i << ": ";
        print_field_dim<Field>(data[i * 3 + 0], fin);
        std::cout << " ";
        print_field_dim<Field>(data[i * 3 + 1], fin);
        std::cout << " ";
        print_field_dim<Field>(data[i * 3 + 2], fout);
        std::cout << "\n";
    }
}

// Rank = number of non-zero terms (by u)
template<typename Field>
int get_rank(const std::vector<Field>& data) {
    const int n_terms = static_cast<int>(data.size()) / 3;
    int rank = 0;
    for (int i = 0; i < n_terms; ++i) {
        if (!data[i * 3].is_zero()) rank++;
    }
    return rank;
}
