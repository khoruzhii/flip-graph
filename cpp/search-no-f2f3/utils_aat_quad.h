// utils_aat_quad.h
#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include "field.h"
#include <string>

// This header provides a trivial rank-1 decomposition for the *commutative quadratic* map
//   A ↦ Up(A * A^T) ∈ Sym_n,
// where A ∈ M_n, and Up(·) denotes taking the upper-triangular (including diagonal) part
// stored in packed form.
//
// Coordinates:
//   a lives in M_n (size n^2) — linear index alpha = i*n + k
//   c lives in Sym_n (size s = n(n+1)/2) — index gamma = pack(i,j) for i ≤ j
//
// Target polynomial:
//   c_{i,j} = sum_{k=0}^{n-1} a_{i,k} * a_{j,k}  for i ≤ j.
//
// Trivial decomposition has r = s * n = n(n+1)/2 * n terms. Each term corresponds to a triple
// (i, j, k) with 0 ≤ i ≤ j < n and 0 ≤ k < n:
//   u has 1 at (i, k)      → linear index i*n + k
//   v has 1 at (j, k)      → linear index j*n + k
//   w has 1 at pack(i, j)  → symmetric packed index
//
// Because variables commute (quadratic commutative scheme), swapping u and v in any term
// yields an equivalent scheme. Verification below respects this symmetry.

 std::string suffix = "aat_quad";

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

// Generate trivial decomposition for the commutative quadratic map Up(A A^T).
// Input A lives in M_n (n^2), output C lives in Sym_n (n(n+1)/2).
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
                const int idx_U = i * n + k; // picks a_{i,k}
                const int idx_V = j * n + k; // picks a_{j,k} (same k)
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

// ----------------------------------------------------------------------------------
// Coefficient extraction per Field

template<typename Field>
inline int get_coefficient(const Field& f, int idx);

// B2: returns 0 or 1
template<>
inline int get_coefficient<B2>(const B2& f, int idx) {
    return (f.val >> idx) & 1;
}

// B3: returns 0, 1, or 2 (two-bit packed: lo + 2*hi)
template<>
inline int get_coefficient<B3>(const B3& f, int idx) {
    int lo_bit = (f.lo >> idx) & 1;
    int hi_bit = (f.hi >> idx) & 1;
    return lo_bit + 2 * hi_bit;
}

// B4: returns 0..3 (two-bit packed: lo + 2*hi)
template<>
inline int get_coefficient<B4>(const B4& f, int idx) {
    int lo_bit = (f.lo >> idx) & 1;
    int hi_bit = (f.hi >> idx) & 1;
    return lo_bit + 2 * hi_bit;
}

// Modulus traits (do not require changes to field.h)
template<typename Field>
struct modulus_traits;

template<> struct modulus_traits<B2> { static constexpr int value = 2; };
template<> struct modulus_traits<B3> { static constexpr int value = 3; };
template<> struct modulus_traits<B4> { static constexpr int value = 4; };

// ----------------------------------------------------------------------------------
// Verify if the scheme implements the commutative quadratic map Up(A A^T).
// a lives in M_n (dimension n^2), c in Sym_n (dimension s = n(n+1)/2).
// We compare coefficients of unordered monomials {alpha, beta}:
//   - if alpha == beta: coeff = u[alpha] * v[alpha]
//   - if alpha < beta:  coeff = u[alpha] * v[beta] + u[beta] * v[alpha]
template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    const int r = static_cast<int>(data.size()) / 3;
    const int s = n * (n + 1) / 2;

    constexpr int mod = modulus_traits<Field>::value;

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

    // Iterate over unordered pairs {alpha, beta} with alpha <= beta, and gamma in Sym_n
    const int dim = n * n;
    for (int alpha = 0; alpha < dim; ++alpha) {
        const int i1 = alpha / n;
        const int k1 = alpha % n;

        for (int beta = alpha; beta < dim; ++beta) {
            const int j1 = beta / n;
            const int k2 = beta % n;

            for (int gamma = 0; gamma < s; ++gamma) {
                const auto [i, j] = unpack[gamma];

                // Expected coefficient for {alpha, beta} at output gamma
                int expected = 0;
                if (k1 == k2) {
                    if (i == j) {
                        // Diagonal output: need i1 == j1 == i and k1 == k2
                        expected = (i1 == i && j1 == i) ? 1 : 0;
                    } else {
                        // Off-diagonal output: {i1, j1} must equal {i, j}
                        const bool case1 = (i1 == i && j1 == j);
                        const bool case2 = (i1 == j && j1 == i);
                        expected = (case1 || case2) ? 1 : 0;
                    }
                }

                int sum = 0;

                for (int l = 0; l < r; ++l) {
                    const Field& u = data[3 * l];
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
                        term = u_a * v_a; // diagonal monomial
                    } else {
                        term = u_a * v_b + u_b * v_a; // symmetric sum for unordered pair
                    }
                    if (!term) continue;

                    sum += term * ww;
                    if constexpr (mod == 2) { sum &= 1; }
                    else if constexpr (mod == 3) { sum %= 3; }
                    else if constexpr (mod == 4) { sum &= 3; }
                }

                if ((sum % mod) != expected) {
                    return false;
                }
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------------------------
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

template<>
inline void print_field<B4>(const B4& f, int n, bool is_symmetric) {
    const int dim = is_symmetric ? n * (n + 1) / 2 : n * n;
    for (int j = 0; j < dim; ++j) {
        std::cout << get_coefficient<B4>(f, j);
    }
}

// Print scheme data in readable format (u | v | w)
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
