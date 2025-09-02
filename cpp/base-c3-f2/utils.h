#pragma once

// Utilities for C3-symmetric trivial decomposition, orbit-representative extraction,
// and verification that EXPECTS ORBIT REPRESENTATIVES.
// Comments are in English.
//
// Public API:
//   std::vector<U64> generate_trivial_decomposition(int n,
//                                                   const std::vector<std::vector<int>>& parts);
//   std::vector<U64> select_orbit_representatives(const std::vector<U64>& data);
//   bool verify_scheme(const std::vector<U64>& orbit_reps, int n,
//                      const std::vector<std::vector<int>>& parts);
//
// Notes:
// - Matrices are packed into U64 with row-major bit layout: bit (r*n + c).
// - n must satisfy n*n <= 64 (i.e., n <= 8).
// - 'parts' is a diagonal partition of {0,...,n-1}: disjoint parts that cover all indices.
// - generate_trivial_decomposition builds a sparse decomposition for RHS = M - T over F2,
//   without duplicates, by skipping exactly (i=j=k) in both M and T.
// - select_orbit_representatives keeps one (lexicographically minimal) representative
//   per C3-orbit: (u,v,w) ~ (v,w,u) ~ (w,u,v).
// - verify_scheme expects *orbit representatives* and uses the cyclic Brent equations.

#include <vector>
#include <cstdint>
#include <stdexcept>

using U32 = std::uint32_t;
using U64 = std::uint64_t;

// ----- Bit helpers ------------------------------------------------------------

// Bit for entry (r,c) in an n x n matrix, row-major.
inline U64 bit_at(int n, int r, int c) {
    return 1ULL << (r * n + c);
}

inline int get_bit(U64 m, int n, int r, int c) {
    return static_cast<int>((m >> (r * n + c)) & 1ULL);
}

// ----- Partition labels -------------------------------------------------------

inline std::vector<int> make_partition_labels(int n,
                                              const std::vector<std::vector<int>>& parts) {
    std::vector<int> label(n, -1);
    for (int pid = 0; pid < (int)parts.size(); ++pid) {
        for (int x : parts[pid]) {
            if (x < 0 || x >= n) throw std::runtime_error("partition index out of range");
            if (label[x] != -1)  throw std::runtime_error("partition parts must be disjoint");
            label[x] = pid;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (label[i] == -1) throw std::runtime_error("partition must cover all indices");
    }
    return label;
}

// ----- Trivial decomposition for RHS = M - T (C3) -----------------------------
//
// M: for all (i,j,k) add a_{i,j} ⊗ b_{j,k} ⊗ c_{k,i}, except i=j=k.
// T: for each part P and ii,jj,kk in P add a_{ii} ⊗ b_{jj} ⊗ c_{kk}, except ii=jj=kk.
// Skipping these precisely cancels the overlap and avoids duplicates.
//
inline std::vector<U64> generate_trivial_decomposition(
    int n,
    const std::vector<std::vector<int>>& parts
) {
    if (n < 1 || n > 8) {
        throw std::runtime_error("n must be in [1..8] for 64-bit packed matrices");
    }
    // Validate partition
    (void)make_partition_labels(n, parts);

    std::vector<U64> data;
    data.reserve(static_cast<std::size_t>(n) * n * n * 3); // rough upper bound

    // 1) M terms, skipping i=j=k
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (i == j && j == k) continue;
                U64 u = bit_at(n, i, j); // a_{i,j}
                U64 v = bit_at(n, j, k); // b_{j,k}
                U64 w = bit_at(n, k, i); // c_{k,i}
                data.push_back(u);
                data.push_back(v);
                data.push_back(w);
            }
        }
    }

    // 2) T terms (diagonal), skipping ii=jj=kk
    for (const auto& P : parts) {
        for (int ii : P) for (int jj : P) for (int kk : P) {
            if (ii == jj && jj == kk) continue;
            U64 u = bit_at(n, ii, ii); // a_{ii}
            U64 v = bit_at(n, jj, jj); // b_{jj}
            U64 w = bit_at(n, kk, kk); // c_{kk}
            data.push_back(u);
            data.push_back(v);
            data.push_back(w);
        }
    }

    return data;
}

// ----- Orbit representatives (C3) ---------------------------------------------
//
// Keep exactly one representative per C3-orbit of (u,v,w) under rotation:
// (u,v,w) -> (v,w,u) -> (w,u,v).
// Canonical rule: keep triple if it is lexicographically minimal among its rotations.
//
inline bool lex_leq_triple(U64 a1, U64 b1, U64 c1, U64 a2, U64 b2, U64 c2) {
    if (a1 != a2) return a1 < a2;
    if (b1 != b2) return b1 < b2;
    return c1 <= c2;
}

inline std::vector<U64> select_orbit_representatives(const std::vector<U64>& data) {
    const int r = (int)data.size() / 3;
    std::vector<U64> reps;
    reps.reserve(r); // at most r reps

    for (int i = 0; i < r; ++i) {
        U64 u = data[3*i + 0];
        U64 v = data[3*i + 1];
        U64 w = data[3*i + 2];

        bool is_canonical = true;
        // (u,v,w) <= (v,w,u)
        if (!lex_leq_triple(u, v, w, v, w, u)) is_canonical = false;
        // (u,v,w) <= (w,u,v)
        if (!lex_leq_triple(u, v, w, w, u, v)) is_canonical = false;

        if (is_canonical) {
            reps.push_back(u);
            reps.push_back(v);
            reps.push_back(w);
        }
    }

    return reps;
}

// ----- Verification (expects orbit representatives) ---------------------------
//
// Cyclic Brent equations with orbit representatives S (one per C3-orbit):
//
// For all (i1,i2,j1,j2,k1,k2):
//   sum_{s in S} [
//         a(i1,i2) b(j1,j2) c(k1,k2)
//       + a(j1,j2) b(k1,k2) c(i1,i2)
//       + a(k1,k2) b(i1,i2) c(j1,j2)
//   ] ==  M_expected XOR T_expected
//
// where
//   M_expected = 1 iff (i2==j1 && j2==k1 && k2==i1), else 0,
//   T_expected = 1 iff (i1==i2, j1==j2, k1==k2) and those indices lie in the same part.
//
inline bool verify_scheme(const std::vector<U64>& orbit_reps, int n,
                          const std::vector<std::vector<int>>& parts) {
    const int r = (int)orbit_reps.size() / 3;
    if ((int)orbit_reps.size() != 3 * r) return false;

    auto label = make_partition_labels(n, parts);

    for (int i1 = 0; i1 < n; ++i1)
    for (int i2 = 0; i2 < n; ++i2)
    for (int j1 = 0; j1 < n; ++j1)
    for (int j2 = 0; j2 < n; ++j2)
    for (int k1 = 0; k1 < n; ++k1)
    for (int k2 = 0; k2 < n; ++k2) {

        int sum = 0;

        for (int l = 0; l < r; ++l) {
            U64 u = orbit_reps[3*l + 0];
            U64 v = orbit_reps[3*l + 1];
            U64 w = orbit_reps[3*l + 2];

            int t1 = get_bit(u, n, i1, i2) & get_bit(v, n, j1, j2) & get_bit(w, n, k1, k2);
            int t2 = get_bit(u, n, j1, j2) & get_bit(v, n, k1, k2) & get_bit(w, n, i1, i2);
            int t3 = get_bit(u, n, k1, k2) & get_bit(v, n, i1, i2) & get_bit(w, n, j1, j2);

            sum ^= (t1 ^ t2 ^ t3);
        }

        int M_expected = (i2 == j1 && j2 == k1 && k2 == i1) ? 1 : 0;

        int T_expected = 0;
        if (i1 == i2 && j1 == j2 && k1 == k2) {
            int pi = label[i1];
            int pj = label[j1];
            int pk = label[k1];
            T_expected = (pi == pj && pj == pk) ? 1 : 0;
        }

        int expected = M_expected ^ T_expected;
        if (sum != expected) return false;
    }

    return true;
}
