// utils.h - Mod3 C3-symmetric utilities
#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

using U32 = std::uint32_t;
using U64 = std::uint64_t;

// Forward declaration of B3 (should be included from scheme.h)
struct B3;

// Extract trit value from B3 at bit position idx (returns 0, 1, or 2)
inline int get_trit(const B3& b, int idx) {
    int lo_bit = (b.lo >> idx) & 1;
    int hi_bit = (b.hi >> idx) & 1;
    return lo_bit + 2 * hi_bit;  // 00->0, 01->1, 10->2
}

// Set trit value in B3 at bit position idx
inline void set_trit(B3& b, int idx, int val) {
    U64 mask = ~(1ULL << idx);
    b.lo = (b.lo & mask) | ((val & 1) ? (1ULL << idx) : 0);
    b.hi = (b.hi & mask) | ((val >> 1) ? (1ULL << idx) : 0);
}

// Create B3 matrix element for position (r,c) with value val in mod3
inline B3 make_b3_element(int n, int r, int c, int val) {
    B3 result{0, 0};
    int idx = r * n + c;
    set_trit(result, idx, val);
    return result;
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

// ----- Trivial decomposition for RHS = M - T (C3) in mod3 --------------------
//
// M: for all (i,j,k) add a_{i,j} ⊗ b_{j,k} ⊗ c_{k,i}, except i=j=k.
// T: for each part P and ii,jj,kk in P add a_{ii} ⊗ b_{jj} ⊗ c_{kk}, except ii=jj=kk.
// All values are 1 in mod3 (could be extended to support other values)
//
inline std::vector<B3> generate_trivial_decomposition_mod3(
    int n,
    const std::vector<std::vector<int>>& parts
) {
    if (n < 1 || n > 8) {
        throw std::runtime_error("n must be in [1..8] for 64-bit packed matrices");
    }
    // Validate partition
    (void)make_partition_labels(n, parts);

    std::vector<B3> data;
    data.reserve(static_cast<std::size_t>(n) * n * n * 3);

    // 1) M terms, skipping i=j=k
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (i == j && j == k) continue;
                B3 u = make_b3_element(n, i, j, 1); // a_{i,j} = 1
                B3 v = make_b3_element(n, j, k, 1); // b_{j,k} = 1
                B3 w = make_b3_element(n, k, i, 1); // c_{k,i} = 1
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
            B3 u = make_b3_element(n, ii, ii, 1); // a_{ii} = 1
            B3 v = make_b3_element(n, jj, jj, 1); // b_{jj} = 1
            B3 w = make_b3_element(n, kk, kk, 1); // c_{kk} = 1
            data.push_back(u);
            data.push_back(v);
            data.push_back(w);
        }
    }

    return data;
}

// ----- Orbit representatives (C3) for B3 --------------------------------------
//
// Keep exactly one representative per C3-orbit of (u,v,w) under rotation:
// (u,v,w) -> (v,w,u) -> (w,u,v).
// Canonical rule: keep triple if it is lexicographically minimal among its rotations.
//
inline bool lex_leq_triple_b3(const B3& a1, const B3& b1, const B3& c1,
                               const B3& a2, const B3& b2, const B3& c2) {
    // Compare by (hi, lo) lexicographically
    if (a1.hi != a2.hi) return a1.hi < a2.hi;
    if (a1.lo != a2.lo) return a1.lo < a2.lo;
    if (b1.hi != b2.hi) return b1.hi < b2.hi;
    if (b1.lo != b2.lo) return b1.lo < b2.lo;
    if (c1.hi != c2.hi) return c1.hi < c2.hi;
    return c1.lo <= c2.lo;
}

inline std::vector<B3> select_orbit_representatives_mod3(const std::vector<B3>& data) {
    const int r = (int)data.size() / 3;
    std::vector<B3> reps;
    reps.reserve(r);

    for (int i = 0; i < r; ++i) {
        const B3& u = data[3*i + 0];
        const B3& v = data[3*i + 1];
        const B3& w = data[3*i + 2];

        bool is_canonical = true;
        // (u,v,w) <= (v,w,u)
        if (!lex_leq_triple_b3(u, v, w, v, w, u)) is_canonical = false;
        // (u,v,w) <= (w,u,v)
        if (!lex_leq_triple_b3(u, v, w, w, u, v)) is_canonical = false;

        if (is_canonical) {
            reps.push_back(u);
            reps.push_back(v);
            reps.push_back(w);
        }
    }

    return reps;
}

// ----- Verification for C3-symmetric mod3 scheme ------------------------------
//
// Cyclic Brent equations with orbit representatives S (one per C3-orbit):
//
// For all (i1,i2,j1,j2,k1,k2):
//   sum_{s in S} [
//         a(i1,i2) b(j1,j2) c(k1,k2)
//       + a(j1,j2) b(k1,k2) c(i1,i2)  
//       + a(k1,k2) b(i1,i2) c(j1,j2)
//   ] ==  M_expected - T_expected (mod 3)
//
// where
//   M_expected = 1 iff (i2==j1 && j2==k1 && k2==i1), else 0,
//   T_expected = 1 iff (i1==i2, j1==j2, k1==k2) and those indices lie in the same part.
//
inline bool verify_scheme_mod3(const std::vector<B3>& orbit_reps, int n,
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
            const B3& u = orbit_reps[3*l + 0];
            const B3& v = orbit_reps[3*l + 1];
            const B3& w = orbit_reps[3*l + 2];

            // Skip if term is zero
            if (u.is_zero()) continue;

            // First cyclic permutation: (u,v,w)
            int a1 = get_trit(u, i1 * n + i2);
            int b1 = get_trit(v, j1 * n + j2);
            int c1 = get_trit(w, k1 * n + k2);
            
            // Second cyclic permutation: (v,w,u)
            int a2 = get_trit(v, i1 * n + i2);
            int b2 = get_trit(w, j1 * n + j2);
            int c2 = get_trit(u, k1 * n + k2);
            
            // Third cyclic permutation: (w,u,v)
            int a3 = get_trit(w, i1 * n + i2);
            int b3 = get_trit(u, j1 * n + j2);
            int c3 = get_trit(v, k1 * n + k2);

            // Sum all three cyclic contributions (mod 3)
            sum = (sum + a1 * b1 * c1 + a2 * b2 * c2 + a3 * b3 * c3) % 3;
        }

        int M_expected = (i2 == j1 && j2 == k1 && k2 == i1) ? 1 : 0;

        int T_expected = 0;
        if (i1 == i2 && j1 == j2 && k1 == k2) {
            int pi = label[i1];
            int pj = label[j1];
            int pk = label[k1];
            T_expected = (pi == pj && pj == pk) ? 1 : 0;
        }

        // In mod3: expected = M - T (mod 3)
        int expected = (M_expected - T_expected + 3) % 3;
        
        if (sum != expected) return false;
    }

    return true;
}

// Simplified verification without partition (for matrix multiplication only)
inline bool verify_scheme_mod3_simple(const std::vector<B3>& orbit_reps, int n) {
    const int r = (int)orbit_reps.size() / 3;
    if ((int)orbit_reps.size() != 3 * r) return false;

    for (int i1 = 0; i1 < n; ++i1)
    for (int i2 = 0; i2 < n; ++i2)
    for (int j1 = 0; j1 < n; ++j1)
    for (int j2 = 0; j2 < n; ++j2)
    for (int k1 = 0; k1 < n; ++k1)
    for (int k2 = 0; k2 < n; ++k2) {

        int sum = 0;

        for (int l = 0; l < r; ++l) {
            const B3& u = orbit_reps[3*l + 0];
            const B3& v = orbit_reps[3*l + 1];
            const B3& w = orbit_reps[3*l + 2];

            if (u.is_zero()) continue;

            // All three cyclic permutations
            int a1 = get_trit(u, i1 * n + i2);
            int b1 = get_trit(v, j1 * n + j2);
            int c1 = get_trit(w, k1 * n + k2);
            
            int a2 = get_trit(v, i1 * n + i2);
            int b2 = get_trit(w, j1 * n + j2);
            int c2 = get_trit(u, k1 * n + k2);
            
            int a3 = get_trit(w, i1 * n + i2);
            int b3 = get_trit(u, j1 * n + j2);
            int c3 = get_trit(v, k1 * n + k2);

            sum = (sum + a1 * b1 * c1 + a2 * b2 * c2 + a3 * b3 * c3) % 3;
        }

        int expected = (i2 == j1 && j2 == k1 && k2 == i1) ? 1 : 0;
        
        if (sum != expected) return false;
    }

    return true;
}