// scheme.h - Mod3 C3-symmetric implementation
#pragma once

#include <vector>
#include <random>
#include <cstdint>
#include <cassert>
#include "hash_dict.h"

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

class Scheme {
public:
    // Initialize with B3 data and optional seed
    explicit Scheme(const std::vector<B3>& initial, uint32_t seed = 42);
    
    // Perform flip operation
    bool flip();
    
    // Get data reference
    const std::vector<B3>& get_data() const { return data; }
    
    // Get rank (number of non-zero terms) and orbit rank
    int get_rank() const { return rank; }
    int get_orank() const { return rank / 3; }  // C3 symmetry
    
private:
    // Add/remove row index r to/from the block of value v
    void add(int r, U64 v);
    void del(int r, U64 v);
    
    // Set component value (term_idx = term index, comp = 0/1/2)
    void set(int term_idx, int comp, const B3& new_val);
    
private:
    int n{0};        // Total size of data array
    int rank{0};     // Number of non-zero components
    std::mt19937 rng{42};
    
    std::vector<B3> data;      // u,v,w repeating per triple
    std::vector<int> idx_next; // v index per row
    std::vector<int> idx_prev; // w index per row
    
    HashDict unique;
    HashDict flippable_idx;
    std::vector<U64> flippable;  // Packed B3 values
    
    // For each distinct value keep a block of length (n+1): [len, idx1, idx2, ..., idx_len]
    std::vector<int> pos;
    std::vector<int> free_slots;
    
    // 0 if i and j are in the same orbit; 1 otherwise
    std::vector<std::vector<uint8_t>> permit;
    
    // Pair selection tables
    std::vector<int> pair_starts;
    std::vector<int> pair_i;
    std::vector<int> pair_j;
};

// Implementation

inline Scheme::Scheme(const std::vector<B3>& initial, uint32_t seed)
    : rng(seed) {
    
    data = initial;
    n = static_cast<int>(data.size());
    assert(n % 3 == 0);
    
    // Build index helpers for v and w components per row
    idx_next.assign(n, 0);
    idx_prev.assign(n, 0);
    for (int i = 0; i < n; i += 3) {
        idx_next[i]     = i + 1; idx_prev[i]     = i + 2;  // u -> v, u -> w
        idx_next[i + 1] = i + 2; idx_prev[i + 1] = i;      // v -> w, v -> u
        idx_next[i + 2] = i;     idx_prev[i + 2] = i + 1;  // w -> u, w -> v
    }
    
    // Permit matrix: 0 if same orbit (C3 symmetry, 3 components per orbit)
    permit.assign(n, std::vector<uint8_t>(n, 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            permit[i][j] = (i / 3 == j / 3) ? 0 : 1;
        }
    }
    
    // Reset dictionaries and lists
    unique = HashDict();
    flippable_idx = HashDict();
    flippable.clear();
    
    // Allocate position blocks
    const int block_len = n + 1;
    pos.assign(n * block_len, 0);
    free_slots.clear();
    free_slots.reserve(n);
    for (int i = 0; i < n; ++i) {
        int b = i * block_len;
        free_slots.push_back(b);
    }
    
    // Initial fill of unique/flippable based on current data
    rank = 0;
    for (int i = 0; i < n; ++i) {
        const B3& val = data[i];
        if (val.is_zero()) continue;
        
        U64 packed = pack_b3(val);
        
        if (unique.contains(packed)) {
            int b = unique.getvaluex(packed);
            int l = pos[b];
            ++l;
            pos[b + l] = i;
            pos[b] = l;
            if (!flippable_idx.contains(packed)) {
                flippable_idx.addx(packed, static_cast<int>(flippable.size()));
                flippable.push_back(packed);
            }
        } else {
            int b = free_slots.back(); 
            free_slots.pop_back();
            unique.addx(packed, b);
            pos[b] = 1;
            pos[b + 1] = i;
        }
        rank += 1;
    }
    
    // Build pair selection tables
    pair_starts.clear(); 
    pair_i.clear(); 
    pair_j.clear();
    pair_starts.reserve(100);
    pair_i.reserve(6400);
    pair_j.reserve(6400);
    pair_starts.push_back(0);
    pair_starts.push_back(0);
    for (int x = 1; x < 80; ++x) {
        for (int y = 0; y < x; ++y) {
            pair_i.push_back(x);
            pair_j.push_back(y);
            pair_i.push_back(y);
            pair_j.push_back(x);
        }
        pair_starts.push_back(static_cast<int>(pair_i.size()));
    }
}

inline void Scheme::del(int r, U64 v) {
    int b = unique.getvalue(v);
    int l = pos[b];
    if (l == 2) {
        // Will drop from 2 -> 1; remove from flippable
        flippable_idx.lasthash = unique.lasthash;
        int idx = flippable_idx.getvaluex(v);
        U64 last_val = flippable.back();
        flippable_idx.replace(last_val, idx);
        flippable[idx] = last_val;
        flippable.pop_back();
        flippable_idx.lasthash = unique.lasthash;
        flippable_idx.removex(v);
    }
    if (l == 1) {
        // Now 0 -> release the block and drop from unique
        free_slots.push_back(b);
        unique.removex(v);
    } else {
        // Remove row r from the block by rotating from the end
        int i = b + l;
        int x = pos[i];
        while (x != r) {
            --i;
            int y = x;
            x = pos[i];
            pos[i] = y;
        }
        pos[b] = l - 1;
    }
}

inline void Scheme::add(int r, U64 v) {
    int present = unique.contains(v);
    if (present) {
        int b = unique.getvaluex(v);
        int l = pos[b];
        if (l == 1) {
            // Will become multiplicity 2 -> insert into flippable
            flippable_idx.lasthash = unique.lasthash;
            flippable_idx.addx(v, static_cast<int>(flippable.size()));
            flippable.push_back(v);
        }
        ++l;
        pos[b + l] = r;
        pos[b] = l;
    } else {
        int b = free_slots.back(); 
        free_slots.pop_back();
        unique.addx(v, b);
        pos[b + 1] = r;
        pos[b] = 1;
    }
}

inline void Scheme::set(int term_idx, int comp, const B3& new_val) {
    int idx = term_idx * 3 + comp;
    B3 old_val = data[idx];
    
    if (old_val == new_val) return;
    
    // Handle zeroing out entire term
    if (new_val.is_zero()) {
        for (int j = 0; j < 3; ++j) {
            if (!data[term_idx * 3 + j].is_zero()) {
                del(term_idx * 3 + j, pack_b3(data[term_idx * 3 + j]));
                data[term_idx * 3 + j] = B3{0, 0};
                rank -= 1;
            }
        }
        return;
    }
    
    // Update single component
    if (!old_val.is_zero()) {
        del(idx, pack_b3(old_val));
        rank -= 1;
    }
    data[idx] = new_val;
    add(idx, pack_b3(new_val));
    rank += 1;
}

inline bool Scheme::flip() {
    if (flippable.empty()) return false;
    
    while (true) {
        unsigned int sample = rng();
        U64 packed_val = flippable[sample % flippable.size()];
        int b = unique.getvalue(packed_val);
        int l = pos[b];
        ++b; // point to first index entry
        
        int p, q;
        if (l == 2) {
            if (sample & 65536U) {
                p = pos[b];
                q = pos[b + 1];
            } else {
                p = pos[b + 1];
                q = pos[b];
            }
        } else {
            int x = static_cast<int>((sample >> 16) % pair_starts[l]);
            p = pos[b + pair_i[x]];
            q = pos[b + pair_j[x]];
        }
        
        if (!permit[p][q]) continue;
        
        // Используем idx_next и idx_prev для правильной навигации!
        B3 pv = data[idx_next[p]];
        B3 pw = data[idx_prev[p]];
        B3 qv = data[idx_next[q]];
        B3 qw = data[idx_prev[q]];
        
        // Mod3 flip operations
        B3 pv_new = pv - qv;  // Mod3 subtraction
        B3 qw_new = pw + qw;  // Mod3 addition
        
        // Обновляем правильные позиции
        int pv_idx = idx_next[p];
        int qw_idx = idx_prev[q];
        
        // Получаем номера термов для set функции
        int term_pv = pv_idx / 3;
        int comp_pv = pv_idx % 3;
        int term_qw = qw_idx / 3;
        int comp_qw = qw_idx % 3;
        
        // Update using set function
        set(term_pv, comp_pv, pv_new);
        set(term_qw, comp_qw, qw_new);
        
        return true;
    }
}