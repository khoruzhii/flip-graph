// scheme.h
#pragma once

#include <vector>
#include <random>
#include <cstdint>

#include "hash_dict.h"

using U64 = std::uint64_t;

class Scheme {
public:
    Scheme(const std::vector<U64>& initial, int sym, uint32_t seed);

    bool flip();
    bool plus();

    const std::vector<U64>& get_data() const { return data; }
    int  get_orank() const { return rank / sym; }
    int  get_rank()  const { return rank; }

private:
    void add(int r, U64 v); // add row index r to the block of value v
    void del(int r, U64 v); // remove row index r from the block of value v

    bool flip3();
    bool flip6();
    bool plus3();
    bool plus6();

private:
    int sym{3};
    int n{0};
    int rank{0};
    std::mt19937 rng{123456u};

    std::vector<U64> data;     // u,v,w repeating per triple
    std::vector<int> idx_next; // v index per row
    std::vector<int> idx_prev; // w index per row

    HashDict unique;
    HashDict flippable_idx;
    std::vector<U64> flippable;

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
