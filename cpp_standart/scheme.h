// scheme.h
#pragma once

#include <vector>
#include <random>
#include <cstdint>
#include <unordered_map>

using U64 = std::uint64_t;

class Scheme {
public:
    Scheme(const std::vector<U64>& initial, uint32_t seed, int n);

    bool flip();
    bool plus();
    bool reduction();

    bool flipu() { return flip_private(0); }
    bool flipv() { return flip_private(1); }
    bool flipw() { return flip_private(2); }

    bool reductionuv() { return reduction_private(0); }
    bool reductionvw() { return reduction_private(1); }
    bool reductionwu() { return reduction_private(2); }

    const std::vector<U64>& get_flip(int type) const { return flippable[type]; }
    const std::unordered_map<U64, std::vector<int> >& get_map(int type) const { return pos[type]; }
    const std::vector<U64>& get_data() const { return data; }


private:
    bool flip_private(int type);
    bool reduction_private(int type);

    void delete_element(int id);
    void add_element(int id, U64 x);
    void remove_tensor(int i);
    void value_assign(int id, U64 x);

private:
    int n = 0;
    int cap;
    int rank = 0;
    std::mt19937 rng{123456u};

    int get_rank()  const { return rank; }
    int get_n() const { return n; }

    /*
    0 -> u
    1 -> v
    2 -> w
    */

    std::vector<U64> data;     // u,v,w repeating per triple

    std::array<std::unordered_map<U64, std::vector<int> >, 3> pos;
    std::array<std::vector<U64>, 3> flippable;

    std::vector<int> idx_next;
    std::vector<int> idx_prev;
};