// hash_dict.h
#pragma once

#include <vector>
#include <cstdint>

using U64 = std::uint64_t;

/*
 * HashDict: flat-bucket dictionary for the scheme.
 *  - 65,536 buckets
 *  - bucket stride = 16 contiguous slots
 *  - count[base] holds number of items in the bucket
 *  - key/value store items in [base, base + count-1]
 */
class HashDict {
public:
    unsigned int lasthash{0};

    HashDict();

    int size() const;
    unsigned int hash(U64 k) const;

    int  contains(U64 k);
    void add(U64 k, int v);
    void addx(U64 k, int v);
    void remove(U64 k);
    void removex(U64 k);
    void replace(U64 k, int v);
    void replacex(U64 k, int v);
    int  getvalue(U64 k);
    int  getvaluex(U64 k) const;

private:
    static constexpr int BucketCount = 65536;
    static constexpr int BucketStride = 16;
    static constexpr int BucketStrideShift = 4;
    static constexpr int KSize = BucketCount * BucketStride;

    std::vector<int>  count;
    std::vector<U64>  key;
    std::vector<int>  value;
};
