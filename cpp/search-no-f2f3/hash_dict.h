// hash_dict.h
#pragma once

#include <vector>
#include <cstdint>
#include <cassert>

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
    mutable unsigned int lasthash{0};

    HashDict() {
        count.assign(KSize, 0);
        key.assign(KSize, 0);
        value.assign(KSize, 0);
        // Ensure bucket counters at bases are zeroed.
        for (int i = 0; i < BucketCount; ++i) {
            count[i << BucketStrideShift] = 0;
        }
    }

    int size() const {
        int total = 0;
        for (int i = 0; i < BucketCount; ++i) {
            total += count[i << BucketStrideShift];
        }
        return total;
    }

    inline unsigned int hash(U64 k) const {
        return static_cast<unsigned int>((k % 65213ULL) << BucketStrideShift);
    }

    int contains(U64 k) {
        lasthash = hash(k);
        int c = count[lasthash];
        if (c == 0) return 0;
        if (c == 1) return key[lasthash] == k ? 1 : 0;
        for (int i = c - 1; i >= 0; --i) {
            if (key[lasthash + i] == k) return 1;
        }
        return 0;
    }

    void add(U64 k, int v) {
        lasthash = hash(k);
        addx(k, v);
    }

    void addx(U64 k, int v) {
        int base = static_cast<int>(lasthash);
        assert(count[base] < BucketStride && "Bucket overflow");
        int idx = base + count[base];
        key[idx] = k;
        value[idx] = v;
        count[base]++;
    }

    void remove(U64 k) {
        lasthash = hash(k);
        removex(k);
    }

    void removex(U64 k) {
        int base = static_cast<int>(lasthash);
        int c = count[base];
        if (c == 1) {
            count[base] = 0;
            return;
        }
        int i = base + c - 1;
        U64 x = key[i];
        int v = value[i];
        while (x != k) {
            --i;
            U64 y = x;
            x = key[i];
            key[i] = y;
            int w = v;
            v = value[i];
            value[i] = w;
        }
        count[base]--;
    }

    void replace(U64 k, int v) {
        lasthash = hash(k);
        replacex(k, v);
    }

    void replacex(U64 k, int v) {
        int base = static_cast<int>(lasthash);
        int c = count[base];
        if (c == 1) {
            value[base] = v;
            return;
        }
        int i = base + c - 1;
        while (key[i] != k) --i;
        value[i] = v;
    }

    int getvalue(U64 k) {
        lasthash = hash(k);
        return getvaluex(k);
    }

    int getvaluex(U64 k) const {
        int base = static_cast<int>(lasthash);
        int c = count[base];
        if (c == 1) return value[base];
        int i = base + c - 1;
        while (key[i] != k) --i;
        return value[i];
    }

private:
    static constexpr int BucketCount = 65536;
    static constexpr int BucketStride = 16;
    static constexpr int BucketStrideShift = 4;
    static constexpr int KSize = BucketCount * BucketStride;

    std::vector<int>  count;
    std::vector<U64>  key;
    std::vector<int>  value;
};