// hash_dict.cpp
#include "hash_dict.h"

HashDict::HashDict() {
    count.assign(KSize, 0);
    key.assign(KSize, 0);
    value.assign(KSize, 0);
    // Ensure bucket counters at bases are zeroed.
    for (int i = 0; i < BucketCount; ++i) {
        count[i << BucketStrideShift] = 0;
    }
}

int HashDict::size() const {
    int total = 0;
    for (int i = 0; i < BucketCount; ++i) {
        total += count[i << BucketStrideShift];
    }
    return total;
}

unsigned int HashDict::hash(U64 k) const {
    return static_cast<unsigned int>((k % 65213ULL) << BucketStrideShift);
}

int HashDict::contains(U64 k) {
    lasthash = hash(k);
    int c = count[lasthash];
    if (c == 0) return 0;
    if (c == 1) return key[lasthash] == k ? 1 : 0;
    for (int i = c - 1; i >= 0; --i) {
        if (key[lasthash + i] == k) return 1;
    }
    return 0;
}

void HashDict::add(U64 k, int v) {
    lasthash = hash(k);
    addx(k, v);
}

void HashDict::addx(U64 k, int v) {
    int base = static_cast<int>(lasthash);
    int idx = base + count[base];
    key[idx] = k;
    value[idx] = v;
    count[base]++;
}

void HashDict::remove(U64 k) {
    lasthash = hash(k);
    removex(k);
}

void HashDict::removex(U64 k) {
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

void HashDict::replace(U64 k, int v) {
    lasthash = hash(k);
    replacex(k, v);
}

void HashDict::replacex(U64 k, int v) {
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

int HashDict::getvalue(U64 k) {
    lasthash = hash(k);
    return getvaluex(k);
}

int HashDict::getvaluex(U64 k) const {
    int base = static_cast<int>(lasthash);
    int c = count[base];
    if (c == 1) return value[base];
    int i = base + c - 1;
    while (key[i] != k) --i;
    return value[i];
}
