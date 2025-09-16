// field.h
#pragma once

#include <iostream>
#include <cstdint>

using U32 = std::uint32_t;
using U64 = std::uint64_t;

// Primary template declaration
template<int N>
struct B;

// Specialization for N = 2
template<>
struct B<2> {
    U64 val;
    
    B() : val(0) {}
    explicit B(U64 v) : val(v) {}
    
    bool is_zero() const { return val == 0; }
    bool operator==(const B& o) const { return val == o.val; }
    bool operator!=(const B& o) const { return val != o.val; }

    static B unit(int position) {
        return B(1ULL << position);
    }

    int get_coefficient(int position) const {
        if (position < 0 || position >= 64) return 0;
        return (val >> position) & 1;
    }

    bool operator<(const B& o) const {
        return val < o.val;
    }
};

inline B<2> operator+(const B<2>& a, const B<2>& b) {
    return B<2>(a.val ^ b.val);
}
inline B<2> operator-(const B<2>& x) { return x; }
inline B<2> operator-(const B<2>& a, const B<2>& b) {
    return B<2>(a.val ^ b.val);
}

inline U64 pack_field(const B<2>& b) { return b.val; }
inline B<2> unpack_field_b2(U64 packed) { return B<2>(packed); }
inline B<2> to_field_b2(U64 v) { return B<2>(v); }

// Specialization for N = 3
template<>
struct B<3> {
    U64 lo, hi;
    B() : lo(0), hi(0) {}
    B(U64 l, U64 h) : lo(l), hi(h) {}
    
    bool is_zero() const { return lo == 0 && hi == 0; }
    bool operator==(const B& o) const { return lo == o.lo && hi == o.hi; }
    bool operator!=(const B& o) const { return !(*this == o); }

    static B unit(int position) {
        return B(1ULL << position, 0);
    }

    int get_coefficient(int position) const {
        if (position < 0 || position >= 64) return 0;
        int lo_bit = (lo >> position) & 1;
        int hi_bit = (hi >> position) & 1;
        return lo_bit + 2 * hi_bit;
    }

    bool operator<(const B& o) const {
        if (lo != o.lo) return lo < o.lo;
        return hi < o.hi;
    }
};

inline B<3> operator+(const B<3>& a, const B<3>& b) {
    const U64 u = ((a.lo | b.lo) & (a.hi | b.hi));
    return B<3>{
        (a.lo ^ b.lo) ^ (a.hi & b.hi) ^ u,
        (a.hi ^ b.hi) ^ (a.lo & b.lo) ^ u
    };
}
inline B<3> operator-(const B<3>& x) { return B<3>{x.hi, x.lo}; }
inline B<3> operator-(const B<3>& a, const B<3>& b) { return a + (-b); }

inline U64 pack_field(const B<3>& b) {
    return (b.hi << 32) | (b.lo & 0xFFFFFFFF);
}
inline B<3> unpack_field_b3(U64 packed) {
    return B<3>{packed & 0xFFFFFFFF, packed >> 32};
}
inline B<3> to_field_b3(U64 v) { return B<3>{v, 0}; }

// Type traits
template<typename Field>
struct field_traits;

template<>
struct field_traits<B<2>> {
    static constexpr bool is_mod2 = true;
    static B<2> unpack(U64 v) { return unpack_field_b2(v); }
    static B<2> from_u64(U64 v) { return to_field_b2(v); }
};

template<>
struct field_traits<B<3>> {
    static constexpr bool is_mod2 = false;
    static B<3> unpack(U64 v) { return unpack_field_b3(v); }
    static B<3> from_u64(U64 v) { return to_field_b3(v); }
};

// Hash specializations
namespace std {
    template<>
    struct hash<B<2>> {
        size_t operator()(const B<2>& b) const noexcept {
            return hash<U64>{}(b.val);
        }
    };
    template<>
    struct hash<B<3>> {
        size_t operator()(const B<3>& b) const noexcept {
            size_t h1 = hash<U64>{}(b.lo);
            size_t h2 = hash<U64>{}(b.hi);
            return h1 ^ (h2 << 1);
        }
    };
}

// Global coefficient access
inline int get_coefficient(const B<2>& f, int idx) {
    return f.get_coefficient(idx);
}
inline int get_coefficient(const B<3>& f, int idx) {
    return f.get_coefficient(idx);
}
