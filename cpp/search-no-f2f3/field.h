// field.h
#pragma once

#include <cstdint>

using U32 = std::uint32_t;
using U64 = std::uint64_t;

// ----------------------------------------------------------------------

// Mod 2 field (binary)
struct B2 {
    U64 val;
    
    B2() : val(0) {}
    explicit B2(U64 v) : val(v) {}
    
    bool is_zero() const { return val == 0; }
    
    bool operator==(const B2& o) const { return val == o.val; }
    bool operator!=(const B2& o) const { return val != o.val; }
};

// Mod 2 operations
inline B2 operator+(const B2& a, const B2& b) { return B2(a.val ^ b.val); }
inline B2 operator-(const B2& x) { return x; }  // negation is identity in mod 2
inline B2 operator-(const B2& a, const B2& b) { return B2(a.val ^ b.val); }

// Pack/unpack for B2 (trivial)
inline U64 pack_field(const B2& b) { return b.val; }
inline B2 unpack_field_b2(U64 packed) { return B2(packed); }

// ----------------------------------------------------------------------

// Mod 3 field (ternary) - two-bit packed: 0=00, 1=01, 2=10
struct B3 {
    U64 lo;
    U64 hi;
    
    B3() : lo(0), hi(0) {}
    B3(U64 l, U64 h) : lo(l), hi(h) {}
    
    bool is_zero() const { return lo == 0 && hi == 0; }
    
    bool operator==(const B3& o) const { return lo == o.lo && hi == o.hi; }
    bool operator!=(const B3& o) const { return !(*this == o); }
};

// Mod 3 addition
inline B3 operator+(const B3& a, const B3& b) {
    const U64 u = ((a.lo | b.lo) & (a.hi | b.hi));
    return B3{
        (a.lo ^ b.lo) ^ (a.hi & b.hi) ^ u,
        (a.hi ^ b.hi) ^ (a.lo & b.lo) ^ u
    };
}

// Mod 3 negation
inline B3 operator-(const B3& x) { return B3{x.hi, x.lo}; }

// Mod 3 subtraction
inline B3 operator-(const B3& a, const B3& b) { return a + (-b); }

// Pack B3 into U64 (half bits for lo, half for hi)
// inline U64 pack_field(const B3& b) {
//     return (b.hi << 32) | (b.lo & 0xFFFFFFFF);
// }
// inline U64 pack_field(const B3& b) {
//     const U64 kMul = 0x9ddfea08eb382d69ULL;
//     U64 a1 = (b.hi ^ (b.lo * kMul)) * kMul;
//     a1 ^= (a1 >> 47);
//     U64 a2 = (b.lo ^ a1) * kMul;
//     a2 ^= (a2 >> 47);
//     a2 *= kMul;
//     return a2;
// }
inline U64 pack_field(const B3& b) {
    // Assumes __int128 is available. Uses wyhash-style MUM mixer + strong avalanche.
    constexpr U64 k0 = 0xa0761d6478bd642full;
    constexpr U64 k1 = 0xe7037ed1a0b428dbull;
    constexpr U64 k2 = 0x8ebc6af09c88c6e3ull;
    constexpr U64 k3 = 0x589965cc75374cc3ull;

    auto mum = [](U64 a, U64 b) -> U64 {
        __uint128_t r = ( (__uint128_t)a * (__uint128_t)b );
        return (U64)r ^ (U64)(r >> 64); // xor-fold high and low halves
    };

    // Collapse two 64-bit lanes with independent secrets
    U64 h = mum(b.hi ^ k0, b.lo ^ k1);
    // One more non-linear round to break remaining structure
    h = mum(h ^ k2, k3);

    // Final avalanche to spread any remaining linearity
    h ^= h >> 29;
    h *= 0x94d049bb133111ebull; // odd constant from splitmix64
    h ^= h >> 31;

    return h;
}


// Unpack U64 to B3
inline B3 unpack_field_b3(U64 packed) {
    return B3{packed & 0xFFFFFFFF, packed >> 32};
}

// Helper to convert U64 to field (assuming binary input)
inline B2 to_field_b2(U64 v) { return B2(v); }
inline B3 to_field_b3(U64 v) { return B3{v, 0}; }  // binary to ternary

// Type traits for field operations
template<typename Field>
struct field_traits;

template<>
struct field_traits<B2> {
    static constexpr bool is_mod2 = true;
    // static B2 unpack(U64 v) { return unpack_field_b2(v); }
    static B2 from_u64(U64 v) { return to_field_b2(v); }
};

template<>
struct field_traits<B3> {
    static constexpr bool is_mod2 = false;
    // static B3 unpack(U64 v) { return unpack_field_b3(v); }
    static B3 from_u64(U64 v) { return to_field_b3(v); }
};


// ----------------------------------------------------------------------

// Mod 4 ring (quaternary) - two-bit packed: 0=00, 1=01, 2=10, 3=11
struct B4 {
    U64 lo;
    U64 hi;
    
    B4() : lo(0), hi(0) {}
    B4(U64 l, U64 h) : lo(l), hi(h) {}
    
    bool is_zero() const { return lo == 0 && hi == 0; }
    
    bool operator==(const B4& o) const { return lo == o.lo && hi == o.hi; }
    bool operator!=(const B4& o) const { return !(*this == o); }
};

// Mod 4 addition
inline B4 operator+(const B4& a, const B4& b) {
    U64 s0 = a.lo ^ b.lo;                    // LSB sum
    U64 s1 = a.hi ^ b.hi ^ (a.lo & b.lo);    // MSB with carry
    return B4{s0, s1};
}

// Mod 4 negation: 0->0, 1->3, 2->2, 3->1
inline B4 operator-(const B4& x) {
    return B4{x.lo, x.hi ^ x.lo};
}

// Mod 4 subtraction
inline B4 operator-(const B4& a, const B4& b) {
    return a + (-b);
}

// Pack B4 into U64 (half bits for lo, half for hi)
inline U64 pack_field(const B4& b) {
    return (b.hi << 32) | (b.lo & 0xFFFFFFFF);
}

// Unpack U64 to B4
inline B4 unpack_field_b4(U64 packed) {
    return B4{packed & 0xFFFFFFFF, packed >> 32};
}

// Helper to convert U64 to field (assuming binary input)
inline B4 to_field_b4(U64 v) {
    return B4{v, 0};  // binary to quaternary
}

// Type traits specialization for B4
template<>
struct field_traits<B4> {
    static constexpr bool is_mod2 = false;
    static B4 unpack(U64 v) { return unpack_field_b4(v); }
    static B4 from_u64(U64 v) { return to_field_b4(v); }
};

// ----------------------------------------------------------------------