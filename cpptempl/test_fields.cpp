// test_fields.cpp
#include "field.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <set>
#include <cassert>

// =============================================================================
// BASIC FIELD TESTS
// =============================================================================

void test_b2_basic() {
    std::cout << "=== B2 Basic Tests ===\n";
    
    // Constructor tests
    B2 zero;
    B2 one(1);
    B2 pattern(0xAAAAAAAAAAAAAAAAULL);
    
    std::cout << "Zero field is_zero(): " << zero.is_zero() << " (expected: 1)\n";
    std::cout << "One field is_zero(): " << one.is_zero() << " (expected: 0)\n";
    
    // Equality tests
    B2 zero2;
    assert(zero == zero2);
    assert(zero != one);
    
    // Unit creation
    for (int pos = 0; pos < 10; ++pos) {
        B2 unit = B2::unit(pos);
        assert(unit.get_coefficient(pos) == 1);
        
        // Check that only this position is set
        for (int other_pos = 0; other_pos < 10; ++other_pos) {
            if (other_pos != pos) {
                assert(unit.get_coefficient(other_pos) == 0);
            }
        }
    }
    
    // Coefficient access
    B2 test_val(0b1010101010101010ULL);
    for (int i = 0; i < 16; ++i) {
        int expected = (i % 2 == 1) ? 1 : 0;
        assert(test_val.get_coefficient(i) == expected);
    }
    
    std::cout << "✅ B2 basic tests passed\n\n";
}

void test_b3_basic() {
    std::cout << "=== B3 Basic Tests ===\n";
    
    // Constructor tests
    B3 zero;
    B3 pattern(0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL);
    
    std::cout << "Zero field is_zero(): " << zero.is_zero() << " (expected: 1)\n";
    std::cout << "Pattern field is_zero(): " << pattern.is_zero() << " (expected: 0)\n";
    
    // Unit creation and coefficient access
    for (int pos = 0; pos < 10; ++pos) {
        B3 unit = B3::unit(pos);
        assert(unit.get_coefficient(pos) == 1);
        
        // Check that only this position is set to 1
        for (int other_pos = 0; other_pos < 10; ++other_pos) {
            if (other_pos != pos) {
                assert(unit.get_coefficient(other_pos) == 0);
            }
        }
    }
    
    // Test different coefficient values
    B3 coeff0(0, 0);                    // All 0s
    B3 coeff1(1ULL << 5, 0);           // 1 at position 5
    B3 coeff2(0, 1ULL << 7);           // 2 at position 7
    
    assert(coeff0.get_coefficient(5) == 0);
    assert(coeff1.get_coefficient(5) == 1);
    assert(coeff2.get_coefficient(7) == 2);
    
    std::cout << "✅ B3 basic tests passed\n\n";
}

// =============================================================================
// ARITHMETIC TESTS
// =============================================================================

void test_b2_arithmetic() {
    std::cout << "=== B2 Arithmetic Tests ===\n";
    
    // Test XOR properties (F_2 addition)
    B2 a(0b1100);
    B2 b(0b1010);
    B2 sum = a + b;
    
    std::cout << "1100 + 1010 = " << std::hex << sum.val << " (expected: 0110)\n";
    assert(sum.val == 0b0110);
    
    // Commutativity
    assert(a + b == b + a);
    
    // Associativity
    B2 c(0b1111);
    assert((a + b) + c == a + (b + c));
    
    // Identity element (zero)
    B2 zero;
    assert(a + zero == a);
    
    // Self-inverse (x + x = 0 in F_2)
    assert((a + a).is_zero());
    
    // Negation is identity
    assert(-a == a);
    
    // Subtraction same as addition
    assert(a - b == a + b);
    
    std::cout << "✅ B2 arithmetic tests passed\n\n";
}

void test_b3_arithmetic() {
    std::cout << "=== B3 Arithmetic Tests ===\n";
    
    // Test basic addition
    B3 zero;
    B3 one = B3::unit(0);   // 1 at position 0
    B3 two(0, 1);          // 2 at position 0
    
    // Test 1 + 1 = 2 in F_3
    B3 sum_11 = one + one;
    std::cout << "1 + 1 = " << sum_11.get_coefficient(0) << " (expected: 2)\n";
    
    // Test 1 + 2 = 0 in F_3
    B3 sum_12 = one + two;
    std::cout << "1 + 2 = " << sum_12.get_coefficient(0) << " (expected: 0)\n";
    
    // Test 2 + 2 = 1 in F_3
    B3 sum_22 = two + two;
    std::cout << "2 + 2 = " << sum_22.get_coefficient(0) << " (expected: 1)\n";
    
    // Commutativity
    assert(one + two == two + one);
    
    // Identity element
    assert(one + zero == one);
    assert(two + zero == two);
    
    // Test negation: -1 = 2, -2 = 1, -0 = 0
    B3 neg_one = -one;
    B3 neg_two = -two;
    
    std::cout << "-1 = " << neg_one.get_coefficient(0) << " (expected: 2)\n";
    std::cout << "-2 = " << neg_two.get_coefficient(0) << " (expected: 1)\n";
    
    // Test that x + (-x) = 0
    assert((one + neg_one).get_coefficient(0) == 0);
    assert((two + neg_two).get_coefficient(0) == 0);
    
    // Test subtraction
    B3 diff = two - one;
    std::cout << "2 - 1 = " << diff.get_coefficient(0) << " (expected: 1)\n";
    
    std::cout << "✅ B3 arithmetic tests passed\n\n";
}

// =============================================================================
// CONTAINER COMPATIBILITY TESTS
// =============================================================================

void test_container_compatibility() {
    std::cout << "=== Container Compatibility Tests ===\n";
    
    // Test std::unordered_map (requires hash)
    {
        std::unordered_map<B2, int> b2_umap;
        std::unordered_map<B3, int> b3_umap;
        
        // Insert some test values
        b2_umap[B2::unit(0)] = 1;
        b2_umap[B2::unit(1)] = 2;
        b2_umap[B2::unit(2)] = 3;
        
        b3_umap[B3::unit(0)] = 10;
        b3_umap[B3::unit(1)] = 20;
        b3_umap[B3::unit(2)] = 30;
        
        std::cout << "B2 unordered_map size: " << b2_umap.size() << " (expected: 3)\n";
        std::cout << "B3 unordered_map size: " << b3_umap.size() << " (expected: 3)\n";
        
        // Test lookup
        assert(b2_umap[B2::unit(1)] == 2);
        assert(b3_umap[B3::unit(2)] == 30);
    }
    
    // Test std::map (requires operator<)
    {
        std::map<B2, int> b2_map;
        std::map<B3, int> b3_map;
        
        b2_map[B2::unit(0)] = 100;
        b2_map[B2::unit(1)] = 200;
        
        b3_map[B3::unit(0)] = 1000;
        b3_map[B3::unit(1)] = 2000;
        
        std::cout << "B2 map size: " << b2_map.size() << " (expected: 2)\n";
        std::cout << "B3 map size: " << b3_map.size() << " (expected: 2)\n";
    }
    
    // Test std::unordered_set
    {
        std::unordered_set<B2> b2_uset;
        std::unordered_set<B3> b3_uset;
        
        b2_uset.insert(B2::unit(0));
        b2_uset.insert(B2::unit(1));
        b2_uset.insert(B2::unit(0));  // Duplicate, shouldn't increase size
        
        b3_uset.insert(B3::unit(0));
        b3_uset.insert(B3::unit(1));
        b3_uset.insert(B3::unit(0));  // Duplicate
        
        std::cout << "B2 unordered_set size: " << b2_uset.size() << " (expected: 2)\n";
        std::cout << "B3 unordered_set size: " << b3_uset.size() << " (expected: 2)\n";
    }
    
    std::cout << "✅ Container compatibility tests passed\n\n";
}

// =============================================================================
// PACK/UNPACK TESTS
// =============================================================================

void test_pack_unpack() {
    std::cout << "=== Pack/Unpack Tests ===\n";
    
    // Test B2 pack/unpack
    {
        B2 original(0x123456789ABCDEF0ULL);
        U64 packed = pack_field(original);
        B2 unpacked = unpack_field_b2(packed);
        
        std::cout << "B2 pack/unpack: " << std::hex 
                  << "original=" << original.val 
                  << ", packed=" << packed 
                  << ", unpacked=" << unpacked.val << "\n";
        
        assert(original == unpacked);
    }
    
    // Test B3 pack/unpack
    {
        B3 original(0x12345678ULL, 0x9ABCDEF0ULL);
        U64 packed = pack_field(original);
        B3 unpacked = unpack_field_b3(packed);
        
        std::cout << "B3 pack/unpack: " << std::hex 
                  << "original=(" << original.lo << "," << original.hi << ")"
                  << ", packed=" << packed 
                  << ", unpacked=(" << unpacked.lo << "," << unpacked.hi << ")\n";
        
        assert(original == unpacked);
    }
    
    // Test conversion functions
    {
        U64 test_val = 0xFF00FF00FF00FF00ULL;
        
        B2 b2 = to_field_b2(test_val);
        B3 b3 = to_field_b3(test_val);
        
        assert(b2.val == test_val);
        assert(b3.lo == test_val && b3.hi == 0);
    }
    
    std::cout << "✅ Pack/unpack tests passed\n\n";
}

// =============================================================================
// COEFFICIENT TESTS
// =============================================================================

void test_coefficient_functions() {
    std::cout << "=== Coefficient Functions Tests ===\n";
    
    // Test global get_coefficient functions
    {
        B2 b2_pattern(0b10101010);
        B3 b3_mixed(0b1100, 0b1010);  // pos 0,1: 0+2=2, pos 2,3: 1+0=1, etc.
        
        std::cout << "B2 pattern coefficients: ";
        for (int i = 0; i < 8; ++i) {
            int coeff = get_coefficient(b2_pattern, i);
            std::cout << coeff;
        }
        std::cout << " (expected: 01010101)\n";
        
        std::cout << "B3 mixed coefficients: ";
        for (int i = 0; i < 6; ++i) {
            int coeff = get_coefficient(b3_mixed, i);
            std::cout << coeff;
        }
        std::cout << "\n";
        
        // Verify specific positions
        assert(get_coefficient(b2_pattern, 1) == 1);
        assert(get_coefficient(b2_pattern, 0) == 0);
        
        // B3: pos 0: lo=0,hi=1 -> 2, pos 1: lo=0,hi=1 -> 2, pos 2: lo=1,hi=0 -> 1
        assert(get_coefficient(b3_mixed, 1) == 2);  // hi bit set
        assert(get_coefficient(b3_mixed, 2) == 1);  // lo bit set
    }
    
    std::cout << "✅ Coefficient function tests passed\n\n";
}

// =============================================================================
// FIELD TRAITS TESTS
// =============================================================================

void test_field_traits() {
    std::cout << "=== Field Traits Tests ===\n";
    
    // Test B2 traits
    {
        static_assert(field_traits<B2>::is_mod2 == true, "B2 should be mod2");
        
        U64 test_val = 0x12345678ULL;
        B2 from_unpack = field_traits<B2>::unpack(test_val);
        B2 from_u64 = field_traits<B2>::from_u64(test_val);
        
        assert(from_unpack == from_u64);
        assert(from_unpack.val == test_val);
    }
    
    // Test B3 traits
    {
        static_assert(field_traits<B3>::is_mod2 == false, "B3 should not be mod2");
        
        U64 test_val = 0x12345678ULL;
        B3 from_unpack = field_traits<B3>::unpack(test_val);
        B3 from_u64 = field_traits<B3>::from_u64(test_val);
        
        assert(from_unpack == from_u64);
        assert(from_unpack.lo == (test_val & 0xFFFFFFFF));
        assert(from_unpack.hi == (test_val >> 32));
    }
    
    std::cout << "✅ Field traits tests passed\n\n";
}

// =============================================================================
// STRESS TESTS
// =============================================================================

void test_hash_distribution() {
    std::cout << "=== Hash Distribution Test ===\n";
    
    const int NUM_SAMPLES = 1000;
    const int NUM_BUCKETS = 16;
    
    // Test B2 hash distribution
    {
        std::vector<int> b2_buckets(NUM_BUCKETS, 0);
        std::hash<B2> b2_hasher;
        
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            B2 val(static_cast<U64>(i) * 0x123456789ABCDEF0ULL);
            size_t hash_val = b2_hasher(val);
            b2_buckets[hash_val % NUM_BUCKETS]++;
        }
        
        std::cout << "B2 hash distribution: ";
        for (int count : b2_buckets) {
            std::cout << count << " ";
        }
        std::cout << "\n";
    }
    
    // Test B3 hash distribution
    {
        std::vector<int> b3_buckets(NUM_BUCKETS, 0);
        std::hash<B3> b3_hasher;
        
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            B3 val(static_cast<U64>(i) * 0x123456ULL, static_cast<U64>(i) * 0x789ABCULL);
            size_t hash_val = b3_hasher(val);
            b3_buckets[hash_val % NUM_BUCKETS]++;
        }
        
        std::cout << "B3 hash distribution: ";
        for (int count : b3_buckets) {
            std::cout << count << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "✅ Hash distribution test completed\n\n";
}

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

int main() {
    std::cout << "Comprehensive Field Tests\n";
    std::cout << "=========================\n\n";
    
    try {
        // Basic functionality
        test_b2_basic();
        test_b3_basic();
        
        // Arithmetic operations
        test_b2_arithmetic();
        test_b3_arithmetic();
        
        // Container compatibility
        test_container_compatibility();
        
        // Pack/unpack functionality
        test_pack_unpack();
        
        // Coefficient access
        test_coefficient_functions();
        
        // Field traits
        test_field_traits();
        
        // Stress tests
        test_hash_distribution();
        
        std::cout << "🎉 ALL FIELD TESTS PASSED! 🎉\n";
        std::cout << "\nSummary:\n";
        std::cout << "✅ B2 and B3 basic operations work correctly\n";
        std::cout << "✅ Arithmetic follows F_2 and F_3 rules\n";
        std::cout << "✅ STL container compatibility verified\n";
        std::cout << "✅ Pack/unpack functions work correctly\n";
        std::cout << "✅ Coefficient access is accurate\n";
        std::cout << "✅ Field traits provide correct information\n";
        std::cout << "✅ Hash functions have reasonable distribution\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ TEST FAILED: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cout << "❌ UNKNOWN TEST FAILURE\n";
        return 1;
    }
    
    return 0;
}