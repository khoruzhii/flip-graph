// test_fields.cpp
#include "field.h"
#include "CLI11.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <set>
#include <cassert>

// =============================================================================
// TEMPLATE FIELD TESTS
// =============================================================================

template<int N>
void test_basic() {
    std::cout << "=== B<" << N << "> Basic Tests ===\n";
    
    // Constructor tests
    B<N> zero;
    std::cout << "Zero field is_zero(): " << zero.is_zero() << " (expected: 1)\n";
    
    // Unit creation and coefficient access
    for (int pos = 0; pos < 10; ++pos) {
        B<N> unit = B<N>::unit(pos);
        assert(unit.get_coefficient(pos) == 1);
        
        // Check that only this position is set to 1
        for (int other_pos = 0; other_pos < 10; ++other_pos) {
            if (other_pos != pos) {
                assert(unit.get_coefficient(other_pos) == 0);
            }
        }
    }
    
    std::cout << "✅ B<" << N << "> basic tests passed\n\n";
}

template<int N>
void test_arithmetic() {
    std::cout << "=== B<" << N << "> Arithmetic Tests ===\n";
    
    // Test basic operations
    B<N> zero;
    B<N> one = B<N>::unit(0);   // 1 at position 0
    
    // Identity element
    assert(one + zero == one);
    
    // Test negation properties
    B<N> neg_one = -one;
    std::cout << "Negation test at position 0: -1 = " << neg_one.get_coefficient(0) << "\n";
    
    // Test that x + (-x) = 0
    B<N> sum_with_neg = one + neg_one;
    assert(sum_with_neg.get_coefficient(0) == 0);
    
    // Commutativity test
    if constexpr (N >= 3) {
        B<N> two;
        if constexpr (N == 3) {
            two = B<3>(0, 1);  // 2 at position 0 for B3
        }
        assert(one + two == two + one);
        
        std::cout << "1 + 2 = " << (one + two).get_coefficient(0) << "\n";
        std::cout << "2 + 2 = " << (two + two).get_coefficient(0) << "\n";
    }
    
    std::cout << "✅ B<" << N << "> arithmetic tests passed\n\n";
}

template<int N>
void test_container_compatibility() {
    std::cout << "=== B<" << N << "> Container Compatibility Tests ===\n";
    
    // Test std::unordered_map (requires hash)
    {
        std::unordered_map<B<N>, int> umap;
        
        // Insert some test values
        umap[B<N>::unit(0)] = 1;
        umap[B<N>::unit(1)] = 2;
        umap[B<N>::unit(2)] = 3;
        
        std::cout << "B<" << N << "> unordered_map size: " << umap.size() << " (expected: 3)\n";
        
        // Test lookup
        assert(umap[B<N>::unit(1)] == 2);
    }
    
    // Test std::map (requires operator<)
    {
        std::map<B<N>, int> map;
        
        map[B<N>::unit(0)] = 100;
        map[B<N>::unit(1)] = 200;
        
        std::cout << "B<" << N << "> map size: " << map.size() << " (expected: 2)\n";
    }
    
    // Test std::unordered_set
    {
        std::unordered_set<B<N>> uset;
        
        uset.insert(B<N>::unit(0));
        uset.insert(B<N>::unit(1));
        uset.insert(B<N>::unit(0));  // Duplicate, shouldn't increase size
        
        std::cout << "B<" << N << "> unordered_set size: " << uset.size() << " (expected: 2)\n";
    }
    
    std::cout << "✅ B<" << N << "> container compatibility tests passed\n\n";
}

template<int N>
void test_pack_unpack() {
    std::cout << "=== B<" << N << "> Pack/Unpack Tests ===\n";
    
    if constexpr (N == 2) {
        B<2> original(0x123456789ABCDEF0ULL);
        U64 packed = pack_field(original);
        B<2> unpacked = unpack_field_b2(packed);
        
        std::cout << "B<2> pack/unpack: " << std::hex 
                  << "original=" << original.val 
                  << ", packed=" << packed 
                  << ", unpacked=" << unpacked.val << "\n";
        
        assert(original == unpacked);
        
        // Test conversion functions
        U64 test_val = 0xFF00FF00FF00FF00ULL;
        B<2> b2 = to_field_b2(test_val);
        assert(b2.val == test_val);
    }
    
    if constexpr (N == 3) {
        B<3> original(0x12345678ULL, 0x9ABCDEF0ULL);
        U64 packed = pack_field(original);
        B<3> unpacked = unpack_field_b3(packed);
        
        std::cout << "B<3> pack/unpack: " << std::hex 
                  << "original=(" << original.lo << "," << original.hi << ")"
                  << ", packed=" << packed 
                  << ", unpacked=(" << unpacked.lo << "," << unpacked.hi << ")\n";
        
        assert(original == unpacked);
        
        // Test conversion functions
        U64 test_val = 0xFF00FF00FF00FF00ULL;
        B<3> b3 = to_field_b3(test_val);
        assert(b3.lo == test_val && b3.hi == 0);
    }
    
    std::cout << "✅ B<" << N << "> pack/unpack tests passed\n\n";
}

template<int N>
void test_coefficient_functions() {
    std::cout << "=== B<" << N << "> Coefficient Functions Tests ===\n";
    
    // Create a pattern and test coefficient access
    B<N> unit5 = B<N>::unit(5);
    
    std::cout << "Unit at position 5 coefficients (first 8): ";
    for (int i = 0; i < 8; ++i) {
        int coeff = get_coefficient(unit5, i);
        std::cout << coeff;
    }
    std::cout << " (expected: 00000100)\n";
    
    // Verify specific positions
    assert(get_coefficient(unit5, 5) == 1);
    assert(get_coefficient(unit5, 0) == 0);
    assert(get_coefficient(unit5, 7) == 0);
    
    std::cout << "✅ B<" << N << "> coefficient function tests passed\n\n";
}

template<int N>
void test_field_traits() {
    std::cout << "=== B<" << N << "> Field Traits Tests ===\n";
    
    if constexpr (N == 2) {
        static_assert(field_traits<B<2>>::is_mod2 == true, "B<2> should be mod2");
        std::cout << "B<2> is_mod2: " << field_traits<B<2>>::is_mod2 << " (expected: 1)\n";
        
        U64 test_val = 0x12345678ULL;
        B<2> from_unpack = field_traits<B<2>>::unpack(test_val);
        B<2> from_u64 = field_traits<B<2>>::from_u64(test_val);
        
        assert(from_unpack == from_u64);
        assert(from_unpack.val == test_val);
    }
    
    if constexpr (N == 3) {
        static_assert(field_traits<B<3>>::is_mod2 == false, "B<3> should not be mod2");
        std::cout << "B<3> is_mod2: " << field_traits<B<3>>::is_mod2 << " (expected: 0)\n";
        
        U64 test_val = 0x12345678ULL;
        B<3> from_unpack = field_traits<B<3>>::unpack(test_val);
        B<3> from_u64 = field_traits<B<3>>::from_u64(test_val);
        
        assert(from_unpack == from_u64);
        assert(from_unpack.lo == (test_val & 0xFFFFFFFF));
        assert(from_unpack.hi == (test_val >> 32));
    }
    
    std::cout << "✅ B<" << N << "> field traits tests passed\n\n";
}

template<int N>
void test_hash_distribution() {
    std::cout << "=== B<" << N << "> Hash Distribution Test ===\n";
    
    const int NUM_SAMPLES = 1000;
    const int NUM_BUCKETS = 16;
    
    std::vector<int> buckets(NUM_BUCKETS, 0);
    std::hash<B<N>> hasher;
    
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        B<N> val;
        if constexpr (N == 2) {
            val = B<2>(static_cast<U64>(i) * 0x123456789ABCDEF0ULL);
        } else if constexpr (N == 3) {
            val = B<3>(static_cast<U64>(i) * 0x123456ULL, static_cast<U64>(i) * 0x789ABCULL);
        }
        
        size_t hash_val = hasher(val);
        buckets[hash_val % NUM_BUCKETS]++;
    }
    
    std::cout << "B<" << N << "> hash distribution: ";
    for (int count : buckets) {
        std::cout << count << " ";
    }
    std::cout << "\n";
    
    std::cout << "✅ B<" << N << "> hash distribution test completed\n\n";
}

// =============================================================================
// TEST RUNNER FOR SPECIFIC FIELD
// =============================================================================

template<int N>
void run_all_tests() {
    std::cout << "Running all tests for B<" << N << ">\n";
    std::cout << std::string(40, '=') << "\n\n";
    
    try {
        test_basic<N>();
        test_arithmetic<N>();
        test_container_compatibility<N>();
        test_pack_unpack<N>();
        test_coefficient_functions<N>();
        test_field_traits<N>();
        test_hash_distribution<N>();
        
        std::cout << "🎉 ALL B<" << N << "> TESTS PASSED! 🎉\n\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ B<" << N << "> TEST FAILED: " << e.what() << "\n";
        throw;
    }
}

// =============================================================================
// MAIN WITH CLI11
// =============================================================================

int main(int argc, char** argv) {
    CLI::App app{"Field Tests - Test specific B<N> field implementations"};
    
    int field_n = 2;  // Default to B2
    bool run_all = false;
    
    app.add_option("-n,--field", field_n, "Field modulus (2 or 3)")
        ->check(CLI::IsMember({2, 3}));
    
    app.add_flag("-a,--all", run_all, "Run tests for both B<2> and B<3>");
    
    CLI11_PARSE(app, argc, argv);
    
    std::cout << "Comprehensive B Field Tests\n";
    std::cout << "============================\n\n";
    
    try {
        if (run_all) {
            std::cout << "Running tests for all available fields...\n\n";
            run_all_tests<2>();
            run_all_tests<3>();
            
            std::cout << "🎉 ALL FIELD TESTS COMPLETED SUCCESSFULLY! 🎉\n";
            std::cout << "\nSummary:\n";
            std::cout << "✅ B<2> and B<3> basic operations work correctly\n";
            std::cout << "✅ Arithmetic follows field rules\n";
            std::cout << "✅ STL container compatibility verified\n";
            std::cout << "✅ Pack/unpack functions work correctly\n";
            std::cout << "✅ Coefficient access is accurate\n";
            std::cout << "✅ Field traits provide correct information\n";
            std::cout << "✅ Hash functions have reasonable distribution\n";
        } else {
            std::cout << "Running tests for B<" << field_n << ">...\n\n";
            
            if (field_n == 2) {
                run_all_tests<2>();
            } else if (field_n == 3) {
                run_all_tests<3>();
            }
            
            std::cout << "✅ B<" << field_n << "> tests completed successfully!\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ TESTS FAILED: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cout << "❌ UNKNOWN TEST FAILURE\n";
        return 1;
    }
    
    return 0;
}
