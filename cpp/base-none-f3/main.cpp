#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <random>
#include "scheme.h"

using U32 = std::uint32_t;
using U64 = std::uint64_t;

// Generate trivial decomposition for n×n matrix multiplication
// Returns binary data (0s and 1s) which will be converted to mod 3
std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    data.reserve(n * n * n * 3);
    
    // For each (i,j,k) triple, add rank-one tensor:
    // u[i,j] ⊗ v[j,k] ⊗ w[k,i]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                U64 u = 1ULL << (i * n + j); // u[i,j]
                U64 v = 1ULL << (j * n + k); // v[j,k]
                U64 w = 1ULL << (k * n + i); // w[k,i]
                
                data.push_back(u);
                data.push_back(v);
                data.push_back(w);
            }
        }
    }
    
    return data;
}

// Extract bit value from B3 (returns 0, 1, or 2)
inline int get_trit(const B3& b, int idx) {
    int lo_bit = (b.lo >> idx) & 1;
    int hi_bit = (b.hi >> idx) & 1;
    return lo_bit + 2 * hi_bit;  // 00->0, 01->1, 10->2, 11->invalid
}

// Check if the scheme correctly implements matrix multiplication in mod 3
bool verify_scheme_mod3(const std::vector<B3>& data, int n) {
    // Number of rank-one tensors
    int r = data.size() / 3;
    
    // Check each Brent equation in mod 3:    
    for (int i1 = 0; i1 < n; ++i1) {
        for (int i2 = 0; i2 < n; ++i2) {
            for (int j1 = 0; j1 < n; ++j1) {
                for (int j2 = 0; j2 < n; ++j2) {
                    for (int k1 = 0; k1 < n; ++k1) {
                        for (int k2 = 0; k2 < n; ++k2) {
                            // Compute sum over all rank-one tensors (mod 3)
                            int sum = 0;
                            
                            for (int l = 0; l < r; ++l) {
                                const B3& u = data[3 * l];
                                const B3& v = data[3 * l + 1]; 
                                const B3& w = data[3 * l + 2];
                                
                                // Skip if term is zero
                                if (u.is_zero()) continue;
                                
                                int a = get_trit(u, i1 * n + i2);
                                int b = get_trit(v, j1 * n + j2);
                                int c = get_trit(w, k1 * n + k2);
                                
                                // Add product mod 3
                                sum = (sum + a * b * c) % 3;
                            }
                            
                            int expected = (i2 == j1 && j2 == k1 && k2 == i1) ? 1 : 0;
                            if (sum != expected) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return true;
}

// Get rank from B3 data
int get_rank_b3(const std::vector<B3>& data) {
    int n_terms = data.size() / 3;
    int rank = 0;
    for (int i = 0; i < n_terms; ++i) {
        if (!data[i*3].is_zero())
            rank++;
    }
    return rank;
}

int main() {
    int n = 3got;
    int N = 1e7;
    auto data = generate_trivial_decomposition(n);
    
    for (int j = 0; j < 10; ++j) {

    U32 seed = std::random_device{}();
    Scheme scheme(data, seed);
    
    // std::cout << "Initial rank = " << scheme.get_rank() 
    //           << ", correct = " << verify_scheme_mod3(scheme.get_data_b3(), n)
    //           << "\n";
    
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    
    for (int i = 0; i < N; ++i) {
        if (!scheme.flip()) {
            std::cout << "No flips at step " << i << "\n";
            break;
        }
    }
    
    auto t1 = clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    std::cout << "Final rank = " << scheme.get_rank() 
              << ", seed = " << seed 
              << ", correct = " << verify_scheme_mod3(scheme.get_data_b3(), n);
    
    if (time > 0) {
        std::cout << ", speed = " << 1e3 / time * N / 1e6 << " M/s";
    }
    std::cout << "\n";

    }
    
    std::cout << "Finished." << "\n";
    return 0;
}