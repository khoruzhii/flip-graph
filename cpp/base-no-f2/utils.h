#pragma once

#include <vector>
#include <cstdint>

using U64 = std::uint64_t;

// Generate trivial decomposition for n×n matrix multiplication
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


// Check if the scheme correctly implements matrix multiplication
bool verify_scheme(const std::vector<U64>& data, int n) {
    // Number of rank-one tensors
    int r = data.size() / 3;
    
    // Check each Brent equation:    
    for (int i1 = 0; i1 < n; ++i1) {
        for (int i2 = 0; i2 < n; ++i2) {
            for (int j1 = 0; j1 < n; ++j1) {
                for (int j2 = 0; j2 < n; ++j2) {
                    for (int k1 = 0; k1 < n; ++k1) {
                        for (int k2 = 0; k2 < n; ++k2) {
                            // Compute sum over all rank-one tensors
                            int sum = 0;
                            
                            for (int l = 0; l < r; ++l) {
                                U64 u = data[3 * l];
                                U64 v = data[3 * l + 1]; 
                                U64 w = data[3 * l + 2]; // transposed
                                
                                int a = (u >> (i1 * n + i2)) & 1;
                                int b = (v >> (j1 * n + j2)) & 1;
                                int с = (w >> (k1 * n + k2)) & 1;
                                
                                sum ^= (a & b & с);
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

