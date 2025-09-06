#include <iostream>
#include <cstdint>
#include <chrono>
#include <random>
#include "field.h"
#include "scheme.h"
#include "utils.h"

using U32 = std::uint32_t;

// Choose field type here: B2 for mod2, B3 for mod3
using FieldType = B3;

// Simple data printer for debugging
void print_data(const std::vector<U64>& data) {
    for (int i = 0; i < data.size(); ++i)
        std::cout << data[i] << ",";
    std::cout << "\n";
}

int main() {
    int n = 4;
    int N = 1e7;  // Number of flips
    
    // Generate trivial decomposition
    auto data = generate_trivial_decomposition(n);
    // print_data(data);
    
    // for (int j = 0; j < 100; ++j) {  // Uncomment for multiple runs
    
    U32 seed = std::random_device{}();
    Scheme<FieldType> scheme(data, seed);
    
    // std::cout << "Initial rank = " << scheme.get_rank() << "\n";
    
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
    
    // Verify correctness
    auto correct = verify_scheme(scheme.get_data_field(), n);
    
    // Print results
    std::cout << "Field: " << (field_traits<FieldType>::is_mod2 ? "mod2" : "mod3")
              << ", final rank = " << scheme.get_rank() 
              << ", seed = " << seed 
              << ", correct = " << correct;
    
    if (time > 0) {
        std::cout << ", speed = " << 1e3 / time * N / 1e6 << " M/s";
    }
    std::cout << "\n";
    
    // }  // End multiple runs
    
    std::cout << "Finished.\n";
    return 0;
}