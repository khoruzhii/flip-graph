#include <iostream>
#include <cstdint>
#include <chrono>
#include <random>
#include "field.h"
#include "scheme.h"

#if   defined(SYM)
    #include "utils_sym.h"
#elif defined(ACOM) 
    #include "utils_acom.h"
#else
    #include "utils.h"
#endif

#ifdef MOD3
    using FieldType = B3;
#else
    using FieldType = B2;
#endif

using U32 = std::uint32_t;

// Simple data printer for debugging
void print_data(const std::vector<U64>& data) {
    for (int i = 0; i < data.size(); ++i)
        std::cout << data[i] << ",";
    std::cout << "\n";
}

int main() {
    int n = 2;
    int N = 1e7;  // Number of flips
    
    // Generate trivial decomposition
    auto data = generate_trivial_decomposition(n);
    // print_data(data);
    
    // for (int j = 0; j < 100; ++j) {  // Uncomment for multiple runs
    
    U32 seed = std::random_device{}();
    Scheme<FieldType> scheme(data, seed);
    
    std::cout << "Initial rank = " << scheme.get_rank() << ", correct = " << verify_scheme(scheme.get_data_field(), n) << "\n";
    
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