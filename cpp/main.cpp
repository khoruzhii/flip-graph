#include "scheme.h"
#include <iostream>
#include <chrono>

void print_data(const std::vector<U64>& data) {
    std::cout << "{";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << (i < data.size() - 1 ? ", " : "");
    }
    std::cout << "}\n";
}

int main() {
    // Get initial data from start_point
    // std::vector<U64> data = {1, 1, 16, 1, 2, 8, 1, 4, 64, 1, 16, 16, 2, 16, 8, 2, 32, 64, 4, 128, 8, 4, 256, 64, 16, 32, 128, 32, 256, 128};
    std::vector<U64> data = {1, 1, 32, 1, 2, 16, 1, 4, 256, 1, 8, 4096, 1, 32, 32, 2, 32, 16, 2, 64, 256, 2, 128, 4096, 4, 512, 16, 4, 1024, 256, 4, 2048, 4096, 8, 8192, 16, 8, 16384, 256, 8, 32768, 4096, 32, 64, 512, 32, 128, 8192, 64, 1024, 512, 64, 2048, 8192, 128, 16384, 512, 128, 32768, 8192, 1024, 1024, 32768, 1024, 2048, 16384, 1024, 32768, 32768, 2048, 32768, 16384};
    // Create scheme
    Scheme scheme(data, 44);
    
    // Print initial state
    // scheme.print_matches();

    // print_data(scheme.data);
    
    // Perform some flips
    auto time_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        // scheme.flip();
        if (!scheme.flip()) {
            std::cout << "No valid pairs at flip " << i << "\n"; 
            break;
        }
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    
    // // Print final state
    // std::cout << "\nFinal state:\n";
    // scheme.print();
    print_data(scheme.data);
    // std::cout << "Non-zero orbits: " << scheme.count_nonzero() << "\n";
    
    std::cout << "[Total runtime " <<  std::chrono::duration_cast<std::chrono::microseconds>(time_end-time_start).count()/1e3 << "ms]\n";
    return 0;
}