#include "scheme.h"
#include "cnpy.h"
#include <iostream>
#include <chrono>

using U16 = std::uint16_t;

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
    // std::vector<U64> data = {1, 1, 32, 1, 1, 1024, 1, 1, 32768, 1, 2, 16, 1, 4, 256, 1, 8, 4096, 1, 32, 32, 1, 32, 1024, 1, 32, 32768, 1, 1024, 32, 1, 1024, 1024, 1, 1024, 32768, 1, 32768, 32, 1, 32768, 1024, 1, 32768, 32768, 2, 32, 16, 2, 64, 256, 2, 128, 4096, 4, 512, 16, 4, 1024, 256, 4, 2048, 4096, 8, 8192, 16, 8, 16384, 256, 8, 32768, 4096, 32, 32, 1024, 32, 32, 32768, 32, 64, 512, 32, 128, 8192, 32, 1024, 1024, 32, 1024, 32768, 32, 32768, 1024, 32, 32768, 32768, 64, 1024, 512, 64, 2048, 8192, 128, 16384, 512, 128, 32768, 8192, 1024, 1024, 32768, 1024, 2048, 16384, 1024, 32768, 32768, 2048, 32768, 16384};
    std::vector<U64> data = {1, 1, 16777216, 1, 2, 32, 1, 4, 1024, 1, 8, 32768, 1, 16, 1048576, 1, 16777216, 16777216, 2, 64, 32, 2, 128, 1024, 2, 256, 32768, 2, 512, 1048576, 4, 2048, 32, 4, 4096, 1024, 4, 8192, 32768, 4, 16384, 1048576, 8, 65536, 32, 8, 131072, 1024, 8, 262144, 32768, 8, 524288, 1048576, 16, 2097152, 32, 16, 4194304, 1024, 16, 8388608, 32768, 16, 16777216, 1048576, 64, 64, 262144, 64, 128, 2048, 64, 256, 65536, 64, 512, 2097152, 64, 262144, 262144, 128, 4096, 2048, 128, 8192, 65536, 128, 16384, 2097152, 256, 131072, 2048, 256, 262144, 65536, 256, 524288, 2097152, 512, 4194304, 2048, 512, 8388608, 65536, 512, 16777216, 2097152, 4096, 8192, 131072, 4096, 16384, 4194304, 8192, 262144, 131072, 8192, 524288, 4194304, 16384, 8388608, 131072, 16384, 16777216, 4194304, 262144, 524288, 8388608, 524288, 16777216, 8388608, 0, 0, 0};

    // Create scheme
    Scheme scheme(data, 48);
    std::cout << "Orbit rank:  " << scheme.get_orank() << "\n";
    std::cout << "Scheme rank: " << 1 + 3*scheme.get_orank() << "\n";
    
    int L = int(1e7);
    std::vector<U16> orank_log = {};
    orank_log.reserve(L);

    // Perform some flips
    auto time_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < L; i++) {
        // scheme.flip();
        if (!scheme.flip()) {
            std::cout << "No valid pairs at flip " << i << "\n"; 
            break;
        } else {
            orank_log.push_back(scheme.get_orank());
        }
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    
    // Save orank log
    cnpy::npy_save("../data/orank_log_48.npy", orank_log);

    // if (!scheme.plus()) std::cout << "No valid plus." << "\n";
    
    std::cout << "\nOrbit rank:  " << scheme.get_orank() << "\n";
    std::cout << "Scheme rank: " << 3 + 3*scheme.get_orank() << "\n\n";

    print_data(scheme.get_data());
    std::cout << "\n";
    
    std::cout << "[Total runtime " <<  std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count()/1e3 << "s]\n";
    return 0;
}