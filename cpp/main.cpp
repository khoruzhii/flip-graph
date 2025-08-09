#include "scheme.h"
#include "cnpy.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>

using U16 = std::uint16_t;
using U64 = std::uint64_t;

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <flip_limit> <plus_limit> <num_runs> [suffix]\n";
        return 1;
    }
    
    const int L = std::atoi(argv[1]);           // Total flip limit
    const int PLUS_LIM = std::atoi(argv[2]);    // Flips without improvement before plus
    const int NUM_RUNS = std::atoi(argv[3]);    // Number of runs with different seeds
    const std::string SUFFIX = (argc > 4) ? argv[4] : "";  // Optional filename suffix
    
    // Starting point for 5x5 matrix with C3 symmetry
    // std::vector<U64> data = {1, 1, 16777216, 1, 2, 32, 1, 4, 1024, 1, 8, 32768, 1, 16, 1048576, 1, 16777216, 16777216, 2, 64, 32, 2, 128, 1024, 2, 256, 32768, 2, 512, 1048576, 4, 2048, 32, 4, 4096, 1024, 4, 8192, 32768, 4, 16384, 1048576, 8, 65536, 32, 8, 131072, 1024, 8, 262144, 32768, 8, 524288, 1048576, 16, 2097152, 32, 16, 4194304, 1024, 16, 8388608, 32768, 16, 16777216, 1048576, 64, 64, 262144, 64, 128, 2048, 64, 256, 65536, 64, 512, 2097152, 64, 262144, 262144, 128, 4096, 2048, 128, 8192, 65536, 128, 16384, 2097152, 256, 131072, 2048, 256, 262144, 65536, 256, 524288, 2097152, 512, 4194304, 2048, 512, 8388608, 65536, 512, 16777216, 2097152, 4096, 8192, 131072, 4096, 16384, 4194304, 8192, 262144, 131072, 8192, 524288, 4194304, 16384, 8388608, 131072, 16384, 16777216, 4194304, 262144, 524288, 8388608, 524288, 16777216, 8388608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<U64> data = {1, 1, 64, 1, 1, 4096, 1, 1, 262144, 1, 1, 16777216, 1, 2, 32, 1, 4, 1024, 1, 8, 32768, 1, 16, 1048576, 1, 64, 64, 1, 64, 4096, 1, 64, 262144, 1, 64, 16777216, 1, 4096, 64, 1, 4096, 4096, 1, 4096, 262144, 1, 4096, 16777216, 1, 262144, 64, 1, 262144, 4096, 1, 262144, 262144, 1, 262144, 16777216, 1, 16777216, 64, 1, 16777216, 4096, 1, 16777216, 262144, 1, 16777216, 16777216, 2, 64, 32, 2, 128, 1024, 2, 256, 32768, 2, 512, 1048576, 4, 2048, 32, 4, 4096, 1024, 4, 8192, 32768, 4, 16384, 1048576, 8, 65536, 32, 8, 131072, 1024, 8, 262144, 32768, 8, 524288, 1048576, 16, 2097152, 32, 16, 4194304, 1024, 16, 8388608, 32768, 16, 16777216, 1048576, 64, 64, 4096, 64, 64, 262144, 64, 64, 16777216, 64, 128, 2048, 64, 256, 65536, 64, 512, 2097152, 64, 4096, 4096, 64, 4096, 262144, 64, 4096, 16777216, 64, 262144, 4096, 64, 262144, 262144, 64, 262144, 16777216, 64, 16777216, 4096, 64, 16777216, 262144, 64, 16777216, 16777216, 128, 4096, 2048, 128, 8192, 65536, 128, 16384, 2097152, 256, 131072, 2048, 256, 262144, 65536, 256, 524288, 2097152, 512, 4194304, 2048, 512, 8388608, 65536, 512, 16777216, 2097152, 4096, 4096, 262144, 4096, 4096, 16777216, 4096, 8192, 131072, 4096, 16384, 4194304, 4096, 262144, 262144, 4096, 262144, 16777216, 4096, 16777216, 262144, 4096, 16777216, 16777216, 8192, 262144, 131072, 8192, 524288, 4194304, 16384, 8388608, 131072, 16384, 16777216, 4194304, 262144, 262144, 16777216, 262144, 524288, 8388608, 262144, 16777216, 16777216, 524288, 16777216, 8388608};
    
    std::cout << "=== 5x5 Flip Graph Search ===\n";
    std::cout << "Flip limit: " << L << ", Plus limit: " << PLUS_LIM << "\n";
    std::cout << "Running " << NUM_RUNS << " seeds starting from 42\n\n";
    
    // Run multiple searches with different seeds
    for (int run = 0; run < NUM_RUNS; run++) {
        int seed = 42 + run;
        
        // Initialize scheme with current seed
        Scheme scheme(data, seed);
        
        int partition_rank = 1;
        int initial_rank = partition_rank + 3 * scheme.get_orank();
        int best_rank = initial_rank;
        int current_rank = initial_rank;
        int flips_since_improvement = 0;
        int total_plus = 0;
        
        std::cout << "Run " << (run + 1) << "/" << NUM_RUNS << " (seed=" << seed << ")\n";
        
        // Track orbit ranks
        std::vector<U16> orank_log;
        orank_log.reserve(L);
        
        auto time_start = std::chrono::high_resolution_clock::now();
        
        // Main search loop
        for (int i = 0; i < L; i++) {
            // Try flip
            if (!scheme.flip()) {
                std::cout << "No valid flips at " << i << "\n";
                break;
            }
            
            current_rank = partition_rank + 3 * scheme.get_orank();
            orank_log.push_back(scheme.get_orank());
            
            // Check improvement
            if (current_rank < best_rank) {
                best_rank = current_rank;
                flips_since_improvement = 0;
                // std::cout << "rank=" << best_rank  << " (orank=" << scheme.get_orank() << ")\n";
            } else {
                flips_since_improvement++;
            }
            
            // Plus transition if stuck
            if (flips_since_improvement >= PLUS_LIM) {
                if (scheme.plus()) {
                    total_plus++;
                    current_rank = partition_rank + 3 * scheme.get_orank();
                    flips_since_improvement = 0;
                }
            }
            
            // Progress report
            if (i > 0 && i % 10000000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - time_start).count();
                std::cout << "  " << i/1000000 << "M flips, " << std::fixed << std::setprecision(1) 
                          << i/elapsed/1e6 << "M/s, best=" << best_rank << "\n";
            }
        }
        
        auto time_end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count();
        
        // Save orbit rank log
        std::string filename = "../data/orank-" + std::to_string(seed);
        if (!SUFFIX.empty()) filename += "-" + SUFFIX;
        filename += ".npy";
        cnpy::npy_save(filename, orank_log);
        
        // Summary for this run
        std::cout << "Completed: best_rank=" << best_rank 
                  << ", time=" << std::fixed << std::setprecision(1) << runtime << "s"
                  << ", saved to " << filename << "\n\n";
    }
    
    return 0;
}