#include "scheme.h"
#include "cnpy.h"
#include "CLI11.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>
#include <map>

using U16 = std::uint16_t;
using U64 = std::uint64_t;

void print_data(const std::vector<U64>& data) {
    std::cout << "data:\n{";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << (i < data.size() - 1 ? ", " : "");
    }
    std::cout << "}\n";
}

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph Search with C3 symmetry"};
    
    // Parameters with defaults
    int flip_lim = 100000000;
    int plus_lim = 50000;
    int save_flag = 0;
    int verbose = 1;
    int progress_interval = 10000000;
    std::string suffix = "";
    
    // Seed options (mutually exclusive)
    std::vector<int> seed_list;
    int num_runs = 1;
    int seed_start = 100;
    
    // Options
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(50000);
    app.add_option("--save", save_flag, "Save mode: 0=none, 1=rank, 2=data, 3=both")->default_val(0)->check(CLI::Range(0, 3));
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=silent, 1=progress, 2=data, 3=both")->default_val(1)->check(CLI::Range(0, 3));
    app.add_option("--progress-interval", progress_interval, "Show progress every N flips")->default_val(10000000);
    app.add_option("--suffix", suffix, "Filename suffix for saved data")->default_str("");
    
    // Mutually exclusive seed options
    auto* seed_opt = app.add_option("-s,--seeds", seed_list, "List of seeds to run");
    auto* runs_opt = app.add_option("-n,--num-runs", num_runs, "Number of runs (starting from seed 100)")->default_val(1);
    seed_opt->excludes(runs_opt);
    runs_opt->excludes(seed_opt);
    app.add_option("--seed-start", seed_start, "Starting seed for --num-runs")->default_val(100)->needs(runs_opt);
    
    CLI11_PARSE(app, argc, argv);
    
    // Prepare list of seeds to run
    std::vector<int> seeds_to_run;
    if (!seed_list.empty()) {
        seeds_to_run = seed_list;
    } else {
        for (int i = 0; i < num_runs; i++) {
            seeds_to_run.push_back(seed_start + i);
        }
    }
    
    // Starting point for 5x5 matrix with C3 symmetry
    std::vector<U64> data = {
        1, 1, 16777216, 1, 2, 32, 1, 4, 1024, 1, 8, 32768, 1, 16, 1048576, 1, 16777216, 16777216,
        2, 64, 32, 2, 128, 1024, 2, 256, 32768, 2, 512, 1048576, 4, 2048, 32, 4, 4096, 1024,
        4, 8192, 32768, 4, 16384, 1048576, 8, 65536, 32, 8, 131072, 1024, 8, 262144, 32768,
        8, 524288, 1048576, 16, 2097152, 32, 16, 4194304, 1024, 16, 8388608, 32768, 16, 16777216, 1048576,
        64, 64, 262144, 64, 128, 2048, 64, 256, 65536, 64, 512, 2097152, 64, 262144, 262144,
        128, 4096, 2048, 128, 8192, 65536, 128, 16384, 2097152, 256, 131072, 2048, 256, 262144, 65536,
        256, 524288, 2097152, 512, 4194304, 2048, 512, 8388608, 65536, 512, 16777216, 2097152, 4096, 8192, 131072,
        4096, 16384, 4194304, 8192, 262144, 131072, 8192, 524288, 4194304, 16384, 8388608, 131072, 16384, 16777216, 4194304,
        262144, 524288, 8388608, 524288, 16777216, 8388608,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    
    std::cout << "=== 5x5 Flip Graph Search ===\n";
    std::cout << "Flip limit: " << flip_lim << ", Plus limit: " << plus_lim << "\n";
    std::cout << "Save mode: ";
    if (save_flag == 0) std::cout << "none";
    else if (save_flag == 1) std::cout << "rank only";
    else if (save_flag == 2) std::cout << "data only";
    else if (save_flag == 3) std::cout << "rank + data";
    std::cout << ", Verbose: " << verbose << "\n";
    std::cout << "Running " << seeds_to_run.size() << " seed(s): ";
    for (size_t i = 0; i < seeds_to_run.size(); ++i) {
        std::cout << seeds_to_run[i] << (i + 1 < seeds_to_run.size() ? ", " : "");
    }
    std::cout << "\n\n";
    
    // Track best ranks across all runs for summary
    std::map<int, int> rank_counts;
    
    // Run multiple searches with different seeds
    for (size_t run = 0; run < seeds_to_run.size(); run++) {
        int seed = seeds_to_run[run];
        
        // Initialize scheme with current seed
        Scheme scheme(data, 3, static_cast<uint32_t>(seed));
        
        // Partition rank is a fixed offset used in this experiment
        int partition_rank = 3;
        int initial_rank = partition_rank + scheme.get_rank();
        int best_rank = initial_rank;
        int current_rank = initial_rank;
        int flips_since_improvement = 0;
        int total_plus = 0;
        
        std::cout << "Run " << (run + 1) << "/" << seeds_to_run.size()
                  << " (seed=" << seed << ")\n";
        std::cout << "Starting rank: " << initial_rank
                  << " (partition=" << partition_rank
                  << ", rank=" << scheme.get_rank() << ")\n";
        
        // Track rank per step
        std::vector<U16> rank_log;
        if (save_flag & 1) {
            rank_log.reserve(flip_lim);
        }
        
        // Track data states
        std::vector<U64> data_log;
        if (save_flag & 2) {
            data_log.reserve(static_cast<size_t>(flip_lim) * data.size());
        }
        
        auto time_start = std::chrono::high_resolution_clock::now();
        
        // Main search loop
        for (int i = 0; i < flip_lim; i++) {
            // Try flip
            if (!scheme.flip()) {
                std::cout << "No valid flips at " << i << "\n";
                break;
            }
            
            current_rank = partition_rank + scheme.get_rank();
            
            // Log rank if needed
            if (save_flag & 1) {
                rank_log.push_back(static_cast<U16>(scheme.get_rank()));
            }
            
            // Log data if needed
            if (save_flag & 2) {
                const auto& current_data = scheme.get_data();
                data_log.insert(data_log.end(), current_data.begin(), current_data.end());
            }
            
            // Check improvement
            if (current_rank < best_rank) {
                best_rank = current_rank;
                flips_since_improvement = 0;
            } else {
                flips_since_improvement++;
            }
            
            // Plus transition if stuck
            if (flips_since_improvement >= plus_lim) {
                if (scheme.plus()) {
                    total_plus++;
                    current_rank = partition_rank + scheme.get_rank();
                    flips_since_improvement = 0;
                }
            }
            
            // Progress report
            if ((verbose & 1) && i > 0 && i % progress_interval == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                // Use high-resolution floating seconds to prevent division by zero (inf M/s)
                double elapsed = std::chrono::duration<double>(now - time_start).count();
                if (elapsed <= 0.0) elapsed = 1e-9; // guard against zero
                double mps = static_cast<double>(i) / elapsed / 1e6;
                std::cout << "  " << std::setw(4) << i/1000000 << "M flips, "
                          << std::fixed << std::setprecision(1)
                          << mps << "M/s, best=" << best_rank
                          << ", current=" << current_rank << "\n";
            }
        }
        
        auto time_end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(time_end - time_start).count();
        
        // Save rank log if needed
        if (save_flag & 1) {
            std::string filename = "../data/rank-" + std::to_string(seed);
            if (!suffix.empty()) filename += "-" + suffix;
            filename += ".npy";
            cnpy::npy_save(filename, rank_log);
            std::cout << "Saved rank to " << filename << "\n";
        }
        
        // Save data log if needed
        if (save_flag & 2) {
            std::string filename = "../data/data-" + std::to_string(seed);
            if (!suffix.empty()) filename += "-" + suffix;
            filename += ".npy";
            
            // Reshape data_log as 2D array (num_steps x data_size)
            size_t num_steps = data_log.size() / data.size();
            std::vector<size_t> shape = {num_steps, data.size()};
            cnpy::npy_save(filename, data_log.data(), shape);
            std::cout << "Saved data (" << num_steps << " steps) to " << filename << "\n";
        }
        
        // Summary for this run
        std::cout << "Completed: best_rank=" << best_rank
                  << ", time=" << std::fixed << std::setprecision(1) << runtime << "s\n";
        
        // Show final data if verbose has data bit
        if (verbose & 2) {
            print_data(scheme.get_data());
        }
        
        // Track best rank for summary
        rank_counts[best_rank]++;
        
        std::cout << "\n";
    }
    
    // Print summary if multiple runs
    if (seeds_to_run.size() > 1) {
        std::cout << "=== Summary of " << seeds_to_run.size() << " runs ===\n";
        std::cout << "Best ranks achieved:\n";
        for (const auto& [r, count] : rank_counts) {
            std::cout << "  Rank " << r << ": " << count
                      << (count == 1 ? " run" : " runs") << "\n";
        }
        int overall_best = rank_counts.begin()->first; // lowest rank
        std::cout << "Overall best rank: " << overall_best << "\n";
    }
    
    return 0;
}
