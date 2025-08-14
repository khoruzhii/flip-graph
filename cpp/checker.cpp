#include "scheme.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <cstdint>
#include <sstream>
#include <iomanip>

using U64 = std::uint64_t;

// Helper: load 1D npy of uint64_t
static bool load_npy_u64_1d(const std::string& path, std::vector<U64>& out) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "File not found: " << path << "\n";
        return false;
    }
    cnpy::NpyArray arr = cnpy::npy_load(path);
    // Expect uint64 and 1D; cnpy stores dtype via word_size only
    if (arr.word_size != sizeof(U64)) {
        std::cerr << "Unexpected dtype size in " << path
                  << " (expected " << sizeof(U64) << " bytes)\n";
        return false;
    }
    size_t count = 1;
    for (auto s : arr.shape) count *= s;
    const U64* ptr = arr.data<U64>();
    out.assign(ptr, ptr + count);
    return true;
}

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph Random Walk Stats (C3 symmetry)"};

    // Walk parameters (same protocol)
    int flip_lim = 100000000;  // max flips per attempt
    int plus_lim = 50000;      // flips without improvement before plus()

    // Attempts / seeds
    int attempts = 1;                 // -n : number of independent runs
    std::vector<int> seed_list;       // explicit seeds
    int seed_start = 100;             // starting seed when generating attempts

    // Input scheme
    std::string data_dir = "../data/schemes/555";  // as requested
    std::string npy_name;                          // required .npy filename inside data_dir

    // CLI
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per attempt")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(50000);
    app.add_option("-n,--attempts", attempts, "Number of attempts (seeds)")->default_val(1)->check(CLI::PositiveNumber);

    auto* seed_opt = app.add_option("--seeds", seed_list, "Explicit list of seeds to attempt");
    app.add_option("--seed-start", seed_start, "Starting seed when generating attempts")->default_val(100)->excludes(seed_opt);

    app.add_option("--data-dir", data_dir, "Directory with scheme npy files (default ../data/schemes/555)");
    app.add_option("--npy", npy_name, "Filename of the .npy scheme inside data-dir")->required();

    CLI11_PARSE(app, argc, argv);

    // Build list of seeds to try
    std::vector<int> seeds_to_run;
    if (!seed_list.empty()) {
        seeds_to_run = seed_list;
    } else {
        seeds_to_run.reserve(attempts);
        for (int i = 0; i < attempts; ++i) seeds_to_run.push_back(seed_start + i);
    }
    if ((int)seeds_to_run.size() > attempts) seeds_to_run.resize(attempts);

    // Load base scheme from npy
    std::string npy_path = (std::filesystem::path(data_dir) / npy_name).string();
    std::vector<U64> base_data;
    if (!load_npy_u64_1d(npy_path, base_data)) {
        std::cerr << "Failed to load base scheme from " << npy_path << "\n";
        return 1;
    }

    // Fixed partition rank offset, as in the original experiment
    const int partition_rank = 3;

    // Aggregation
    std::map<int, std::uint64_t> best_rank_hist; // best reached rank -> count
    long double sum_best_rank = 0.0L;
    long double sum_delta     = 0.0L;

    // No verbose/progress printing; run attempts silently
    for (size_t run = 0; run < seeds_to_run.size(); ++run) {
        int seed = seeds_to_run[run];

        // Create a fresh Scheme from the same base state for each run
        Scheme scheme(base_data, 3, static_cast<uint32_t>(seed));

        // Track ranks
        int start_rank = partition_rank + scheme.get_rank();
        int best_rank  = start_rank;
        int current_rank = start_rank;
        int flips_since_improvement = 0;

        for (int i = 1; i <= flip_lim; ++i) {
            // Flip; if no valid flips remain, stop this attempt
            if (!scheme.flip()) {
                break;
            }

            current_rank = partition_rank + scheme.get_rank();
            if (current_rank < best_rank) {
                best_rank = current_rank;
                flips_since_improvement = 0;
            } else {
                ++flips_since_improvement;
            }

            // If stuck, perform plus() transition
            if (flips_since_improvement >= plus_lim) {
                if (scheme.plus()) {
                    current_rank = partition_rank + scheme.get_rank();
                    if (current_rank < best_rank) best_rank = current_rank;
                    flips_since_improvement = 0;
                }
            }
        }

        // Aggregate stats for this run
        ++best_rank_hist[best_rank];
        sum_best_rank += static_cast<long double>(best_rank);
        sum_delta     += static_cast<long double>(start_rank - best_rank);
    }

    // Print final aggregated statistics
    const long double N = static_cast<long double>(seeds_to_run.size());
    long double mean_best  = (N > 0 ? sum_best_rank / N : 0.0L);
    long double mean_delta = (N > 0 ? sum_delta     / N : 0.0L);

    std::cout << "=== Random Walk Stats (5x5, C3) ===\n";
    std::cout << "Scheme: " << npy_path << "\n";
    std::cout << "Attempts: " << seeds_to_run.size()
              << ", flip_lim=" << flip_lim << ", plus_lim=" << plus_lim << "\n\n";

    std::cout << "Histogram of best reached ranks:\n";
    for (const auto& [rank, cnt] : best_rank_hist) {
        std::cout << "  rank " << std::setw(3) << rank << " : " << cnt << "\n";
    }
    std::cout << "\n";
    std::cout << "Mean best rank : " << std::fixed << std::setprecision(3) << static_cast<double>(mean_best)  << "\n";
    std::cout << "Mean delta     : " << std::fixed << std::setprecision(3) << static_cast<double>(mean_delta) << " (start - best)\n";

    return 0;
}
