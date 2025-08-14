#include "scheme.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>
#include <map>
#include <filesystem>
#include <sstream>
#include <cstdint>

using U16 = std::uint16_t;
using U64 = std::uint64_t;

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph Sampler (C3 symmetry)"};

    // General search parameters
    int flip_lim = 100000000;     // max flips per attempt
    int plus_lim = 50000;         // flips without improvement before plus()
    int verbose  = 1;             // 0=silent, 1=progress
    int progress_interval = 10000000;

    // Sampling targets
    int target_rank = -1;        // required: rank to hit
    int attempts     = 1;         // -n : how many attempts (seeds) to try
    int samples_need = 1;         // how many successful samples to collect

    // Seeding
    std::vector<int> seed_list;   // optional explicit list of seeds
    int seed_start = 100;         // starting seed for generated attempts

    // Output
    std::string out_dir = "../data/schemes/555"; // fixed as requested

    // CLI options
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per attempt")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(50000);
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=silent, 1=progress")->default_val(1)->check(CLI::Range(0, 1));
    app.add_option("--progress-interval", progress_interval, "Show progress every N flips")->default_val(10000000);

    app.add_option("-r,--target-rank", target_rank, "Target rank to sample (required)")->required();
    app.add_option("-n,--attempts", attempts, "Maximum number of attempts (seeds)")->default_val(1);
    app.add_option("-s,--samples", samples_need, "Number of successful samples to collect")->default_val(1)->check(CLI::PositiveNumber);

    auto* seed_opt = app.add_option("--seeds", seed_list, "Explicit list of seeds to attempt");
    app.add_option("--seed-start", seed_start, "Starting seed when generating attempts")->default_val(100)->excludes(seed_opt);

    app.add_option("--out-dir", out_dir, "Output directory (default ../data/schemes/555)");

    CLI11_PARSE(app, argc, argv);

    // Build list of seeds to try
    std::vector<int> seeds_to_run;
    if (!seed_list.empty()) {
        seeds_to_run = seed_list;
    } else {
        seeds_to_run.reserve(attempts);
        for (int i = 0; i < attempts; ++i) seeds_to_run.push_back(seed_start + i);
    }
    // Trim if attempts < seed_list size (when both were given via validation this should not happen, but keep safe)
    if ((int)seeds_to_run.size() > attempts) seeds_to_run.resize(attempts);

    // Ensure output directory exists
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Failed to create output directory: " << out_dir << " (" << ec.message() << ")\n";
        return 1;
    }

    // Initial 5x5 C3-symmetric data
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

    std::cout << "=== 5x5 Flip Graph Sampler ===\n";
    std::cout << "Target orank: " << target_rank << "\n";
    std::cout << "Attempts: " << seeds_to_run.size() << ", Samples needed: " << samples_need << "\n";
    std::cout << "Flip limit: " << flip_lim << ", Plus limit: " << plus_lim << "\n";
    std::cout << "Output dir: " << out_dir << "\n\n";

    int samples_collected = 0;

    for (size_t run = 0; run < seeds_to_run.size(); ++run) {
        if (samples_collected >= samples_need) break;

        int seed = seeds_to_run[run];
        Scheme scheme(data, 3, static_cast<uint32_t>(seed));

        // Partition rank is a fixed offset used in this experiment
        int partition_rank = 3;
        int current_rank   = partition_rank + scheme.get_rank();
        int best_rank      = current_rank;
        int flips_since_improvement = 0;
        int total_plus = 0;

        if (verbose) {
            std::cout << "Attempt " << (run + 1) << "/" << seeds_to_run.size()
                      << " (seed=" << seed << "), start rank = " << current_rank << "\n";
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        bool hit_target = (current_rank == target_rank);

        for (int i = 1; i <= flip_lim && !hit_target; ++i) {
            // Try a flip; if no valid flips remain, bail out this attempt
            if (!scheme.flip()) {
                if (verbose) std::cout << "  No valid flips at step " << i << "\n";
                break;
            }

            current_rank = partition_rank + scheme.get_rank();
            if (current_rank < best_rank) {
                best_rank = current_rank;
                flips_since_improvement = 0;
            } else {
                ++flips_since_improvement;
            }

            // Check target
            if (current_rank == target_rank) {
                hit_target = true;
                break;
            }

            // If stuck, perform plus transition
            if (flips_since_improvement >= plus_lim) {
                if (scheme.plus()) {
                    ++total_plus;
                    current_rank = partition_rank + scheme.get_rank();
                    flips_since_improvement = 0;
                    if (current_rank == target_rank) {
                        hit_target = true;
                        break;
                    }
                }
            }

            // Progress output
            if (verbose && i % progress_interval == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - t0).count();
                if (elapsed <= 0.0) elapsed = 1e-9;
                double mps = static_cast<double>(i) / elapsed / 1e6;
                std::cout << "  " << std::setw(4) << (i / 1000000) << "M flips, "
                          << std::fixed << std::setprecision(1) << mps << "M/s, "
                          << "best=" << best_rank << ", current=" << current_rank << "\n";
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(t1 - t0).count();

        if (hit_target) {
            // Save current data snapshot
            const auto& cur = scheme.get_data();

            // Build filename {rank:03d}-{seed:06d}.npy
            std::ostringstream oss;
            oss << out_dir << "/";
            oss << std::setfill('0') << std::setw(3) << target_rank;
            oss << "-";
            oss << std::setfill('0') << std::setw(6) << seed;
            oss << ".npy";
            std::string filename = oss.str();

            // Save as 1D array
            std::vector<size_t> shape = {cur.size()};
            cnpy::npy_save(filename, cur.data(), shape);

            ++samples_collected;

            std::cout << "Saved: " << filename
                      << "  (rank=" << target_rank << ", seed=" << seed
                      << ", time=" << std::fixed << std::setprecision(2) << runtime << "s)\n";

        } else {
            if (verbose) {
                std::cout << "Attempt with seed " << seed << " did not reach target (best=" << best_rank
                          << "). Time=" << std::fixed << std::setprecision(2) << runtime << "s\n";
            }
        }
    }

    std::cout << "\nDone. Collected " << samples_collected << " / " << samples_need
              << " sample(s) for orank=" << target_rank << ".\n";

    return 0;
}
