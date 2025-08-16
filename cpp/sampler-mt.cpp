#include "scheme.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <chrono>
#include <cstdint>
#include <map>
#include <iomanip>
#include <filesystem>
#include <algorithm>

using U64 = std::uint64_t;

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph Sampler (C3 symmetry, multithreaded, 2D .npy output)"};

    // Core parameters
    int flip_lim = 100000000;          // max flips per attempt
    int plus_lim = 100000;              // flips without improvement before plus()
    int threads  = 1;                  // worker threads
    int verbose  = 0;                  // 0=silent, 1=progress

    // Sampling targets
    int target_rank  = -1;             // required: rank to hit
    int attempts     = 1;              // maximum number of attempts (seeds)
    int samples_need = 1;              // how many successful samples to collect

    // Seeding
    std::vector<int> seed_list;        // explicit list of seeds
    int seed_start = 100000;           // starting seed when generating attempts

    // Output directory (fixed)
    const std::string out_dir = "../data/schemes/555";

    // CLI options
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per attempt")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(100000);
    app.add_option("-t,--threads", threads, "Number of worker threads")->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=silent, 1=progress")->default_val(0)->check(CLI::Range(0, 1));

    app.add_option("-r,--target-rank", target_rank, "Target rank to sample (required)")->required();
    app.add_option("-n,--attempts", attempts, "Maximum number of attempts (seeds)")->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("-s,--samples", samples_need, "Number of successful samples to collect")->default_val(1)->check(CLI::PositiveNumber);

    auto* seed_opt = app.add_option("--seeds", seed_list, "Explicit list of seeds to attempt");
    app.add_option("--seed-start", seed_start, "Starting seed when generating attempts")->default_val(100000)->excludes(seed_opt);

    CLI11_PARSE(app, argc, argv);

    // Build list of seeds to try
    std::vector<int> seeds_to_run;
    if (!seed_list.empty()) {
        seeds_to_run = seed_list;
    } else {
        seeds_to_run.reserve(attempts);
        for (int i = 0; i < attempts; ++i) seeds_to_run.push_back(seed_start + i);
    }
    if (seeds_to_run.empty()) {
        std::cout << "No seeds to run.\n";
        return 0;
    }

    // Ensure output directory exists
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Failed to create output directory: " << out_dir << " (" << ec.message() << ")\n";
        return 1;
    }

    // Initial data for 5x5 with C3 symmetry
    const std::vector<U64> initial = {
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

    // Prepare flat buffer for collected samples (1D backing storage).
    // It will be saved as a 2D array with shape [actual, data_size].
    const size_t data_size = initial.size();
    std::vector<U64> flat_data(static_cast<size_t>(samples_need) * data_size, 0);

    std::cout << "=== 5x5 Flip Graph Sampler (C3, multithreaded) ===\n";
    std::cout << "Target rank: " << target_rank
              << ", Attempts: " << seeds_to_run.size()
              << ", Samples needed: " << samples_need << "\n";
    std::cout << "Flip limit: " << flip_lim
              << ", Plus limit: " << plus_lim
              << ", Threads: " << threads << "\n";
    std::cout << "Output dir: " << out_dir << "\n\n";

    // Threading setup
    const size_t total_tasks = seeds_to_run.size();
    const int worker_count = std::max(1, std::min<int>(threads, static_cast<int>(total_tasks)));

    std::atomic<size_t> next_index{0};
    std::atomic<int> samples_collected{0};
    std::atomic<unsigned long long> total_flips{0};
    std::atomic<bool> done{false};

    std::mutex cout_mtx;

    auto t_all_start = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            if (done.load(std::memory_order_relaxed)) break;

            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_tasks) break;

            const int seed = seeds_to_run[idx];
            const auto t0 = std::chrono::steady_clock::now();

            // Each task builds its own Scheme
            Scheme scheme(initial, 3, static_cast<uint32_t>(seed));

            // Partition rank is a fixed offset used in this experiment
            const int partition_rank = 3;
            int current_rank = partition_rank + scheme.get_rank();
            int best_rank = current_rank;
            int flips_since_improvement = 0;
            bool hit_target = (current_rank == target_rank);

            unsigned long long local_flips = 0;

            for (int i = 0; i < flip_lim && !hit_target && !done.load(std::memory_order_relaxed); ++i) {
                if (!scheme.flip()) break;
                ++local_flips;

                current_rank = partition_rank + scheme.get_rank();
                if (current_rank < best_rank) {
                    best_rank = current_rank;
                    flips_since_improvement = 0;
                } else {
                    ++flips_since_improvement;
                }

                if (current_rank == target_rank) {
                    hit_target = true;
                    break;
                }

                if (flips_since_improvement >= plus_lim) {
                    if (scheme.plus()) {
                        current_rank = partition_rank + scheme.get_rank();
                        flips_since_improvement = 0;
                        if (current_rank == target_rank) {
                            hit_target = true;
                            break;
                        }
                    }
                }
            }

            const auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            if (secs <= 0.0) secs = 1e-9; // guard

            total_flips.fetch_add(local_flips, std::memory_order_relaxed);

            if (hit_target) {
                // Reserve a slot for this sample
                int slot = samples_collected.fetch_add(1, std::memory_order_acq_rel);
                if (slot < samples_need) {
                    // Copy current data into flat buffer at position [slot, :]
                    const auto& cur = scheme.get_data();
                    std::copy(cur.begin(), cur.end(), flat_data.begin() + static_cast<size_t>(slot) * data_size);

                    if (verbose) {
                        std::lock_guard<std::mutex> lk(cout_mtx);
                        std::cout << "seed=" << seed
                                  << " hit target, slot=" << slot
                                  << ", time=" << std::fixed << std::setprecision(3) << secs << "s\n";
                    }
                    // If last required slot is filled, signal early stop
                    if (slot + 1 >= samples_need) {
                        done.store(true, std::memory_order_relaxed);
                    }
                } else {
                    if (verbose) {
                        std::lock_guard<std::mutex> lk(cout_mtx);
                        std::cout << "seed=" << seed
                                  << " hit target after quota filled, ignoring. time="
                                  << std::fixed << std::setprecision(3) << secs << "s\n";
                    }
                }
            } else {
                if (verbose) {
                    std::lock_guard<std::mutex> lk(cout_mtx);
                    std::cout << "seed=" << seed
                              << " did not reach target (best=" << best_rank
                              << "), time=" << std::fixed << std::setprecision(3) << secs << "s\n";
                }
            }
        }
    };

    // Launch workers
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (int w = 0; w < worker_count; ++w) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    auto t_all_end = std::chrono::steady_clock::now();
    double all_secs = std::chrono::duration<double>(t_all_end - t_all_start).count();
    if (all_secs <= 0.0) all_secs = 1e-9; // guard

    // Determine actual number of collected samples (clamped)
    int actual = std::min(samples_collected.load(std::memory_order_acquire), samples_need);

    // Resize backing storage to exactly actual rows
    flat_data.resize(static_cast<size_t>(actual) * data_size);

    // Save as 2D array if anything collected
    if (actual > 0) {
        // Filename: {target_rank:03d}-{samples_colected:06d}.npy in ../data/schemes/555
        std::ostringstream oss;
        oss << out_dir << "/";
        oss << std::setfill('0') << std::setw(3) << target_rank;
        oss << "-";
        oss << std::setfill('0') << std::setw(6) << actual; // samples_colected count
        oss << ".npy";
        const std::string filename = oss.str();

        // 2D shape: [actual, data_size] (row-major)
        std::vector<size_t> shape = { static_cast<size_t>(actual), data_size };
        cnpy::npy_save(filename, flat_data.data(), shape);

        std::cout << "\nSaved: " << filename
                  << "  (shape=[" << actual << ", " << data_size << "])\n";
    } else {
        std::cout << "\nNo samples of target rank were collected. Nothing saved.\n";
    }

    // Throughput
    double mps = static_cast<double>(total_flips.load()) / all_secs / 1e6;
    std::cout << "All tasks completed in " << std::fixed << std::setprecision(3) << all_secs << "s"
              << ", effective rate=" << std::setprecision(1) << mps << " M/s\n";

    return 0;
}
