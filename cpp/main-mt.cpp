#include "scheme.h"
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

using U64 = std::uint64_t;

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph Search with C3 symmetry (multithreaded, no saving)"};

    // Core parameters
    int flip_lim = 100000000;
    int plus_lim = 50000;
    int threads  = 1;

    // Seed options (mutually exclusive)
    std::vector<int> seed_list;
    int num_runs   = 1;
    int seed_start = 100;

    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(50000);
    app.add_option("-t,--threads", threads, "Number of worker threads")->default_val(1)->check(CLI::PositiveNumber);

    auto* seed_opt = app.add_option("-s,--seeds", seed_list, "List of seeds to run");
    auto* runs_opt = app.add_option("-n,--num-runs", num_runs, "Number of runs (starting from seed-start)")
                        ->default_val(1)->check(CLI::PositiveNumber);
    seed_opt->excludes(runs_opt);
    runs_opt->excludes(seed_opt);
    app.add_option("--seed-start", seed_start, "Starting seed for --num-runs")->default_val(100)->needs(runs_opt);

    CLI11_PARSE(app, argc, argv);

    // Build list of seeds
    std::vector<int> seeds_to_run;
    if (!seed_list.empty()) {
        seeds_to_run = seed_list;
    } else {
        seeds_to_run.reserve(num_runs);
        for (int i = 0; i < num_runs; ++i) {
            seeds_to_run.push_back(seed_start + i);
        }
    }
    if (seeds_to_run.empty()) {
        std::cout << "No seeds to run.\n";
        return 0;
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

    std::cout << "=== 5x5 Flip Graph Search (C3) ===\n";
    std::cout << "Flip limit: " << flip_lim << ", Plus limit: " << plus_lim << ", Threads: " << threads << "\n";
    std::cout << "Seeds: ";
    for (size_t i = 0; i < seeds_to_run.size(); ++i) {
        std::cout << seeds_to_run[i] << (i + 1 < seeds_to_run.size() ? ", " : "");
    }
    std::cout << "\n\n";

    const size_t total_tasks = seeds_to_run.size();
    const int worker_count = std::max(1, std::min<int>(threads, static_cast<int>(total_tasks)));

    std::atomic<size_t> next_index{0};
    std::atomic<unsigned long long> total_flips{0};

    std::mutex cout_mtx;

    // Each task will write its best rank into this vector at its own index
    std::vector<int> best_ranks(total_tasks, -1);

    auto t_all_start = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            size_t idx = next_index.fetch_add(1);
            if (idx >= total_tasks) break;

            const int seed = seeds_to_run[idx];

            const auto t0 = std::chrono::steady_clock::now();

            // Each task builds its own Scheme
            Scheme scheme(initial, 3, static_cast<uint32_t>(seed));

            // Partition rank is a fixed offset used in this experiment
            const int partition_rank = 3;
            int best_rank = partition_rank + scheme.get_rank();
            int current_rank = best_rank;
            int flips_since_improvement = 0;

            unsigned long long local_flips = 0;

            for (int i = 0; i < flip_lim; ++i) {
                if (!scheme.flip()) break;
                ++local_flips;

                current_rank = partition_rank + scheme.get_rank();
                if (current_rank < best_rank) {
                    best_rank = current_rank;
                    flips_since_improvement = 0;
                } else {
                    ++flips_since_improvement;
                }

                if (flips_since_improvement >= plus_lim) {
                    if (scheme.plus()) {
                        current_rank = partition_rank + scheme.get_rank();
                        flips_since_improvement = 0;
                    }
                }
            }

            const auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            if (secs <= 0.0) secs = 1e-9; // guard

            best_ranks[idx] = best_rank;
            total_flips.fetch_add(local_flips, std::memory_order_relaxed);

            // Single output block per task
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << "seed=" << seed
                << ", best_rank=" << best_rank
                << ", time=" << std::setprecision(3) << secs << "s\n";

            std::lock_guard<std::mutex> lk(cout_mtx);
            std::cout << oss.str();
        }
    };

    // Launch workers
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (int w = 0; w < worker_count; ++w) {
        pool.emplace_back(worker);
    }
    for (auto& th : pool) th.join();

    auto t_all_end = std::chrono::steady_clock::now();
    double all_secs = std::chrono::duration<double>(t_all_end - t_all_start).count();
    if (all_secs <= 0.0) all_secs = 1e-9; // guard

    // Build summary of best ranks
    std::map<int, int> rank_counts;
    for (int r : best_ranks) {
        if (r >= 0) rank_counts[r]++;
    }

    // Print summary
    if (rank_counts.size() > 1 || total_tasks > 1) {
        std::cout << "\n=== Summary of " << total_tasks << " runs ===\n";
        std::cout << "Best ranks achieved:\n";
        for (const auto& [r, cnt] : rank_counts) {
            std::cout << "  Rank " << r << ": " << cnt << (cnt == 1 ? " run" : " runs") << "\n";
        }
        int overall_best = rank_counts.begin()->first;
        std::cout << "Overall best rank: " << overall_best << "\n";
    }

    // Effective flips throughput
    double mps = static_cast<double>(total_flips.load()) / all_secs / 1e6;
    std::cout << "\nAll tasks completed in " << std::fixed << std::setprecision(3) << all_secs << "s"
              << ", effective rate=" << std::setprecision(1) << mps << " M/s\n";

    return 0;
}
