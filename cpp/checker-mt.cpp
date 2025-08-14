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

// Helper: load 1D npy of uint64_t into vector
static bool load_npy_u64_1d(const std::string& path, std::vector<U64>& out) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "File not found: " << path << "\n";
        return false;
    }
    cnpy::NpyArray arr = cnpy::npy_load(path);
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
    CLI::App app{"5x5 Flip Graph Random Walk Stats (C3 symmetry, multithreaded)"};

    // Walk protocol parameters
    int flip_lim = 100000000;  // total flip limit per attempt
    int plus_lim = 50000;      // flips without improvement before plus()
    int threads  = 1;          // number of worker threads
    int verbose  = 1;          // 0 = do not print per-seed results, 1 = print

    // Seeds: either explicit list or generated [seed_start, seed_start + n)
    std::vector<int> seed_list;
    int num_runs   = 1;
    int seed_start = 100;

    // Input scheme location
    std::string data_dir = "../data/schemes/555";
    std::string npy_name; // required

    // CLI options
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per attempt")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(50000);
    app.add_option("-t,--threads", threads, "Number of worker threads")->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=silent runs, 1=print completed seeds")
       ->default_val(1)->check(CLI::Range(0, 1));

    auto* seed_opt = app.add_option("-s,--seeds", seed_list, "Explicit list of seeds to run");
    auto* runs_opt = app.add_option("-n,--num-runs", num_runs, "Number of runs (generated from seed-start)")
                        ->default_val(1)->check(CLI::PositiveNumber);
    seed_opt->excludes(runs_opt);
    runs_opt->excludes(seed_opt);
    app.add_option("--seed-start", seed_start, "Starting seed for --num-runs")->default_val(100)->needs(runs_opt);

    app.add_option("--data-dir", data_dir, "Directory with scheme npy files (default ../data/schemes/555)");
    app.add_option("--npy", npy_name, "Filename of the .npy scheme inside data-dir")->required();

    CLI11_PARSE(app, argc, argv);

    // Build list of seeds to run
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

    // Load base scheme from .npy
    const std::string npy_path = (std::filesystem::path(data_dir) / npy_name).string();
    std::vector<U64> base_data;
    if (!load_npy_u64_1d(npy_path, base_data)) {
        std::cerr << "Failed to load base scheme from " << npy_path << "\n";
        return 1;
    }

    // Partition rank is a fixed offset used in this experiment
    const int partition_rank = 3;

    // Prepare multithreading
    const size_t total_tasks = seeds_to_run.size();
    const int worker_count = std::max(1, std::min<int>(threads, static_cast<int>(total_tasks)));

    std::atomic<size_t> next_index{0};
    std::atomic<unsigned long long> total_flips{0};
    std::mutex cout_mtx;

    // Collect per-run results for aggregation after join
    std::vector<int> best_ranks(total_tasks, -1);
    std::vector<int> start_ranks(total_tasks, -1);
    std::vector<double> run_secs(total_tasks, 0.0);

    std::cout << "=== 5x5 Flip Graph RW Stats (C3, MT) ===\n";
    std::cout << "Scheme: " << npy_path << "\n";
    std::cout << "Flip limit: " << flip_lim << ", Plus limit: " << plus_lim
              << ", Threads: " << worker_count << "\n";
    std::cout << "Total runs: " << total_tasks << "\n\n";

    const auto t_all_start = std::chrono::steady_clock::now();

    // Worker lambda: pulls next seed index, runs the protocol, records results
    auto worker = [&]() {
        while (true) {
            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_tasks) break;

            const int seed = seeds_to_run[idx];
            const auto t0 = std::chrono::steady_clock::now();

            // Each task builds its own Scheme from the same base_data
            Scheme scheme(base_data, 3, static_cast<uint32_t>(seed));

            int start_rank = partition_rank + scheme.get_rank();
            int best_rank  = start_rank;
            int current_rank = start_rank;
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
                        if (current_rank < best_rank) best_rank = current_rank;
                        flips_since_improvement = 0;
                    }
                }
            }

            const auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            if (secs <= 0.0) secs = 1e-9;

            best_ranks[idx]  = best_rank;
            start_ranks[idx] = start_rank;
            run_secs[idx]    = secs;
            total_flips.fetch_add(local_flips, std::memory_order_relaxed);

            // Conditional per-run line based on verbosity
            if (verbose) {
                std::ostringstream oss;
                oss.setf(std::ios::fixed);
                oss << "seed=" << seed
                    << ", start_rank=" << start_rank
                    << ", best_rank=" << best_rank
                    << ", time=" << std::setprecision(3) << secs << "s\n";
                std::lock_guard<std::mutex> lk(cout_mtx);
                std::cout << oss.str();
            }
        }
    };

    // Launch pool
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (int w = 0; w < worker_count; ++w) {
        pool.emplace_back(worker);
    }
    for (auto& th : pool) th.join();

    const auto t_all_end = std::chrono::steady_clock::now();
    double all_secs = std::chrono::duration<double>(t_all_end - t_all_start).count();
    if (all_secs <= 0.0) all_secs = 1e-9;

    // Aggregate results
    std::map<int, std::uint64_t> hist_best;
    long double sum_best = 0.0L;
    long double sum_delta = 0.0L;
    std::size_t valid = 0;

    for (size_t i = 0; i < total_tasks; ++i) {
        int br = best_ranks[i];
        int sr = start_ranks[i];
        if (br >= 0 && sr >= 0) {
            hist_best[br] += 1;
            sum_best  += static_cast<long double>(br);
            sum_delta += static_cast<long double>(sr - br);
            ++valid;
        }
    }

    // Print summary
    std::cout << "\n=== Summary of " << valid << " runs ===\n";
    std::cout << "Histogram of best reached ranks:\n";
    for (const auto& [r, cnt] : hist_best) {
        std::cout << "  rank " << std::setw(3) << r << " : " << cnt << "\n";
    }

    long double mean_best  = (valid ? sum_best / static_cast<long double>(valid) : 0.0L);
    long double mean_delta = (valid ? sum_delta / static_cast<long double>(valid) : 0.0L);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Mean best rank : " << static_cast<double>(mean_best)  << "\n";
    std::cout << "Mean delta     : " << static_cast<double>(mean_delta) << " (start - best)\n";

    double mps = static_cast<double>(total_flips.load()) / all_secs / 1e6;
    std::cout << "\nAll tasks completed in " << std::setprecision(3) << all_secs << "s"
              << ", effective rate=" << std::setprecision(1) << mps << " M/s\n";

    return 0;
}
