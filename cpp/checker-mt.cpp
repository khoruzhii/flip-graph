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
    CLI::App app{"5x5 Flip Graph Random Walk Stats over multiple schemes (C3 symmetry, multithreaded by schemes)"};

    // Walk protocol parameters
    int flip_lim = 10000000;  // total flip limit per attempt
    int plus_lim = 100000;      // flips without improvement before plus()
    int threads  = 1;          // number of worker threads
    int verbose  = 0;          // 0=summary, 1=per-scheme + summary

    // Seeds: either explicit list or generated [seed_start, seed_start + n)
    std::vector<int> seed_list;
    int num_runs   = 1;
    int seed_start = 100;

    // Input schemes location
    std::string data_dir = "../data/schemes/555";
    std::string npy_name; // required

    // CLI options
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per attempt")->default_val(10000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(100000);
    app.add_option("-t,--threads", threads, "Number of worker threads (parallel by schemes)")
        ->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=summary, 1=per-scheme + summary")
        ->default_val(0)->check(CLI::Range(0, 1));

    auto* seed_opt = app.add_option("-s,--seeds", seed_list, "Explicit list of seeds to run (overrides -n/--num-runs)");
    auto* runs_opt = app.add_option("-n,--num-runs", num_runs, "Number of runs per scheme (generated from seed-start)")
                        ->default_val(1)->check(CLI::PositiveNumber);
    seed_opt->excludes(runs_opt);
    runs_opt->excludes(seed_opt);
    app.add_option("--seed-start", seed_start, "Starting seed for --num-runs")->default_val(100)->needs(runs_opt);

    app.add_option("--data-dir", data_dir, "Directory with 2D scheme npy file (default ../data/schemes/555)");
    app.add_option("--npy", npy_name, "Filename of the 2D .npy with schemes (each row is one scheme)")->required();

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
        std::cout << "No seeds specified.\n";
        return 0;
    }
    const size_t runs_per_scheme = seeds_to_run.size();

    // Load 2D npy of uint64_t
    const std::filesystem::path in_path = std::filesystem::path(data_dir) / npy_name;
    if (!std::filesystem::exists(in_path)) {
        std::cerr << "File not found: " << in_path.string() << "\n";
        return 1;
    }
    cnpy::NpyArray arr = cnpy::npy_load(in_path.string());
    if (arr.word_size != sizeof(U64)) {
        std::cerr << "Unexpected dtype size in " << in_path.string()
                  << " (expected " << sizeof(U64) << " bytes)\n";
        return 1;
    }
    if (arr.shape.size() != 2) {
        std::cerr << "Expected a 2D .npy file, got " << arr.shape.size() << "D\n";
        return 1;
    }
    const size_t num_schemes = arr.shape[0];
    const size_t scheme_len  = arr.shape[1];
    const U64*   arr_ptr     = arr.data<U64>();

    if (num_schemes == 0 || scheme_len == 0) {
        std::cerr << "Empty 2D array in " << in_path.string() << "\n";
        return 1;
    }

    // Output path for labeled ranks (same dir, suffix "-labeled")
    const std::string out_name = in_path.stem().string() + "-labeled.npy";
    const std::filesystem::path out_path = in_path.parent_path() / out_name;

    // Partition rank is a fixed offset used in this experiment
    const int partition_rank = 3;

    // Prepare multithreading by schemes
    const int worker_count = std::max(1, std::min<int>(threads, static_cast<int>(num_schemes)));
    std::atomic<size_t> next_scheme{0};
    std::atomic<unsigned long long> total_flips{0};
    std::mutex cout_mtx;

    // Per-scheme outputs and stats
    // labeled_best_ranks is (num_schemes x runs_per_scheme) row-major
    std::vector<int32_t> labeled_best_ranks(num_schemes * runs_per_scheme, -1);
    std::vector<double>  mean_best_per_scheme(num_schemes, 0.0);
    std::vector<double>  mean_delta_per_scheme(num_schemes, 0.0);

    if (verbose >= 0) {
        std::cout << "=== 5x5 Flip Graph RW Stats over Schemes (C3, MT by schemes) ===\n";
        std::cout << "File: " << in_path.string() << "\n";
        std::cout << "Schemes: " << num_schemes << " (rows), Scheme length: " << scheme_len << "\n";
        std::cout << "Runs per scheme: " << runs_per_scheme
                  << ", Flip limit: " << flip_lim
                  << ", Plus limit: " << plus_lim
                  << ", Threads: " << worker_count << "\n\n";
    }

    const auto t_all_start = std::chrono::steady_clock::now();

    // Worker: takes next scheme row, runs all seeds sequentially for that scheme
    auto worker = [&]() {
        // Local buffer to avoid reallocations
        std::vector<U64> base_data;
        base_data.resize(scheme_len);

        while (true) {
            size_t i_scheme = next_scheme.fetch_add(1, std::memory_order_relaxed);
            if (i_scheme >= num_schemes) break;

            // Copy row i_scheme into base_data (read-only source is arr_ptr)
            const U64* row_ptr = arr_ptr + i_scheme * scheme_len;
            std::copy(row_ptr, row_ptr + scheme_len, base_data.begin());

            long double sum_best  = 0.0L;
            long double sum_delta = 0.0L;

            for (size_t r = 0; r < runs_per_scheme; ++r) {
                const int seed = seeds_to_run[r];

                // Build scheme and run the protocol
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

                total_flips.fetch_add(local_flips, std::memory_order_relaxed);

                // Store per-run best rank into labeled matrix
                labeled_best_ranks[i_scheme * runs_per_scheme + r] = static_cast<int32_t>(best_rank);

                // Accumulate per-scheme stats
                sum_best  += static_cast<long double>(best_rank);
                sum_delta += static_cast<long double>(start_rank - best_rank);
            }

            // Finalize per-scheme means
            double mean_best  = static_cast<double>(sum_best  / static_cast<long double>(runs_per_scheme));
            double mean_delta = static_cast<double>(sum_delta / static_cast<long double>(runs_per_scheme));
            mean_best_per_scheme[i_scheme]  = mean_best;
            mean_delta_per_scheme[i_scheme] = mean_delta;

            if (verbose >= 1) {
                std::lock_guard<std::mutex> lk(cout_mtx);
                std::cout.setf(std::ios::fixed);
                std::cout << "scheme=" << i_scheme
                          << ", mean_best="  << std::setprecision(3) << mean_best
                          << ", mean_delta=" << std::setprecision(3) << mean_delta
                          << "\n";
            }
        }
    };

    // Launch pool
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (int w = 0; w < worker_count; ++w) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    const auto t_all_end = std::chrono::steady_clock::now();
    double all_secs = std::chrono::duration<double>(t_all_end - t_all_start).count();
    if (all_secs <= 0.0) all_secs = 1e-9;

    // Aggregate global stats
    long double sum_mean_best  = 0.0L;
    long double sum_mean_delta = 0.0L;
    for (size_t i = 0; i < num_schemes; ++i) {
        sum_mean_best  += static_cast<long double>(mean_best_per_scheme[i]);
        sum_mean_delta += static_cast<long double>(mean_delta_per_scheme[i]);
    }
    double global_mean_best  = static_cast<double>(sum_mean_best  / static_cast<long double>(num_schemes));
    double global_mean_delta = static_cast<double>(sum_mean_delta / static_cast<long double>(num_schemes));

    // Save labeled best ranks matrix: shape (num_schemes, runs_per_scheme), dtype=int32
    {
        std::vector<size_t> shape = { num_schemes, runs_per_scheme };
        cnpy::npy_save(out_path.string(), labeled_best_ranks.data(), shape, "w");
    }

    // Summary
    if (verbose >= 0) {
        std::cout << "\n=== Summary over " << num_schemes << " schemes ===\n";
        std::cout.setf(std::ios::fixed);
        std::cout << "Global mean best rank : " << std::setprecision(3) << global_mean_best  << "\n";
        std::cout << "Global mean delta     : " << std::setprecision(3) << global_mean_delta << " (start - best)\n";

        double mps = static_cast<double>(total_flips.load()) / all_secs / 1e6;
        std::cout << "\nElapsed: " << std::setprecision(3) << all_secs << "s"
                  << ", effective rate=" << std::setprecision(1) << mps << " M flips/s\n";
        std::cout << "Labeled ranks saved to: " << out_path.string() << "\n";
    }

    return 0;
}
