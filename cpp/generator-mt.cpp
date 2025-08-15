#include "scheme.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <map>
#include <deque>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>

using U16 = std::uint16_t;
using U64  = std::uint64_t;

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph K-step Label Generator (strided, multithreaded)"};

    // Core parameters
    int flip_lim = 100000000;    // total flips per run
    int plus_lim = 50000;        // flips without improvement before plus()
    int verbose  = 1;            // 0 = silent per-run, 1 = per-run info
    int k_steps  = 1000;         // label horizon k (steps)
    int d_steps  = 1000;         // save stride d (steps); must divide k
    int partition_rank = 3;      // fixed offset used in experiments
    int threads = 1;             // number of worker threads

    // Seeding / runs (mutually exclusive: -s vs -n)
    std::vector<int> seed_list;
    int num_runs = 1;
    int seed_start = 100;

    // CLI
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per run")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(100000);
    app.add_option("-k,--steps", k_steps, "Label horizon k (steps)")->default_val(10000000)->check(CLI::PositiveNumber);
    app.add_option("-d,--stride", d_steps, "Save stride d (steps), must divide k")->default_val(10000)->check(CLI::PositiveNumber);
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=silent, 1=per-run info")->default_val(0)->check(CLI::Range(0, 1));
    app.add_option("-t,--threads", threads, "Number of worker threads")->default_val(1)->check(CLI::PositiveNumber);

    auto* seed_opt = app.add_option("-s,--seeds", seed_list, "Explicit list of seeds to run");
    auto* runs_opt = app.add_option("-n,--num-runs", num_runs, "Number of runs (starting from --seed-start)")->default_val(1)->check(CLI::PositiveNumber);
    seed_opt->excludes(runs_opt);
    runs_opt->excludes(seed_opt);
    app.add_option("--seed-start", seed_start, "Starting seed for --num-runs")->default_val(100)->needs(runs_opt);

    CLI11_PARSE(app, argc, argv);

    if (k_steps <= 0 || d_steps <= 0) {
        std::cerr << "k and d must be positive.\n";
        return 1;
    }
    if (k_steps % d_steps != 0) {
        std::cerr << "k must be divisible by d. Given k=" << k_steps << ", d=" << d_steps << "\n";
        return 1;
    }

    // Prepare the list of seeds
    std::vector<int> seeds_to_run;
    if (!seed_list.empty()) {
        seeds_to_run = seed_list;
    } else {
        seeds_to_run.reserve(num_runs);
        for (int i = 0; i < num_runs; ++i) seeds_to_run.push_back(seed_start + i);
    }
    if (seeds_to_run.empty()) {
        std::cout << "No seeds to run.\n";
        return 0;
    }

    // Initial 5x5 C3-symmetric starting data (fixed size; does not change during search)
    const std::vector<U64> init_data = {
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
    const size_t data_size = init_data.size();

    // Global aggregates (filled after workers join)
    std::vector<int>    best_ranks(seeds_to_run.size(), -1);
    std::vector<double> run_secs(seeds_to_run.size(), 0.0);
    std::atomic<std::uint64_t> total_states_saved{0};
    std::atomic<std::uint64_t> total_pairs_labeled{0};
    std::atomic<unsigned long long> total_flips{0};

    // RNG for 6-digit file ids (used to reduce name collisions)
    std::random_device rd;
    std::mt19937_64 id_rng((static_cast<std::uint64_t>(rd()) << 1) ^ 0x9E3779B97F4A7C15ULL);
    std::mutex id_mtx; // protects id_rng

    // Output header
    if (verbose) {
        std::cout << "=== K-step Label Generator (strided, multithreaded) ===\n";
        std::cout << "k=" << k_steps << ", d=" << d_steps
                  << ", flip_lim=" << flip_lim << ", plus_lim=" << plus_lim
                  << ", threads=" << std::max(1, std::min(threads, static_cast<int>(seeds_to_run.size())))
                  << "\n";
        std::cout << "Running " << seeds_to_run.size() << " run(s): ";
        for (size_t i = 0; i < seeds_to_run.size(); ++i) {
            std::cout << seeds_to_run[i] << (i + 1 < seeds_to_run.size() ? ", " : "");
        }
        std::cout << "\n\n";
    }

    // Work distribution
    const size_t total_tasks = seeds_to_run.size();
    const int worker_count = std::max(1, std::min<int>(threads, static_cast<int>(total_tasks)));
    std::atomic<size_t> next_index{0};
    std::mutex cout_mtx;

    auto t_all_start = std::chrono::high_resolution_clock::now();

    auto worker = [&]() {
        while (true) {
            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_tasks) break;

            const int seed = seeds_to_run[idx];

            // Per-task timer
            auto t0 = std::chrono::high_resolution_clock::now();

            // Build scheme for this seed
            Scheme scheme(init_data, 3, static_cast<uint32_t>(seed));

            int current_rank = partition_rank + scheme.get_rank();
            int best_rank = current_rank;
            int flips_since_improvement = 0;

            // Derived counts from limits
            const int D = d_steps;
            const int K = k_steps;
            const int F = flip_lim;
            const int k_blocks = K / D;
            const int stride_total = F / D;                 // how many stride boundaries exist up to F
            const int max_state_rows = std::max(0, stride_total - k_blocks); // do not store the last k steps

            // Flat storage for labeled samples:
            // Each row is [data..., rank_t, rank_t_plus_k] as uint64
            const size_t row_width = data_size + 2;
            std::vector<U64> flat;
            flat.assign(static_cast<size_t>(max_state_rows) * row_width, 0ULL);

            auto row_ptr = [&](int row_index) -> U64* {
                return flat.data() + static_cast<size_t>(row_index) * row_width;
            };

            int states_written = 0;  // rows with state + rank_t written
            int labels_written = 0;  // rows where rank_t_plus_k also written
            unsigned long long local_flips = 0;

            // Steps counter; sampling happens at steps s = D, 2D, 3D, ...
            int s = 0;

            for (int i = 0; i < flip_lim; ++i) {
                // Try a flip, stop if there are no valid flips
                if (!scheme.flip()) break;
                ++s;
                ++local_flips;

                // Update absolute rank
                current_rank = partition_rank + scheme.get_rank();

                // On every stride boundary, process save/label logic
                if (s % D == 0) {
                    const int stride_idx = s / D; // 1-based

                    // Save state at t if not in last k steps window
                    if (stride_idx <= stride_total - k_blocks && states_written < max_state_rows) {
                        U64* row = row_ptr(states_written);
                        const auto& cur = scheme.get_data();

                        // Copy state
                        for (size_t j = 0; j < data_size; ++j) row[j] = cur[j];

                        // rank_t
                        row[data_size] = static_cast<U64>(current_rank);

                        // rank_t_plus_k placeholder remains zero
                        ++states_written;
                    }

                    // Label sample from (t - k)
                    if (stride_idx > k_blocks) {
                        const int labeled_row = stride_idx - k_blocks - 1; // zero-based
                        if (labeled_row >= 0 && labeled_row < states_written) {
                            U64* row = row_ptr(labeled_row);
                            if (row[data_size + 1] == 0ULL) {
                                row[data_size + 1] = static_cast<U64>(current_rank);
                                ++labels_written;
                            }
                        }
                    }
                }

                // Schedule plus() when stuck
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

            auto t1 = std::chrono::high_resolution_clock::now();
            double runtime_sec = std::chrono::duration<double>(t1 - t0).count();
            if (runtime_sec <= 0.0) runtime_sec = 1e-9;

            // Keep only fully labeled pairs
            const int rows_to_save = std::min(states_written, labels_written);

            // Generate random 6-digit id (guarded RNG)
            int id;
            {
                std::lock_guard<std::mutex> lk(id_mtx);
                std::uniform_int_distribution<int> dist(0, 999999);
                id = dist(id_rng);
            }
            std::ostringstream oss;
            // Include seed in filename to avoid collisions between threads
            // oss << "../data/labeled/555/" << seed << "_" << std::setfill('0') << std::setw(6) << id << ".npy";
            oss << "../data/labeled/555/" << seed << ".npy";
            const std::string filename = oss.str();

            // Save as 2D array [rows_to_save, data_size + 2] of uint64
            std::vector<size_t> shape = {static_cast<size_t>(rows_to_save), row_width};
            if (!flat.empty()) {
                cnpy::npy_save(filename, flat.data(), shape);
            } else {
                cnpy::npy_save(filename, flat.data(), shape);
            }

            // Aggregates
            best_ranks[idx] = best_rank;
            run_secs[idx] = runtime_sec;
            total_states_saved.fetch_add(static_cast<std::uint64_t>(states_written), std::memory_order_relaxed);
            total_pairs_labeled.fetch_add(static_cast<std::uint64_t>(rows_to_save), std::memory_order_relaxed);
            total_flips.fetch_add(local_flips, std::memory_order_relaxed);

            // Per-run output
            if (verbose) {
                std::ostringstream line;
                line.setf(std::ios::fixed);
                line << "seed=" << seed
                     << ", best_rank=" << best_rank
                     << ", states_saved=" << states_written
                     << ", labeled_pairs=" << rows_to_save
                     << ", time=" << std::setprecision(3) << runtime_sec << "s\n";
                std::lock_guard<std::mutex> lk(cout_mtx);
                std::cout << line.str();
            }
        }
    };

    // Launch workers
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (int w = 0; w < worker_count; ++w) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    auto t_all_end = std::chrono::high_resolution_clock::now();
    double total_runtime_wall = std::chrono::duration<double>(t_all_end - t_all_start).count();
    if (total_runtime_wall <= 0.0) total_runtime_wall = 1e-9;

    // Build summary maps
    std::map<int, int> best_rank_counts;
    for (int r : best_ranks) if (r >= 0) best_rank_counts[r]++;

    // Final summary
    std::cout << "\n=== Summary of " << seeds_to_run.size() << " run(s) ===\n";
    std::cout << "Best ranks achieved:\n";
    for (const auto& [r, cnt] : best_rank_counts) {
        std::cout << "  Rank " << r << ": " << cnt << (cnt == 1 ? " run" : " runs") << "\n";
    }
    if (!best_rank_counts.empty()) {
        std::cout << "Overall best rank: " << best_rank_counts.begin()->first << "\n";
    }

    double sum_secs = 0.0;
    for (double s : run_secs) sum_secs += s;

    std::cout << "Total states saved: " << total_states_saved.load() << "\n";
    std::cout << "Total labeled pairs: " << total_pairs_labeled.load() << "\n";
    std::cout << "Total flips: " << total_flips.load() << "\n";
    std::cout << "Total wall time: " << std::fixed << std::setprecision(3) << total_runtime_wall << "s\n";
    std::cout << "Avg per run (CPU time): " << std::fixed << std::setprecision(3)
              << (seeds_to_run.empty() ? 0.0 : sum_secs / seeds_to_run.size()) << "s\n";

    // Effective flips throughput based on wall time
    double mps = static_cast<double>(total_flips.load()) / total_runtime_wall / 1e6;
    std::cout << "Effective flips rate: " << std::setprecision(1) << mps << " M/s\n";

    return 0;
}
