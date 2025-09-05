#include "scheme.h"
#include "utils.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <chrono>
#include <cstdint>
#include <map>
#include <iomanip>
#include <random>

using U64 = std::uint64_t;
using U32 = std::uint32_t;

// Get rank (count non-zero terms)
int get_rank(const std::vector<U64>& data) {
    int n_term = data.size() / 3;
    int rank = 0;
    for (int i = 0; i < n_term; ++i)
        if (data[3*i] != 0)
            rank++;
    return rank;
}

int main(int argc, char* argv[]) {
    CLI::App app{"Matrix Multiplication Flip Graph Search (multithreaded)"};

    // Core parameters
    int n = 4;
    int flip_lim = 10000000;
    int plus_lim = 10000;
    int plus_count = 3;
    int threads = 1;
    int num_runs = 10;

    app.add_option("-n,--size", n, "Matrix size (nxn)")->default_val(4)->check(CLI::PositiveNumber);
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit")->default_val(10000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(10000);
    app.add_option("-c,--plus-count", plus_count, "Initial plus transitions")->default_val(3);
    app.add_option("-t,--threads", threads, "Number of worker threads")->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("-r,--runs", num_runs, "Number of runs with random seeds")->default_val(10)->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);

    // Generate random seeds
    std::random_device rd;
    std::mt19937 seed_gen(rd());
    std::uniform_int_distribution<U32> seed_dist(0, UINT32_MAX);
    
    std::vector<U32> seeds_to_run;
    seeds_to_run.reserve(num_runs);
    for (int i = 0; i < num_runs; ++i) {
        seeds_to_run.push_back(seed_dist(seed_gen));
    }

    // Generate initial decomposition
    auto initial_data = generate_trivial_decomposition(n);

    std::cout << "=== " << n << "x" << n << " Matrix Multiplication Flip Graph Search ===\n";
    std::cout << "Flip limit: " << flip_lim << ", Plus limit: " << plus_lim;
    std::cout << ", Initial plus: " << plus_count << ", Threads: " << threads << "\n";
    std::cout << "Running " << num_runs << " searches with random seeds\n\n";

    const size_t total_tasks = seeds_to_run.size();
    const int worker_count = std::min<int>(threads, static_cast<int>(total_tasks));

    std::atomic<size_t> next_index{0};
    std::atomic<unsigned long long> total_flips{0};
    std::mutex cout_mtx;

    // Store results for each task
    std::vector<int> best_ranks(total_tasks, -1);
    std::vector<int> initial_ranks(total_tasks, -1);
    std::vector<bool> verified(total_tasks, false);

    auto t_all_start = std::chrono::steady_clock::now();

    // Worker thread function
    auto worker = [&]() {
        while (true) {
            size_t idx = next_index.fetch_add(1);
            if (idx >= total_tasks) break;

            const U32 seed = seeds_to_run[idx];
            auto t0 = std::chrono::steady_clock::now();

            // Create scheme with seed
            Scheme scheme(initial_data, seed);
            
            // Initial plus transitions
            for (int i = 0; i < plus_count; ++i) {
                scheme.plus();
            }

            initial_ranks[idx] = get_rank(scheme.get_data());
            int best_rank = initial_ranks[idx];
            int current_rank = best_rank;
            int flips_since_improvement = 0;
            int plus_transitions = 0;

            unsigned long long local_flips = 0;

            // Main search loop
            for (int i = 0; i < flip_lim; ++i) {
                if (!scheme.flip()) {
                    // No more flips possible
                    std::lock_guard<std::mutex> lk(cout_mtx);
                    std::cout << "seed=" << seed << " - No flips possible at step " << i << "\n";
                    break;
                }
                ++local_flips;

                current_rank = get_rank(scheme.get_data());
                if (current_rank < best_rank) {
                    best_rank = current_rank;
                    flips_since_improvement = 0;
                } else {
                    ++flips_since_improvement;
                }

                // Plus transition if stuck
                if (flips_since_improvement >= plus_lim) {
                    if (scheme.plus()) {
                        ++plus_transitions;
                        current_rank = get_rank(scheme.get_data());
                        flips_since_improvement = 0;
                        
                        // Update best if improved
                        if (current_rank < best_rank) {
                            best_rank = current_rank;
                        }
                    }
                }
            }

            // Verify final result
            verified[idx] = verify_scheme(scheme.get_data(), n);

            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            if (secs <= 0.0) secs = 1e-9;

            best_ranks[idx] = best_rank;
            total_flips.fetch_add(local_flips, std::memory_order_relaxed);

            // Output results
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << "seed=" << std::setw(10) << seed
                << ", initial_rank=" << std::setw(3) << initial_ranks[idx]
                << ", best_rank=" << std::setw(3) << best_rank
                << ", flips=" << std::setw(8) << local_flips
                << ", plus=" << std::setw(3) << plus_transitions
                << ", correct=" << (verified[idx] ? "yes" : "NO!")
                << ", time=" << std::setprecision(2) << secs << "s"
                << ", speed=" << std::setprecision(1) << (local_flips/secs/1e6) << " M/s\n";

            std::lock_guard<std::mutex> lk(cout_mtx);
            std::cout << oss.str();
        }
    };

    // Launch worker threads
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (int w = 0; w < worker_count; ++w) {
        pool.emplace_back(worker);
    }
    for (auto& th : pool) th.join();

    auto t_all_end = std::chrono::steady_clock::now();
    double all_secs = std::chrono::duration<double>(t_all_end - t_all_start).count();
    if (all_secs <= 0.0) all_secs = 1e-9;

    // Build summary
    std::map<int, int> rank_counts;
    int overall_best = INT_MAX;
    U32 best_seed = 0;
    bool all_correct = true;
    
    for (size_t i = 0; i < total_tasks; ++i) {
        if (best_ranks[i] >= 0) {
            rank_counts[best_ranks[i]]++;
            if (best_ranks[i] < overall_best) {
                overall_best = best_ranks[i];
                best_seed = seeds_to_run[i];
            }
        }
        if (!verified[i]) all_correct = false;
    }

    // Print summary
    if (total_tasks > 1) {
        std::cout << "\n=== Summary of " << total_tasks << " runs ===\n";
        std::cout << "Best ranks achieved:\n";
        for (const auto& [r, cnt] : rank_counts) {
            std::cout << "  Rank " << r << ": " << cnt << (cnt == 1 ? " run" : " runs") << "\n";
        }
        std::cout << "Overall best: rank " << overall_best << " (seed " << best_seed << ")\n";
        std::cout << "Verification: " << (all_correct ? "All correct" : "SOME FAILED!") << "\n";
    }

    // Performance stats
    double mps = static_cast<double>(total_flips.load()) / all_secs / 1e6;
    std::cout << "\nTotal time: " << std::fixed << std::setprecision(2) << all_secs << "s"
              << ", Total flips: " << total_flips.load()
              << ", Effective rate: " << std::setprecision(1) << mps << " M/s\n";

    return 0;
}