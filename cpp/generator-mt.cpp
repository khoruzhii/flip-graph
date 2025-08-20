#include "scheme.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <filesystem>

using U64 = std::uint64_t;

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph K-step Label Generator (strided, multithreaded)"};

    // Core parameters
    int flip_lim = 100000000;    // total flips per run
    int plus_lim = 100000;       // flips without improvement before plus()
    int verbose  = 0;            // 0 = silent per-run, 1 = per-run info
    int k_steps  = 10000000;     // label horizon k (steps)
    int d_steps  = 10000;        // save stride d (steps); must divide k
    int partition_rank = 3;      // fixed offset used in experiments

    // Seeding / runs (mutually exclusive: -s vs -n)
    std::vector<int> seed_list;
    int num_runs = 1;
    int seed_start = 100;

    // Rank filtering and final quota
    int rank_filter = -1;        // -1 disables filtering; otherwise keep rows with rank_t == rank_filter
    int max_rows    = -1;        // -1 disables quota; otherwise stop when this many filtered rows collected

    // Threading
    int threads = 1;             // number of worker threads

    // CLI
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per run")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(100000);
    app.add_option("-k,--steps", k_steps, "Label horizon k (steps)")->default_val(10000000)->check(CLI::PositiveNumber);
    app.add_option("-d,--stride", d_steps, "Save stride d (steps), must divide k")->default_val(10000)->check(CLI::PositiveNumber);
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=silent, 1=per-run info")->default_val(0)->check(CLI::Range(0, 1));
    app.add_option("-r,--rank", rank_filter, "Keep only states whose initial absolute rank equals this value (-1 disables filtering)")->default_val(-1);
    app.add_option("-m,--max", max_rows, "Stop after collecting this many filtered labeled rows (-1 disables)")->default_val(-1);
    app.add_option("-t,--threads", threads, "Number of worker threads")->default_val(1)->check(CLI::PositiveNumber);

    auto* seed_opt = app.add_option("-s,--seeds", seed_list, "Explicit list of seeds to run");
    auto* runs_opt = app.add_option("-n,--num-runs", num_runs, "Number of runs (starting from --seed-start)")->default_val(1);
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
    if (threads <= 0) {
        std::cerr << "threads must be positive.\n";
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
    const size_t row_width = data_size + 2; // [state..., rank_t, rank_t_plus_k]

    // Output file (always overwritten at the end)
    const std::string out_file = "../data/labeled/555/data.npy";

    // Ensure output directory exists
    {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(out_file).parent_path(), ec);
        if (ec) {
            std::cerr << "Failed to create output directory: " << ec.message() << "\n";
            return 1;
        }
    }

    // Aggregated stats
    std::atomic<std::uint64_t> total_states_saved{0};   // pre-filter
    std::atomic<std::uint64_t> total_pairs_labeled{0};  // pre-filter

    std::mutex cout_mtx;
    std::mutex best_mtx;
    std::map<int,int> best_rank_counts;                 // per-run best rank histogram

    // Global accumulator for all runs; merged at the end from thread-local buffers
    std::mutex accum_mtx;
    std::vector<U64> all_rows; // flat buffer of size (rows_total * row_width)
    if (max_rows > 0) {
        all_rows.reserve(static_cast<size_t>(max_rows) * row_width);
    }

    // Global stop/goal bookkeeping
    std::atomic<std::uint64_t> kept_total{0};           // rows appended to all_rows
    std::atomic<bool> stop{false};                      // signal to stop workers when quota reached

    if (verbose) {
        std::cout << "=== K-step Label Generator (strided, multithreaded) ===\n";
        std::cout << "k=" << k_steps << ", d=" << d_steps
                  << ", flip_lim=" << flip_lim << ", plus_lim=" << plus_lim << "\n";
        std::cout << "rank filter: " << (rank_filter < 0 ? std::string("disabled") : std::to_string(rank_filter)) << "\n";
        std::cout << "threads: " << threads << ", quota m=" << max_rows << "\n";
        std::cout << "Output: " << out_file << " (overwrite at end)\n\n";
    }

    const size_t total_tasks = seeds_to_run.size();
    const int worker_count = std::max(1, std::min<int>(threads, static_cast<int>(total_tasks)));
    std::atomic<size_t> next_index{0};

    auto t_all_start = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            if (stop.load(std::memory_order_relaxed)) break;

            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_tasks) break;

            const int seed = seeds_to_run[idx];

            auto t0 = std::chrono::steady_clock::now();

            // Each task builds its own Scheme
            Scheme scheme(init_data, 3, static_cast<uint32_t>(seed));

            int current_rank = partition_rank + scheme.get_rank();
            int best_rank = current_rank;
            int flips_since_improvement = 0;

            // Derived counts from limits
            const int D = d_steps;
            const int K = k_steps;
            const int F = flip_lim;
            const int k_blocks = K / D;
            const int stride_total = F / D;                 // number of stride boundaries up to F
            const int max_state_rows = std::max(0, stride_total - k_blocks); // do not store the last k steps

            // Per-run storage for labeled samples:
            std::vector<U64> flat(static_cast<size_t>(max_state_rows) * row_width, 0ULL);

            auto row_ptr = [&](int row_index) -> U64* {
                return flat.data() + static_cast<size_t>(row_index) * row_width;
            };

            int states_written = 0;  // rows with state + rank_t written
            int labels_written = 0;  // rows where rank_t_plus_k written

            // Steps counter; sampling happens at steps s = D, 2D, 3D, ...
            int s = 0;

            for (int i = 0; i < flip_lim; ++i) {
                if (max_rows > 0 && stop.load(std::memory_order_relaxed)) break;
                if (!scheme.flip()) break;
                ++s;

                current_rank = partition_rank + scheme.get_rank();

                // On every stride boundary, process save/label logic
                if (s % D == 0) {
                    const int stride_idx = s / D; // 1-based

                    // Save state if not in the last k steps window
                    if (stride_idx <= stride_total - k_blocks && states_written < max_state_rows) {
                        U64* row = row_ptr(states_written);
                        const auto& cur = scheme.get_data();

                        // Copy state
                        for (size_t j = 0; j < data_size; ++j) row[j] = cur[j];

                        // rank_t
                        row[data_size] = static_cast<U64>(current_rank);

                        // rank_t_plus_k placeholder already zero
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

                // plus() policy when stuck
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

            auto t1 = std::chrono::steady_clock::now();
            double runtime_sec = std::chrono::duration<double>(t1 - t0).count();

            // Fully labeled pairs
            const int rows_to_save = std::min(states_written, labels_written);

            // Build filtered batch for this run into a local vector
            std::vector<U64> local_rows;
            std::uint64_t kept_this_run = 0;
            if (rows_to_save > 0) {
                local_rows.reserve(static_cast<size_t>(rows_to_save) * row_width);
                for (int i = 0; i < rows_to_save; ++i) {
                    U64* row = row_ptr(i);
                    const U64 rank_t = row[data_size];
                    if (rank_filter < 0 || static_cast<int>(rank_t) == rank_filter) {
                        local_rows.insert(local_rows.end(), row, row + row_width);
                        ++kept_this_run;
                    }
                }
            }

            // Update pre-filter stats
            total_states_saved.fetch_add(static_cast<std::uint64_t>(states_written), std::memory_order_relaxed);
            total_pairs_labeled.fetch_add(static_cast<std::uint64_t>(rows_to_save), std::memory_order_relaxed);

            // Merge into the global accumulator with quota enforcement
            std::uint64_t appended_this_run = 0;
            if (!local_rows.empty()) {
                std::lock_guard<std::mutex> lk(accum_mtx);
                if (max_rows > 0) {
                    std::uint64_t have = kept_total.load(std::memory_order_relaxed);
                    if (have < static_cast<std::uint64_t>(max_rows)) {
                        std::uint64_t remaining = static_cast<std::uint64_t>(max_rows) - have;
                        std::uint64_t rows_local = static_cast<std::uint64_t>(local_rows.size() / row_width);
                        std::uint64_t to_take = std::min(remaining, rows_local);
                        if (to_take > 0) {
                            all_rows.insert(all_rows.end(),
                                            local_rows.begin(),
                                            local_rows.begin() + static_cast<std::ptrdiff_t>(to_take * row_width));
                            kept_total.store(have + to_take, std::memory_order_relaxed);
                            appended_this_run = to_take;
                            if (have + to_take >= static_cast<std::uint64_t>(max_rows)) {
                                stop.store(true, std::memory_order_relaxed);
                            }
                        }
                    }
                } else {
                    // No quota: take everything
                    all_rows.insert(all_rows.end(),
                                    std::make_move_iterator(local_rows.begin()),
                                    std::make_move_iterator(local_rows.end()));
                    appended_this_run = static_cast<std::uint64_t>(local_rows.size() / row_width);
                    kept_total.fetch_add(appended_this_run, std::memory_order_relaxed);
                }
            }

            // Update best rank histogram
            {
                std::lock_guard<std::mutex> lk(best_mtx);
                best_rank_counts[best_rank] += 1;
            }

            if (verbose) {
                std::lock_guard<std::mutex> lk(cout_mtx);
                std::cout << "seed=" << seed
                          << ": best_rank=" << best_rank
                          << ", rows_labeled=" << rows_to_save
                          << ", rows_kept_local=" << kept_this_run
                          << ", rows_appended=" << appended_this_run
                          << ", time=" << std::fixed << std::setprecision(3) << runtime_sec << "s\n";
            }
        }
    };

    // Launch workers
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (int w = 0; w < worker_count; ++w) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    auto t_all_end = std::chrono::steady_clock::now();
    double total_runtime_sec = std::chrono::duration<double>(t_all_end - t_all_start).count();
    if (total_runtime_sec <= 0.0) total_runtime_sec = 1e-9;

    // Determine number of rows to save, enforce the -m quota exactly
    std::uint64_t total_rows = (row_width == 0) ? 0 : (all_rows.size() / row_width);
    std::uint64_t to_save = total_rows;
    if (max_rows > 0 && to_save > static_cast<std::uint64_t>(max_rows)) {
        to_save = static_cast<std::uint64_t>(max_rows);
        all_rows.resize(static_cast<size_t>(to_save * row_width));
    }

    // Save final array
    std::vector<size_t> final_shape = { static_cast<size_t>(to_save), row_width };
    cnpy::npy_save(out_file, all_rows.data(), final_shape);

    // Final summary
    std::cout << "=== Summary ===\n";
    std::cout << "Runs attempted: " << seeds_to_run.size() << ", threads: " << worker_count << "\n";
    std::cout << "Rank filter: " << (rank_filter < 0 ? std::string("disabled") : std::to_string(rank_filter))
              << ", quota m=" << max_rows << "\n";
    std::cout << "Best ranks achieved:\n";
    for (const auto& [r, cnt] : best_rank_counts) {
        std::cout << "  Rank " << r << ": " << cnt << (cnt == 1 ? " run" : " runs") << "\n";
    }
    if (!best_rank_counts.empty()) {
        std::cout << "Overall best rank: " << best_rank_counts.begin()->first << "\n";
    }
    std::cout << "Total states saved (pre-filter): " << total_states_saved.load() << "\n";
    std::cout << "Total labeled pairs (pre-filter): " << total_pairs_labeled.load() << "\n";
    std::cout << "Total rows appended (post-filter): " << kept_total.load() << "\n";
    std::cout << "Wrote " << to_save << " rows to " << out_file
              << " with width " << row_width << "\n";
    std::cout << "Wall time: " << std::fixed << std::setprecision(3) << total_runtime_sec << "s\n";

    return 0;
}
