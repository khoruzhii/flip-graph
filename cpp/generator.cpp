#include "scheme.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdint>
#include <algorithm>

using U16 = std::uint16_t;
using U64 = std::uint64_t;

int main(int argc, char* argv[]) {
    CLI::App app{"5x5 Flip Graph K-step Label Generator (strided)"};

    // Core parameters
    int flip_lim = 100000000;    // total flips per run
    int plus_lim = 100000;        // flips without improvement before plus()
    int verbose  = 1;            // 0 = silent per-run, 1 = per-run info
    int k_steps  = 1000000;         // label horizon k (steps)
    int d_steps  = 10000;         // save stride d (steps); must divide k
    int partition_rank = 3;      // fixed offset used in experiments

    // Seeding / runs (mutually exclusive: -s vs -n)
    std::vector<int> seed_list;
    int num_runs = 1;
    int seed_start = 100;

    // Rank filtering
    int rank_filter = -1;        // -1 disables filtering; otherwise keep rows with rank_t == rank_filter

    // CLI
    app.add_option("-f,--flip-lim", flip_lim, "Total flip limit per run")->default_val(100000000);
    app.add_option("-p,--plus-lim", plus_lim, "Flips without improvement before plus transition")->default_val(100000);
    app.add_option("-k,--steps", k_steps, "Label horizon k (steps)")->default_val(1000000)->check(CLI::PositiveNumber);
    app.add_option("-d,--stride", d_steps, "Save stride d (steps), must divide k")->default_val(10000)->check(CLI::PositiveNumber);
    app.add_option("-v,--verbose", verbose, "Verbosity: 0=silent, 1=per-run info")->default_val(0)->check(CLI::Range(0, 1));
    app.add_option("-r,--rank", rank_filter, "Keep only states whose initial absolute rank equals this value (-1 disables filtering)")->default_val(-1);

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

    // Prepare the list of seeds
    std::vector<int> seeds_to_run;
    if (!seed_list.empty()) {
        seeds_to_run = seed_list;
    } else {
        for (int i = 0; i < num_runs; ++i) seeds_to_run.push_back(seed_start + i);
    }

    // Initial 5x5 C3-symmetric starting data (fixed size; does not change during search)
    std::vector<U64> init_data = {
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

    // Aggregated stats
    std::map<int, int> best_rank_counts;            // best absolute rank distribution across runs
    std::uint64_t total_states_saved = 0;           // rows with state+rank_t written (before filtering)
    std::uint64_t total_pairs_labeled = 0;          // rows where rank_t_plus_k is also written (before filtering)
    std::uint64_t total_rows_kept = 0;              // rows kept after filtering and added to global buffer
    double total_runtime_sec = 0.0;

    // Global accumulator for all runs; will be saved once at the end
    const size_t row_width = data_size + 2;         // [state..., rank_t, rank_t_plus_k]
    std::vector<U64> all_rows;                      // flat buffer of size (rows_total * row_width)

    if (verbose) {
        std::cout << "=== K-step Label Generator (strided) ===\n";
        std::cout << "k=" << k_steps << ", d=" << d_steps
                  << ", flip_lim=" << flip_lim << ", plus_lim=" << plus_lim << "\n";
        std::cout << "rank filter: " << (rank_filter < 0 ? std::string("disabled") : std::to_string(rank_filter)) << "\n";
        std::cout << "Output: data.npy (overwrite at end)\n";
        std::cout << "Running " << seeds_to_run.size() << " run(s): ";
        for (size_t i = 0; i < seeds_to_run.size(); ++i) {
            std::cout << seeds_to_run[i] << (i + 1 < seeds_to_run.size() ? ", " : "");
        }
        std::cout << "\n\n";
    }

    for (size_t run = 0; run < seeds_to_run.size(); ++run) {
        const int seed = seeds_to_run[run];

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

        // Flat storage for labeled samples of this run:
        // Each row is [data..., rank_t, rank_t_plus_k] as uint64
        std::vector<U64> flat;
        flat.assign(static_cast<size_t>(max_state_rows) * row_width, 0ULL);

        auto row_ptr = [&](int row_index) -> U64* {
            return flat.data() + static_cast<size_t>(row_index) * row_width;
        };

        int states_written = 0;  // how many rows have state + rank_t written
        int labels_written = 0;  // how many rows also have rank_t_plus_k written

        auto t0 = std::chrono::high_resolution_clock::now();

        // Steps counter; sampling happens at steps s = D, 2D, 3D, ...
        int s = 0;

        for (int i = 0; i < flip_lim; ++i) {
            // Try a flip, stop if there are no valid flips
            if (!scheme.flip()) break;
            ++s;

            // Update absolute rank
            current_rank = partition_rank + scheme.get_rank();

            // On every stride boundary, process save/label logic
            if (s % D == 0) {
                const int stride_idx = s / D; // 1-based: 1 for step D, 2 for 2D, ...

                // Save state at t = stride_idx * D if it is not in the last k steps window
                // We save rows for stride_idx in [1, stride_total - k_blocks]
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

                // If we are at time t, we can now label sample from (t - k)
                if (stride_idx > k_blocks) {
                    const int labeled_row = stride_idx - k_blocks - 1; // zero-based row index
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
        const double runtime_sec = std::chrono::duration<double>(t1 - t0).count();
        total_runtime_sec += runtime_sec;

        // Keep only fully labeled pairs in this run
        const int rows_to_save = std::min(states_written, labels_written);

        // Build filtered batch for this run and append to global accumulator
        std::uint64_t kept_this_run = 0;
        if (rows_to_save > 0) {
            for (int i = 0; i < rows_to_save; ++i) {
                U64* row = row_ptr(i);
                const U64 rank_t = row[data_size]; // initial absolute rank for this row
                if (rank_filter < 0 || static_cast<int>(rank_t) == rank_filter) {
                    all_rows.insert(all_rows.end(), row, row + row_width);
                    ++kept_this_run;
                }
            }
        }
        total_rows_kept += kept_this_run;

        if (verbose) {
            std::cout << std::setw(4) << (run + 1)
                      << ": best_rank=" << best_rank
                      << ", rows_labeled=" << rows_to_save
                      << ", rows_kept=" << kept_this_run
                      << ", time=" << std::fixed << std::setprecision(1) << runtime_sec << "s\n";
        }

        // Aggregate stats
        best_rank_counts[best_rank] += 1;
        total_states_saved += static_cast<std::uint64_t>(states_written);
        total_pairs_labeled += static_cast<std::uint64_t>(rows_to_save);
    }

    // Final save: overwrite data.npy with all collected rows
    const std::string out_file = "../data/labeled/555/data.npy";
    const size_t total_rows = (row_width == 0) ? 0 : (all_rows.size() / row_width);
    std::vector<size_t> final_shape = {total_rows, row_width};
    cnpy::npy_save(out_file, all_rows.data(), final_shape);

    // Final summary
    std::cout << "=== Summary of " << seeds_to_run.size() << " run(s) ===\n";
    std::cout << "Best ranks achieved:\n";
    for (const auto& [r, cnt] : best_rank_counts) {
        std::cout << "  Rank " << r << ": " << cnt << (cnt == 1 ? " run" : " runs") << "\n";
    }
    if (!best_rank_counts.empty()) {
        std::cout << "Overall best rank: " << best_rank_counts.begin()->first << "\n";
    }
    std::cout << "Total states saved (pre-filter): " << total_states_saved << "\n";
    std::cout << "Total labeled pairs (pre-filter): " << total_pairs_labeled << "\n";
    std::cout << "Total rows kept (post-filter): " << total_rows_kept << "\n";
    std::cout << "Wrote " << total_rows << " rows to " << out_file << " with width " << row_width << "\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_runtime_sec
              << "s, avg per run: "
              << (seeds_to_run.empty() ? 0.0 : total_runtime_sec / seeds_to_run.size()) << "s\n";

    return 0;
}
