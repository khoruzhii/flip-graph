#include "scheme.h"
#include "utils.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <chrono>
#include <condition_variable>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iomanip>
#include <optional>

using U64 = std::uint64_t;
using U32 = std::uint32_t;

// Structure to hold a scheme and its metadata
struct SchemeData {
    std::vector<U64> data;
    int rank;
    U32 seed;  // Seed that generated this scheme
    
    SchemeData() : rank(0), seed(0) {}  // Default constructor
    SchemeData(const std::vector<U64>& d, int r, U32 s) 
        : data(d), rank(r), seed(s) {}
};

// Get rank (count non-zero terms)
int get_rank(const std::vector<U64>& data) {
    int n_term = data.size() / 3;
    int rank = 0;
    for (int i = 0; i < n_term; ++i)
        if (data[3*i] != 0)
            rank++;
    return rank;
}

// Simple hash for scheme data (for deduplication)
size_t hash_scheme(const std::vector<U64>& data) {
    size_t h = 0;
    for (U64 v : data) {
        h ^= std::hash<U64>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

// Pool-based search algorithm (Algorithm 2 from paper)
class PoolSearch {
private:
    int n;                    // Matrix size
    int path_limit;           // Length limit for random walks
    int pool_size;            // Target size for each pool
    int target_rank;          // Target rank to reach
    int plus_lim;             // Flips without improvement before plus
    int threads;              // Number of worker threads
    int max_attempts;         // Maximum attempts per rank level
    
    std::mutex pool_mutex;
    std::mutex result_mutex;
    std::condition_variable pool_cv;
    std::atomic<bool> stop_workers{false};
    std::atomic<int> active_workers{0};
    std::atomic<int> attempts_made{0};
    
    std::vector<SchemeData> current_pool;
    std::vector<SchemeData> next_pool;
    std::unordered_set<size_t> seen_hashes;  // For deduplication
    
    std::mt19937 seed_gen;
    
public:
    PoolSearch(int n_, int path_lim, int pool_sz, int target, 
               int plus_l, int thr, int max_att)
        : n(n_), path_limit(path_lim), pool_size(pool_sz), 
          target_rank(target), plus_lim(plus_l), 
          threads(thr), max_attempts(max_att),
          seed_gen(std::random_device{}()) {}
    
    // Try to find a scheme of rank-1 from the given starting scheme
    std::optional<SchemeData> search_from_scheme(const SchemeData& start) {
        // Generate random seed for this search
        U32 search_seed = seed_gen();
        
        // Create scheme with the starting data
        Scheme scheme(start.data, search_seed);
        
        int current_rank = start.rank;
        int best_rank = current_rank;
        int flips_since_improvement = 0;
        
        // Main search loop
        for (int i = 0; i < path_limit; ++i) {
            if (!scheme.flip()) {
                // No more flips possible, try plus transition
                if (!scheme.plus()) {
                    break;
                }
                flips_since_improvement = 0;
            }
            
            int new_rank = get_rank(scheme.get_data());
            
            // Found a scheme with rank-1!
            if (new_rank < current_rank) {
                return SchemeData(scheme.get_data(), new_rank, search_seed);
            }
            
            // Track improvement
            if (new_rank < best_rank) {
                best_rank = new_rank;
                flips_since_improvement = 0;
            } else {
                ++flips_since_improvement;
            }
            
            // Plus transition if stuck
            if (flips_since_improvement >= plus_lim) {
                if (scheme.plus()) {
                    flips_since_improvement = 0;
                    new_rank = get_rank(scheme.get_data());
                    if (new_rank < current_rank) {
                        return SchemeData(scheme.get_data(), new_rank, search_seed);
                    }
                }
            }
        }
        
        return std::nullopt;
    }
    
    // Worker thread function
    void worker() {
        while (!stop_workers) {
            SchemeData* to_process = nullptr;
            
            // Get a scheme from the pool
            {
                std::unique_lock<std::mutex> lock(pool_mutex);
                pool_cv.wait(lock, [this] { 
                    return !current_pool.empty() || stop_workers; 
                });
                
                if (stop_workers) break;
                
                if (!current_pool.empty()) {
                    // Random selection from pool
                    std::uniform_int_distribution<size_t> dist(0, current_pool.size() - 1);
                    size_t idx = dist(seed_gen);
                    to_process = new SchemeData(current_pool[idx]);
                    active_workers++;
                    attempts_made++;  // Increment attempt counter
                }
            }
            
            if (to_process) {
                // Search from this scheme
                auto result = search_from_scheme(*to_process);
                
                if (result.has_value()) {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    
                    // Check for duplicates
                    size_t h = hash_scheme(result->data);
                    if (seen_hashes.find(h) == seen_hashes.end()) {
                        seen_hashes.insert(h);
                        next_pool.push_back(*result);
                    }
                }
                
                delete to_process;
                active_workers--;
            }
        }
    }
    
    // Run pool-based search
    void run() {
        // Generate initial decomposition
        auto initial_data = generate_trivial_decomposition(n);
        int initial_rank = get_rank(initial_data);
        
        std::cout << "=== Pool-based " << n << "x" << n 
                  << " Matrix Multiplication Search ===\n";
        std::cout << "Path limit: " << path_limit 
                  << ", Pool size: " << pool_size
                  << ", Target rank: " << target_rank
                  << ", Threads: " << threads << "\n\n";
        
        // Initialize with starting scheme
        current_pool.push_back(SchemeData(initial_data, initial_rank, 0));
        
        std::cout << "Starting from rank " << initial_rank << "\n\n";
        
        // Search for progressively lower ranks
        for (int current_target = initial_rank - 1; 
             current_target >= target_rank; 
             --current_target) {
            
            std::cout << "=== Searching for rank " << current_target << " ===\n";
            
            next_pool.clear();
            seen_hashes.clear();
            stop_workers = false;
            attempts_made = 0;  // Reset counter for this level
            
            auto start_time = std::chrono::steady_clock::now();
            
            // Start worker threads
            std::vector<std::thread> workers;
            for (int i = 0; i < threads; ++i) {
                workers.emplace_back(&PoolSearch::worker, this);
            }
            
            // Wait until we have enough schemes or hit attempt limit
            while (next_pool.size() < pool_size && attempts_made < max_attempts) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                // Notify workers that there's work
                pool_cv.notify_all();
                
                // // Periodic status update
                // if (attempts_made > 0 && attempts_made % 1000 == 0) {
                //     std::cout << "  Attempts: " << attempts_made 
                //               << ", Found: " << next_pool.size() 
                //               << "/" << pool_size << "\n";
                // }
            }
            
            // Stop workers
            stop_workers = true;
            pool_cv.notify_all();
            for (auto& w : workers) {
                w.join();
            }
            
            auto end_time = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(end_time - start_time).count();
            
            std::cout << "Completed " << attempts_made << " attempts in " 
                      << std::fixed << std::setprecision(1) << secs << "s\n";
            std::cout << "Found " << next_pool.size() << " schemes of rank " 
                      << current_target << "\n";
            
            if (next_pool.empty()) {
                std::cout << "Failed to find schemes of rank " << current_target 
                          << " after " << attempts_made << " attempts\n";
                break;
            }
            
            // Verify schemes
            int verified = 0;
            for (const auto& scheme : next_pool) {
                if (verify_scheme(scheme.data, n)) {
                    verified++;
                }
            }
            std::cout << "Verified: " << verified << "/" << next_pool.size() << "\n\n";
            
            // Move to next level
            current_pool = std::move(next_pool);
            
            // Keep only diverse subset if pool is too large
            if (current_pool.size() > pool_size) {
                std::shuffle(current_pool.begin(), current_pool.end(), seed_gen);
                current_pool.erase(current_pool.begin() + pool_size, current_pool.end());
            }
        }
        
        // Output best result
        if (!current_pool.empty()) {
            std::cout << "\n=== Final Results ===\n";
            std::cout << "Best rank achieved: " << current_pool[0].rank << "\n";
            std::cout << "Pool size at best rank: " << current_pool.size() << "\n";
            
            // // Output one example scheme
            // std::cout << "\nExample scheme (seed " << current_pool[0].seed << "):\n";
            // const auto& data = current_pool[0].data;
            // int n_terms = data.size() / 3;
            // int printed = 0;
            // for (int i = 0; i < n_terms && printed < 5; ++i) {
            //     if (data[i*3] != 0) {
            //         std::cout << "  [" << std::hex << data[i*3] 
            //                   << ", " << data[i*3+1] 
            //                   << ", " << data[i*3+2] << "]\n";
            //         printed++;
            //     }
            // }
            // if (n_terms > 5) {
            //     std::cout << "  ... (" << (n_terms - 5) << " more terms)\n";
            // }
        }
    }
};

int main(int argc, char* argv[]) {
    CLI::App app{"Pool-based Matrix Multiplication Flip Graph Search"};
    
    // Parameters
    int n = 4;
    int path_limit = 1000000;
    int pool_size = 200;
    int target_rank = 0;  // Default target rank is 0
    int plus_lim = 50000;
    int threads = 4;
    int max_attempts = 1000;  // Default max attempts per rank level
    
    app.add_option("-n,--size", n, "Matrix size (nxn)")
        ->default_val(4)->check(CLI::PositiveNumber);
    app.add_option("-l,--path-limit", path_limit, "Path length limit")
        ->default_val(1000000);
    app.add_option("-s,--pool-size", pool_size, "Pool size limit")
        ->default_val(200);
    app.add_option("-r,--target-rank", target_rank, "Target rank")
        ->default_val(0);
    app.add_option("-p,--plus-lim", plus_lim, "Flips before plus transition")
        ->default_val(50000);
    app.add_option("-t,--threads", threads, "Number of worker threads")
        ->default_val(4)->check(CLI::PositiveNumber);
    app.add_option("-m,--max-attempts", max_attempts, "Max attempts per rank level")
        ->default_val(1000)->check(CLI::PositiveNumber);
    
    CLI11_PARSE(app, argc, argv);
    
    PoolSearch search(n, path_limit, pool_size, target_rank, 
                      plus_lim, threads, max_attempts);
    search.run();
    
    return 0;
}