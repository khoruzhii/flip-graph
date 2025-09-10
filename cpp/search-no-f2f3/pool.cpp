#include "field.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <chrono>
#include <condition_variable>
#include <algorithm>
#include <iomanip>
#include <optional>
#include <filesystem>
#include <cassert>

#if defined(QUAD)
    #include "scheme_quad.h"
    #if   defined(SYM)
        #include "utils_sym_quad.h"
    #elif defined(ACOM) 
         assert(!"Not implemented acom with quad.");
    #elif defined(AAT) 
        #include "utils_aat_quad.h"
    #else
        #include "utils_gmm_quad.h"
    #endif
#else
    #include "scheme.h"
    #if   defined(SYM)
        #include "utils_sym.h"
    #elif defined(ACOM) 
        #include "utils_acom.h"
    #elif defined(AAT) 
        #include "utils_aat.h"
    #else
        #include "utils_gmm.h"
    #endif
#endif

#ifdef MOD3
    using FieldType = B3;
#else
    using FieldType = B2;
#endif

// Structure to hold a scheme and its metadata
template<typename Field>
struct SchemeData {
    std::vector<Field> data;
    int rank;
    U32 seed;  // Seed that generated this scheme
    
    SchemeData() : rank(0), seed(0) {}
    SchemeData(const std::vector<Field>& d, int r, U32 s) 
        : data(d), rank(r), seed(s) {}
};

// Save pool to file using cnpy
template<typename Field>
void save_pool(const std::vector<SchemeData<Field>>& pool, int rank, int n) {
    if (pool.empty()) return;
    
    // Create pool directory if it doesn't exist
    std::filesystem::create_directories("pool");
    
    // Prepare data for saving
    std::vector<U64> save_data;
    
    for (const auto& scheme : pool) {
        // Filter out zero elements (check only first component)
        std::vector<Field> nonzero_data;
        for (size_t i = 0; i < scheme.data.size(); i += 3) {
            if (!scheme.data[i].is_zero()) {
                nonzero_data.push_back(scheme.data[i]);
                nonzero_data.push_back(scheme.data[i+1]);
                nonzero_data.push_back(scheme.data[i+2]);
            }
        }
        
        // For B3, we need to save both lo and hi parts
        if constexpr (!field_traits<Field>::is_mod2) {
            for (const auto& f : nonzero_data) {
                B3 b3 = static_cast<B3>(f);
                save_data.push_back(b3.lo);
                save_data.push_back(b3.hi);
            }
        } else {
            // For B2, just save the value
            for (const auto& f : nonzero_data) {
                save_data.push_back(pack_field(f));
            }
        }
    }
    
    // Construct filename
    std::string field_name = field_traits<Field>::is_mod2 ? "mod2" : "mod3";
    std::string filename = "pool/" + suffix + "/" + "n" + std::to_string(n) + "_" + field_name + 
                          "_rank" + std::to_string(rank) + ".npy";
    
    // Calculate shape
    size_t elements_per_scheme = rank * 3;  // 3 components per non-zero summand
    if constexpr (!field_traits<Field>::is_mod2) {
        elements_per_scheme *= 2;  // lo and hi for B3
    }
    std::vector<size_t> shape = {pool.size(), elements_per_scheme};
    
    // Save using cnpy
    cnpy::npy_save(filename, save_data.data(), shape);
}

// Pool-based search algorithm
template<typename Field>
class PoolSearch {
private:
    int n;                    // Matrix size
    int path_limit;           // Length limit for random walks
    int pool_size;            // Target size for each pool
    int target_rank;          // Target rank to reach
    int plus_lim;             // Flips without improvement before plus
    int threads;              // Number of worker threads
    int max_attempts;         // Maximum attempts per rank level
    bool use_plus;            // Whether to use plus transitions
    bool save_pools;          // Whether to save pools
    
    std::mutex pool_mutex;
    std::mutex result_mutex;
    std::condition_variable pool_cv;
    std::atomic<bool> stop_workers{false};
    std::atomic<int> active_workers{0};
    std::atomic<int> attempts_made{0};
    
    std::vector<SchemeData<Field>> current_pool;
    std::vector<SchemeData<Field>> next_pool;
    
    std::mt19937 seed_gen;
    
public:
    PoolSearch(int n_, int path_lim, int pool_sz, int target, 
               int plus_l, int thr, int max_att, bool use_p, bool save_p)
        : n(n_), path_limit(path_lim), pool_size(pool_sz), 
          target_rank(target), plus_lim(plus_l), 
          threads(thr), max_attempts(max_att), use_plus(use_p), save_pools(save_p),
          seed_gen(std::random_device{}()) {}
    
    // Try to find a scheme of rank-1 from the given starting scheme
    std::optional<SchemeData<Field>> search_from_scheme(const SchemeData<Field>& start) {
        // Generate random seed for this search
        U32 search_seed = seed_gen();
        
        // Create scheme with the starting data
        Scheme<Field> scheme(start.data, search_seed);
        
        int current_rank = start.rank;
        int best_rank = current_rank;
        int flips_since_improvement = 0;
        
        // Main search loop
        for (int i = 0; i < path_limit; ++i) {
            if (!scheme.flip()) {
                // No more flips possible
                if (use_plus && !scheme.plus()) {
                    break;
                }
                flips_since_improvement = 0;
            }
            
            int new_rank = get_rank(scheme.get_data_field());
            
            // Found a scheme with rank-1!
            if (new_rank < current_rank) {
                return SchemeData<Field>(scheme.get_data_field(), new_rank, search_seed);
            }
            
            // Track improvement
            if (new_rank < best_rank) {
                best_rank = new_rank;
                flips_since_improvement = 0;
            } else {
                ++flips_since_improvement;
            }
            
            // Plus transition if stuck (and enabled)
            if (use_plus && flips_since_improvement >= plus_lim) {
                if (scheme.plus()) {
                    flips_since_improvement = 0;
                }
            }
        }
        
        return std::nullopt;
    }
    
    // Worker thread function
    void worker() {
        while (!stop_workers) {
            SchemeData<Field>* to_process = nullptr;
            
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
                    to_process = new SchemeData<Field>(current_pool[idx]);
                    active_workers++;
                    attempts_made++;
                }
            }
            
            if (to_process) {
                // Search from this scheme
                auto result = search_from_scheme(*to_process);
                
                if (result.has_value()) {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    next_pool.push_back(*result);
                }
                
                delete to_process;
                active_workers--;
            }
        }
    }
    
    // Run pool-based search
    void run() {
        // Generate initial decomposition
        auto initial_data_u64 = generate_trivial_decomposition(n);
        std::vector<Field> initial_data;
        initial_data.reserve(initial_data_u64.size());
        for (U64 val : initial_data_u64) initial_data.push_back(field_traits<Field>::from_u64(val));

        // std::vector<U64> data_raw = {460238,47137,33792,0,262280,131332,1056,32769,10,320,32769,65538,135296,4,2050,0,10240,320,135168,270732,38912,131072,113760,393216,236807,270568,396,33,393216,0,364683,142692,328000,0,2306,8264,270592,8,64,65536,32896,131104,269575,237800,1025,8200,327680,0,67968,33804,473475,33900,66978,440397,1057,32768,14,14336,0,163845,269544,237831,320,327680,131072,32768,141450,32800,39976,467399,327685,179360,135172,128,12289,1036,160,5120,132,135168,132096,36864,3072,96,32805,136320,2,1024,463214,44161,142475,364900,8193,1032,5125,160,67683,33792,32,134,503808,3084,34254,473121,32800,131200,65794,262216,32906,141344,375137,132238,199752,307458,439362,68013,32896,131104,1024,4096,131076,4224,337920,0,8320,4352,267525,65600,40391,466984,501760,5120,270344,256,393600,32800,65792,262208,439373,68002,385,44,2048,1024,135171,270432,47203,460172,46157,461218,79277,428098,473088,0,4096,8192,67650,0,198,0,0,12300,66956,34816,507329,46,266472,240903,473537,33838,64,2,32833,65570,270600,0,196608,6144,98304,96,2050,65600,257,40,1025,2050,65797,267328,132449,374926,107520,399360,372068,135307,0,66,12300,384,266624,139276,335872,1024,201856,305408,266339,139264,491982,15393,138503,368872,439725,67650,6150,0,10240,0,2114,65536,459200,0,8200,2050,272458,65792,459136,32,242957,264418,65602,2048,32768,32,1025,4100,66656,34819,321,8,333825,173058,473550,33825,6,6144,163845,160,8200,262400,33792,270336,131328,262272,32801,1024,1025,0,0,327690,136229,32896,14337,12,434413,72962,364907,142468,32768,1024,3075,96,269543,237832,14336,458752,98304,0,33792,33,33,132,0,393228,272640,65610,32832,0,374926,132449,1,0,16,0,1048576,0,2,0,512,0,1048576,0,4,0,16384,0,1048576,0,8,0,524288,0,1048576,0,16,0,1048576,0,1,0,16,0,2097152,0,32,0,16,0,4194304,0,1024,0,16,0,8388608,0,32768,0,16,0,16777216,0,1048576,0,32,0,16,0,2097152,0,64,0,512,0,2097152,0,128,0,16384,0,2097152,0,256,0,524288,0,2097152,0,512,0,1048576,0,2,0,512,0,2097152,0,64,0,512,0,4194304,0,2048,0,512,0,8388608,0,65536,0,512,0,16777216,0,2097152,0,1024,0,16,0,4194304,0,2048,0,512,0,4194304,0,4096,0,16384,0,4194304,0,8192,0,524288,0,4194304,0,16384,0,1048576,0,4,0,16384,0,2097152,0,128,0,16384,0,4194304,0,4096,0,16384,0,8388608,0,131072,0,16384,0,16777216,0,4194304,0,32768,0,16,0,8388608,0,65536,0,512,0,8388608,0,131072,0,16384,0,8388608,0,262144,0,524288,0,8388608,0,524288,0,1048576,0,8,0,524288,0,2097152,0,256,0,524288,0,4194304,0,8192,0,524288,0,8388608,0,262144,0,524288,0,16777216,0,8388608,0,1048576,0,1,0,16,0,1048576,0,2,0,512,0,1048576,0,4,0,16384,0,1048576,0,8,0,524288,0,1048576,0,16,0,16777216,0,2097152,0,32,0,16,0,2097152,0,64,0,512,0,2097152,0,128,0,16384,0,2097152,0,256,0,524288,0,2097152,0,512,0,16777216,0,4194304,0,1024,0,16,0,4194304,0,2048,0,512,0,4194304,0,4096,0,16384,0,4194304,0,8192,0,524288,0,4194304,0,16384,0,16777216,0,8388608,0,32768,0,16,0,8388608,0,65536,0,512,0,8388608,0,131072,0,16384,0,8388608,0,262144,0,524288,0,8388608,0,524288,0,16777216,0,16777216,0,1048576,0,16,0,16777216,0,2097152,0,512,0,16777216,0,4194304,0,16384,0,16777216,0,8388608,0,524288,0,16777216,0,16777216,0,16777216,0};
        // std::vector<B3> initial_data = {};
        // for (int i = 0; i < data_raw.size(); i+=2) initial_data.push_back(B3{data_raw[i], data_raw[i+1]});

        
        int initial_rank = get_rank(initial_data);
        
        std::cout << "=== Pool-based " << n << "x" << n 
                  << " Matrix Multiplication Search ===\n";
        std::cout << "Field: " << (field_traits<Field>::is_mod2 ? "mod2" : "mod3") << "\n";
        std::cout << "Path limit: " << path_limit 
                  << ", Pool size: " << pool_size
                  << ", Target rank: " << target_rank
                  << ", Threads: " << threads;
        if (use_plus) {
            std::cout << ", Plus transitions: enabled (limit: " << plus_lim << ")";
        } else {
            std::cout << ", Plus transitions: disabled";
        }
        std::cout << "\n\n";
        
        // Initialize with starting scheme
        current_pool.push_back(SchemeData<Field>(initial_data, initial_rank, 0));
        
        // Save initial pool if enabled
        if (save_pools) {
            save_pool(current_pool, initial_rank, n);
        }
        
        std::cout << "Starting from rank " << initial_rank << "\n\n";
        
        // Search for progressively lower ranks
        for (int current_target = initial_rank - 1; 
             current_target >= target_rank; 
             --current_target) {
            
            std::cout << "=== Searching for rank " << current_target << " ===\n";
            
            next_pool.clear();
            stop_workers = false;
            attempts_made = 0;
            
            auto start_time = std::chrono::steady_clock::now();
            
            // Start worker threads
            std::vector<std::thread> workers;
            for (int i = 0; i < threads; ++i) {
                workers.emplace_back(&PoolSearch::worker, this);
            }
            
            // Wait until we have enough schemes or hit attempt limit
            while (next_pool.size() < pool_size && attempts_made < max_attempts) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                pool_cv.notify_all();
                
                // Periodic status update
                if (attempts_made > 0 && attempts_made % 100 == 0) {
                    std::cout << "  Attempts: " << attempts_made 
                              << ", Found: " << next_pool.size() 
                              << "/" << pool_size << "\r" << std::flush;
                }
            }
            std::cout << "\n";
            
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
            
            // Save pool if enabled
            if (save_pools && !next_pool.empty()) {
                save_pool(next_pool, current_target, n);
            }
            
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
        }
    }
};

int main(int argc, char* argv[]) {
    CLI::App app{"Pool-based Matrix Multiplication Flip Graph Search"};
    
    // Parameters
    int n = 4;
    int path_limit = 1000000;
    int pool_size = 200;
    int target_rank = 0;
    int plus_lim = 50000;
    int threads = 4;
    int max_attempts = 1000;
    bool use_plus = false;  // Default: plus transitions disabled
    bool save_pools = false; // Default: saving disabled
    
    app.add_option("-n,--size", n, "Matrix size (nxn)")
        ->default_val(4)->check(CLI::PositiveNumber);
    app.add_option("-f,--path-limit", path_limit, "Path length limit")
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
    app.add_flag("--plus", use_plus, "Enable plus transitions");
    app.add_flag("--save", save_pools, "Save pools to files");
    
    CLI11_PARSE(app, argc, argv);
    
    PoolSearch<FieldType> search(n, path_limit, pool_size, target_rank, 
                                  plus_lim, threads, max_attempts, use_plus, save_pools);
    search.run();
    
    return 0;
}