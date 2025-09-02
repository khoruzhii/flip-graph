// main.cpp - Mod3 C3-symmetric test program
#include <iostream>
#include <cstdint>
#include <chrono>
#include <random>
#include "scheme.h"
#include "utils.h"

// Print B3 matrix as trit string
void print_b3(int n, const B3& x) {
    for (int j = 0; j < n * n; ++j) {
        std::cout << get_trit(x, j);
    }
}

// Print data array with B3 elements
void print_data(int n, const std::vector<B3>& data) {
    for (int i = 0; i < (int)data.size(); ++i) {
        print_b3(n, data[i]);
        std::cout << ((i + 1) % 3 ? " " : "\n");
    }
}

// Get rank considering orbits and partition correction
int get_rank(const std::vector<B3>& data, const std::vector<std::vector<int>>& parts) {
    int n_term = (int)data.size() / 3;
    int rank = 0;
    for (int i = 0; i < n_term; ++i) {
        if (!data[3 * i].is_zero()) {
            rank++;
        }
    }
    return 3 * rank + parts.size();
}

// Count non-zero terms (for debugging)
int count_nonzero_terms(const std::vector<B3>& data) {
    int n_term = (int)data.size() / 3;
    int count = 0;
    for (int i = 0; i < n_term; ++i) {
        if (!data[3 * i].is_zero()) {
            count++;
        }
    }
    return count;
}

int main() {
    // Matrix size
    int n = 5;

    // Number of flips
    int N = 1e7;

    // Diagonal partition for C3 symmetry
    std::vector<std::vector<int>> parts = {{0, 4}, {1, 3}, {2}};  // For n=5
    // std::vector<std::vector<int>> parts = {{0, 1}, {2, 3}, {4}};  // For n=5
    // std::vector<std::vector<int>> parts = {{0, 3}, {1, 2}};  // For n=4

    // Build the trivial symmetric decomposition S for M_n - T(parts) in mod3
    auto data_full = generate_trivial_decomposition_mod3(n, parts);
    auto data = select_orbit_representatives_mod3(data_full);

    // print_data(n, data);

    std::cout << "initial rank = " << get_rank(data, parts)
              << ", correct = " << verify_scheme_mod3(data, n, parts)
              << "\n";

    // Optionally print initial data for small cases
    if (n <= 3 && data.size() <= 30) {
        std::cout << "\nInitial data:\n";
        print_data(n, data);
        std::cout << "\n";
    }

    // Run optimization
    U32 seed = std::random_device{}();
    Scheme scheme(data, seed);

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    
    int successful_flips = 0;
    for (int i = 0; i < N; ++i) {
        if (!scheme.flip()) {
            std::cout << "No more flips possible at step " << i << "\n";
            successful_flips = i;
            break;
        }
        successful_flips = i + 1;
    }
    
    auto t1 = clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Get final data
    auto final_data = scheme.get_data();

    // Verify cyclic Brent equations for M - T(parts) in mod3
    bool correct = verify_scheme_mod3(final_data, n, parts);

    std::cout << "final rank = " << get_rank(final_data, parts)
              << ", seed = " << seed
              << ", correct = " << (correct ? 1 : 0)
              << ", speed = " << (time_ms ? (1e3 / time_ms * successful_flips / 1e6) : 0.0) << " M/s"
              << "\n";
    

    // print_data(n, final_data);

    std::cout << "Finished.";
    return 0;
}