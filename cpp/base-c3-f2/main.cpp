#include <iostream>
#include <cstdint>
#include <chrono>
#include <random>
#include "scheme.h"
#include "utils.h"

void print_uint(int n, const U64& x) {
    for (int j = 0; j < n * n; ++j) {
        std::cout << ((x >> j) & 1);
    }
}

void print_data(int n, const std::vector<U64>& data) {
    for (int i = 0; i < (int)data.size(); ++i) {
        print_uint(n, data[i]);
        std::cout << ((i + 1) % 3 ? " " : "\n");
    }
}

int get_rank(const std::vector<U64>& data, std::vector<std::vector<int>> parts) {
    int n_term = (int)data.size() / 3;
    int rank = 0;
    for (int i = 0; i < n_term; ++i)
        if (data[3 * i] != 0)
            rank++;
    return 3*rank + parts.size();
}

int main() {
    // Matrix size
    int n = 5;

    // Number of flips
    int N = 1e8;

    // Diagonal partition for C3 symmetry.
    // std::vector<std::vector<int>> parts = {{0, 4}, {1, 3}, {2}};
    std::vector<std::vector<int>> parts = {{0, 1}, {2, 3}, {4}};
    // std::vector<std::vector<int>> parts = {{0, 3}, {1, 2}};

    // Build the trivial symmetric decomposition S for M_n - T(parts).
    auto data_full = generate_trivial_decomposition(n, parts);
    auto data = select_orbit_representatives(data_full);

    std::cout << "initial rank = " << get_rank(data, parts) 
              << ", correct = " << verify_scheme(data, n, parts)
              << "\n";

    // std::cout << "data.size() = " << (data.size() / 3) << "\n";
    
    // auto f1 = verify_scheme(data_full, n, parts);
    auto correct_init = verify_scheme(data, n, parts);

    // print_data(n, data);

    // for (int j = 0; j < 100; ++j) {
        U32 seed = std::random_device{}();
        Scheme scheme(data, 3, seed);

        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        for (int i = 0; i < N; ++i) {
            if (!scheme.flip()) {
                std::cout << "No flips at step " << i << "\n";
                break;
            }
        }
        auto t1 = clock::now();
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // Verify cyclic Brent equations for M - T(parts)
        auto correct = verify_scheme(scheme.get_data(), n, parts);

        std::cout << "final rank = " << get_rank(scheme.get_data(), parts)
                  << ", seed = " << seed
                  << ", correct = " << (correct ? 1 : 0)
                  << ", speed = " << (time_ms ? (1e3 / time_ms * N / 1e6) : 0.0) << " M/s"
                  << "\n";
    // }

    std::cout << "Finished." << "\n";
    return 0;
}
