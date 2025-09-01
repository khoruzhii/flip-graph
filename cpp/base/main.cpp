#include <iostream>
#include <cstdint>
#include <chrono>
#include <random>
#include "scheme.h"
#include "utils.h"

void print_uint(int n, const U64& x) {
	for (int j = 0; j < n*n; ++j) {
		std::cout << ((x >> j) & 1);
	}
}

void print_data(int n, const std::vector<U64>& data) {
	for (int i = 0; i < data.size(); ++i) {
		print_uint(n, data[i]);
		std::cout << ((i+1) % 3 ? " " : "\n");
	}
}

int get_rank(const std::vector<U64>& data) {
	int n_term = data.size()/3;
	int rank = 0;
	for (int i = 0; i < n_term; ++i)
		if (data[3*i] != 0)
			rank++;
	return rank;
}

int main() {

	int n = 5;
	int N = 1e8;
	auto data = generate_trivial_decomposition(n);

	for (int j = 0; j < 100; ++j) {
	
	U32 seed = std::random_device{}();
	Scheme scheme(data, seed);
	// std::cout << "initial rank = " << get_rank(scheme.get_data()) << "\n";

	using clock = std::chrono::steady_clock;
	auto t0 = clock::now();
	for (int i = 0; i < N; ++i) {
		if (!scheme.flip()) {
			std::cout << "No flips at step " << i << "\n";
			break;
		}
	}
	auto t1 = clock::now();
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	// print_data(n, scheme.get_data());

	auto correct = verify_scheme(scheme.get_data(), n);
	std::cout << "final rank = " << get_rank(scheme.get_data()) 
	          << ", seed = " << seed 
	          << ", correct = " << correct
	          << ", speed = " << 1e3 / time * N / 1e6 << " M/s" 
	          << "\n";

	}

	std::cout << "Finished." << "\n";
	return 0;
}