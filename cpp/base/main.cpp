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
	int r0 = data.size()/3;
	int rank = 0;
	for (int i = 0; i < r0; ++i)
		if (data[i] != 0)
			rank++;
	return rank;
}

int main() {

	int n = 4;
	int N = 1e7;
	auto data = generate_trivial_decomposition(n);
	int r0 = data.size()/3;

	Scheme scheme(data, std::random_device{}());
	std::cout << "initial rank = " << get_rank(scheme.get_data()) << "\n";

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

	std::cout << "final rank = " << get_rank(scheme.get_data()) << "\n";
	std::cout << "# flips: " << 1e3 / time * N / 1e6 << " M/s" << "\n";

	// print_data(n, scheme.get_data());

	std::cout << (verify_scheme(scheme.get_data(), n) ? "Scheme is correct." : "Scheme is incorrect.") << "\n";

	std::cout << "Finished." << "\n";
	return 0;
}