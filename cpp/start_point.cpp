// Get initial data for scheme

#include <bitset>
#include <vector>
#include <iostream>
#include <tuple>
#include <array>
#include <cstdint>
#include <set>
#include <algorithm>

constexpr int N = 4;
std::vector<std::vector<int>> partition = {{1, 2}, {3, 4}};
constexpr int SIZE = N * N;

using I8 = std::int8_t;
using U64 = std::uint64_t;
using Tensor = std::array<std::array<std::array<I8, SIZE>, SIZE>, SIZE>;
using BitSet = std::bitset<SIZE>;
using BitMatrix = std::vector<BitSet>;
using Partition = std::vector<std::vector<int>>;

// Build matrix multiplication tensor
Tensor build_mm_tensor() {
    Tensor T{};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                T[k*N + i][i*N + j][j*N + k] = 1;
    return T;
}

// Build partition matrix
BitMatrix build_partition_matrix(const Partition& partition) {
    BitMatrix vec(partition.size());
    for (size_t j = 0; j < partition.size(); ++j)
        for (int i : partition[j])
            vec[j].set(N * (i - 1) + (i - 1));
    return vec;
}

// Tensor XOR
Tensor operator^(const Tensor& a, const Tensor& b) {
    Tensor result{};
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            for (int k = 0; k < SIZE; ++k)
                result[i][j][k] = a[i][j][k] ^ b[i][j][k];
    return result;
}

// Decompose tensor into bitmatrices
std::tuple<BitMatrix, BitMatrix, BitMatrix> decompose(const Tensor& T) {
    std::vector<std::tuple<int,int,int>> nonzero;
    
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            for (int k = 0; k < SIZE; ++k)
                if (T[i][j][k])
                    nonzero.emplace_back(i, j, k);
    
    int r = nonzero.size();
    BitMatrix U(r), V(r), W(r);
    
    for (int p = 0; p < r; ++p) {
        auto [i, j, k] = nonzero[p];
        U[p].set(i); V[p].set(j); W[p].set(k);
    }
    
    return {U, V, W};
}

// Tensor product
Tensor tensor_product(const BitMatrix& U, const BitMatrix& V, const BitMatrix& W) {
    Tensor T{};
    int r = U.size();
    
    for (int p = 0; p < r; ++p) {
        for (int i = 0; i < SIZE; ++i) {
            if (!U[p][i]) continue;
            for (int j = 0; j < SIZE; ++j) {
                if (!V[p][j]) continue;
                for (int k = 0; k < SIZE; ++k)
                    if (W[p][k])
                        T[i][j][k] ^= 1;
            }
        }
    }
    return T;
}

// Get unique orbit representatives
std::tuple<BitMatrix, BitMatrix, BitMatrix> get_representatives(
    const BitMatrix& U, const BitMatrix& V, const BitMatrix& W) {
    
    using Orbit = std::array<std::array<U64, 3>, 3>;
    
    std::set<Orbit> seen_orbits;
    BitMatrix U_repr, V_repr, W_repr;
    
    for (size_t i = 0; i < U.size(); ++i) {
        U64 u = U[i].to_ullong();
        U64 v = V[i].to_ullong();
        U64 w = W[i].to_ullong();
        
        Orbit orbit = {{
            {u, v, w},
            {v, w, u},
            {w, u, v}
        }};
        
        std::sort(orbit.begin(), orbit.end());
        
        if (seen_orbits.insert(orbit).second) {
            U_repr.push_back(U[i]);
            V_repr.push_back(V[i]);
            W_repr.push_back(W[i]);
        }
    }
    
    return {U_repr, V_repr, W_repr};
}

// Convert BitMatrix triplet to flat U64 data
std::vector<U64> uvw2data(const BitMatrix& U, const BitMatrix& V, const BitMatrix& W) {
    std::vector<U64> data;
    data.reserve(U.size() * 3);
    
    for (size_t i = 0; i < U.size(); ++i) {
        data.push_back(U[i].to_ullong());
        data.push_back(V[i].to_ullong());
        data.push_back(W[i].to_ullong());
    }
    
    return data;
}

// Restore BitMatrix from flat U64 data
std::tuple<BitMatrix, BitMatrix, BitMatrix> data2uvw(const std::vector<U64>& data) {
    int n_orbits = data.size() / 3;
    BitMatrix U, V, W;
    U.reserve(n_orbits);
    V.reserve(n_orbits);
    W.reserve(n_orbits);
    
    for (int i = 0; i < n_orbits; i++) {
        U.push_back(BitSet(data[i * 3]));
        V.push_back(BitSet(data[i * 3 + 1]));
        W.push_back(BitSet(data[i * 3 + 2]));
    }
    
    return {U, V, W};
}

BitMatrix concat(const BitMatrix& m) {
    return m;
}
template<typename... Matrices>
BitMatrix concat(const BitMatrix& first, const Matrices&... rest) {
    BitMatrix result = first;
    size_t total_size = first.size() + (rest.size() + ...);
    result.reserve(total_size);
    ((result.insert(result.end(), rest.begin(), rest.end())), ...);
    return result;
}

void print_data(const std::vector<U64>& data) {
    std::cout << "{";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << (i < data.size() - 1 ? ", " : "");
    }
    std::cout << "}\n";
}

int main() {
    // Build MM tensor
    auto M = build_mm_tensor();
    
    // Build partition tensor
    BitMatrix partition_matrix = build_partition_matrix(partition);
    auto T = tensor_product(partition_matrix, partition_matrix, partition_matrix);
    
    // XOR and decompose
    Tensor S = M ^ T;
    auto [U, V, W] = decompose(S);

    auto [Ur, Vr, Wr] = get_representatives(U, V, W);
    
    // std::vector<U64> data = uvw2data(Ur, Vr, Wr);
    // std::vector<U64> data = {1, 1, 16, 1, 2, 8, 1, 4, 64, 1, 16, 16, 2, 16, 8, 2, 32, 64, 4, 128, 8, 4, 256, 64, 16, 32, 128, 32, 256, 128};
    // std::vector<U64> data = {60, 5, 22, 5, 6, 36, 1, 4, 462, 45, 133, 16, 134, 16, 9, 2, 32, 320, 4, 150, 25, 4, 256, 65, 0, 36, 400, 32, 256, 130};
    std::vector<U64> data = {17, 17479, 96, 8, 26208, 4353, 8, 61152, 4097, 16460, 2048, 8224, 1024, 33280, 62603, 12336, 17420, 2048, 273, 8, 8736, 3017, 0, 49180, 59, 1088, 2560, 1284, 50177, 0, 128, 32768, 8234, 48000, 1, 42, 34, 45232, 1, 128, 12288, 32, 42, 128, 36864, 16384, 2048, 37008, 15560, 33168, 0, 17, 17484, 2656, 28807, 1024, 256, 62576, 1024, 35584, 2816, 1, 110, 17408, 2560, 145, 35072, 34043, 1024, 11, 64, 2833};
    // std::vector<U64> data = {1, 1, 32, 1, 2, 16, 1, 4, 256, 1, 8, 4096, 1, 32, 32, 2, 32, 16, 2, 64, 256, 2, 128, 4096, 4, 512, 16, 4, 1024, 256, 4, 2048, 4096, 8, 8192, 16, 8, 16384, 256, 8, 32768, 4096, 32, 64, 512, 32, 128, 8192, 64, 1024, 512, 64, 2048, 8192, 128, 16384, 512, 128, 32768, 8192, 1024, 1024, 32768, 1024, 2048, 16384, 1024, 32768, 32768, 2048, 32768, 16384};

    auto [Uc, Vc, Wc] = data2uvw(data);
    auto GUc = concat(Uc, Vc, Wc);
    auto GVc = concat(Vc, Wc, Uc);
    auto GWc = concat(Wc, Uc, Vc);
    if (tensor_product(GUc, GVc, GWc) == S) std::cout << "Scheme is correct." << "\n";

    // print_data(data);

	// Print statistics
    // std::cout << "Initial setup complete:\n";
    // std::cout << "  N = " << N << ", SIZE = " << SIZE << "\n";
    // std::cout << "  Unique orbits: " << Ur.size() << "\n";
    
    return 0;
}
