// Get initial data for scheme

#include <bitset>
#include <vector>
#include <iostream>
#include <tuple>
#include <array>
#include <cstdint>
#include <set>
#include <algorithm>

constexpr int N = 5;
std::vector<std::vector<int>> partition = {{1, 2, 3, 4, 5}}; // {{1, 5}, {3, 4}, {5}}
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
    
    std::vector<U64> data = uvw2data(Ur, Vr, Wr);
    // std::vector<U64> data = {544, 9469984, 3145731, 7346369, 10240, 4325376, 22013950, 10240, 22183968, 18382848, 11535360, 8388616, 22739972, 2048, 131204, 17629184, 264, 787200, 4194436, 3072, 3440928, 320, 262176, 98400, 0, 0, 0, 4194304, 29700, 25486651, 0, 0, 0, 4198528, 25979672, 4194308, 28311552, 8388608, 2982754, 262944, 10518560, 8388936, 4199552, 25978648, 4194308, 22709247, 8388608, 2162754, 24576, 2287615, 4194308, 32, 26214425, 28311552, 3072, 7569632, 4194432, 27648, 27984580, 132, 4194436, 27648, 17498820, 8390656, 262400, 29213696, 5124, 25461051, 4194304, 131204, 10240, 264192, 11534337, 8399400, 17858592, 33548287, 10244, 19456, 0, 0, 0, 0, 0, 0, 25165848, 17825792, 33, 33548287, 10240, 22203424, 32, 8661256, 10813440, 320, 32, 3244131, 330, 8389408, 32800, 8388608, 3244131, 28311899, 8389632, 2556358, 22708356, 28114944, 4194572, 18432, 272640, 10813440, 8390752, 10240, 15138816, 12585152, 393604, 22725764, 8390656, 28311882, 8388640, 1048577, 13002751, 4, 10240, 10944974, 22708224, 1024, 28114944, 264, 805632, 1048577, 561, 25165824, 0, 0, 0};

    auto [Uc, Vc, Wc] = data2uvw(data);
    auto GUc = concat(Uc, Vc, Wc);
    auto GVc = concat(Vc, Wc, Uc);
    auto GWc = concat(Wc, Uc, Vc);
    if (tensor_product(GUc, GVc, GWc) == S) std::cout << "Scheme is correct." << "\n";

    print_data(data);

	// Print statistics
    // std::cout << "Initial setup complete:\n";
    // std::cout << "  N = " << N << ", SIZE = " << SIZE << "\n";
    // std::cout << "  Unique orbits: " << Ur.size() << "\n";
    
    return 0;
}
