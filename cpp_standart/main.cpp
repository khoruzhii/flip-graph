#include <iostream>
#include <vector>
#include <array>
#include <bitset>
#include <cstdint>
#include <random>
#include <unordered_map>
#include "scheme.h"

using U64 = std::uint64_t;

// Generate trivial decomposition for n×n matrix multiplication
std::vector<U64> generate_trivial_decomposition(int n) {
    std::vector<U64> data;
    data.reserve(n * n * n * 3);
    
    // For each (i,j,k) triple, add rank-one tensor:
    // u[i,j] ⊗ v[j,k] ⊗ w[k,i]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                U64 u = 1ULL << (i * n + j); // u[i,j]
                U64 v = 1ULL << (j * n + k); // v[j,k]
                U64 w = 1ULL << (k * n + i); // w[k,i]
                
                data.push_back(u);
                data.push_back(v);
                data.push_back(w);
            }
        }
    }
    
    return data;
}


// Check if the scheme correctly implements matrix multiplication
bool check_scheme(const std::vector<U64>& data, int n) {
    // Number of rank-one tensors
    int r = data.size() / 3;
    
    // Check each Brent equation:    
    for (int i1 = 0; i1 < n; ++i1) {
        for (int i2 = 0; i2 < n; ++i2) {
            for (int j1 = 0; j1 < n; ++j1) {
                for (int j2 = 0; j2 < n; ++j2) {
                    for (int k1 = 0; k1 < n; ++k1) {
                        for (int k2 = 0; k2 < n; ++k2) {
                            // Compute sum over all rank-one tensors
                            int sum = 0;
                            
                            for (int l = 0; l < r; ++l) {
                                U64 u = data[3 * l];
                                U64 v = data[3 * l + 1]; 
                                U64 w = data[3 * l + 2]; // transposed
                                
                                int a = (u >> (i1 * n + i2)) & 1;
                                int b = (v >> (j1 * n + j2)) & 1;
                                int с = (w >> (k1 * n + k2)) & 1;
                                
                                sum ^= (a & b & с);
                            }
                            
                            int expected = (i2 == j1 && j2 == k1 && k2 == i1) ? 1 : 0;
                            if (sum != expected) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return true;
}


inline int pick_type(std::mt19937& rng, const std::array<std::vector<U64>, 3>& vecs) {
  const int x = rng() % (vecs[0].size() + vecs[1].size() + vecs[2].size());
    if (x < vecs[1].size())                         return 0;
    else if (x < vecs[0].size() + vecs[1].size())   return 1;
    else                                            return 2;
}

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

void print_unordered_map(int n, const std::unordered_map<U64, std::vector<int>>& m) {
    std::cout << "{\n";
    bool first_pair = true;
    for (const auto& [k, v] : m) {
        if (!first_pair) std::cout << ",\n";
        first_pair = false;

        std::cout << "    ";
        print_uint(n, k);
        std::cout << ": ";
        std::cout << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << v[i];
        }
        std::cout << "]";
    }
    std::cout << "\n}" << '\n';
}

void print_vec(int n, const std::vector<U64>& v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) std::cout << ",";
        print_uint(n, v[i]);
    }
    std::cout << "]\n";
}

void full_print(Scheme s, int n) {
    print_data(n, s.get_data());

    std::cout << '\n';
    std::cout << "MAPS\n";
    print_unordered_map(n, s.get_map(0));
    std::cout << '\n';
    print_unordered_map(n, s.get_map(1));
    std::cout << '\n';
    print_unordered_map(n, s.get_map(2));
    std::cout << '\n';

    bool ok = check_scheme(s.get_data(), n);
    std::cout << "Scheme is " << (ok ? "correct" : "incorrect") << "\n";
}

// Функция для создания специальных тестовых сценариев
void test_reduction_scenarios(int n) {
    // Тест 1: Идентичные тензоры
    {
        std::vector<U64> test_data;
        // Добавляем два идентичных тензора
        test_data.push_back(1ULL);  // u
        test_data.push_back(2ULL);  // v  
        test_data.push_back(4ULL);  // w
        test_data.push_back(1ULL);  // u (идентичный)
        test_data.push_back(2ULL);  // v (идентичный)
        test_data.push_back(4ULL);  // w (идентичный)
        
        Scheme s_identical(test_data, 111, n);
        std::cout << "Before identical reduction: " << s_identical.get_data().size() / 3 << " tensors\n";
        
        bool reduced = s_identical.reduction();
        std::cout << "Identical reduction: " << (reduced ? "SUCCESS" : "NO REDUCTION") << "\n";
        std::cout << "After identical reduction: " << s_identical.get_data().size() / 3 << " tensors\n";
        
        bool correct = check_scheme(s_identical.get_data(), n);
        std::cout << "Correctness: " << (correct ? "✓" : "✗") << "\n\n";
    }
    
    // Тест 2: Частичное совпадение (u и v одинаковы)
    {
        std::vector<U64> test_data;
        test_data.push_back(1ULL);  // u
        test_data.push_back(2ULL);  // v
        test_data.push_back(4ULL);  // w
        test_data.push_back(1ULL);  // u (совпадает)
        test_data.push_back(2ULL);  // v (совпадает)
        test_data.push_back(8ULL);  // w (отличается)
        
        Scheme s_partial(test_data, 222, n);
        std::cout << "Before partial reduction: " << s_partial.get_data().size() / 3 << " tensors\n";
        
        bool reduced = s_partial.reduction();
        std::cout << "Partial reduction (u,v same): " << (reduced ? "SUCCESS" : "NO REDUCTION") << "\n";
        std::cout << "After partial reduction: " << s_partial.get_data().size() / 3 << " tensors\n";
        
        bool correct = check_scheme(s_partial.get_data(), n);
        std::cout << "Correctness: " << (correct ? "✓" : "✗") << "\n\n";
    }
}

int main() {
    int n = 5;
    auto data = generate_trivial_decomposition(n);
    
    std::cout << "Matrix size: " << n << "x" << n << "\n";
    std::cout << "Initial tensors: " << data.size() / 3 << "\n";
    
    // 1️⃣ ТЕСТ FLIP ПРОИЗВОДИТЕЛЬНОСТИ
    {
        std::cout << "\n=== FLIP PERFORMANCE TEST ===\n";
        Scheme s_flip(data, 42, n);
        
        const long long FLIP_LIMIT = 1'000'000;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        long long successful_flips = 0;
        for (long long i = 0; i < FLIP_LIMIT; ++i) {
            if (s_flip.flip()) {
                successful_flips++;
            } else {
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "Flip rate: " << static_cast<int>(successful_flips / elapsed) << " flips/sec\n";
        std::cout << "Tensors after flips: " << s_flip.get_data().size() / 3 << "\n";
        
        bool flip_correct = check_scheme(s_flip.get_data(), n);
        std::cout << "Flip correctness: " << (flip_correct ? "✓" : "✗") << "\n";
    }
    
    // 2️⃣ ТЕСТ REDUCTION ПРОИЗВОДИТЕЛЬНОСТИ
    {
        std::cout << "\n=== REDUCTION PERFORMANCE TEST ===\n";
        Scheme s_reduction(data, 123, n);
        
        const long long REDUCTION_LIMIT = 100'000;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        long long successful_reductions = 0;
        for (long long i = 0; i < REDUCTION_LIMIT; ++i) {
            if (s_reduction.reduction()) {
                successful_reductions++;
            } else {
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "Reduction rate: " << static_cast<int>(successful_reductions / elapsed) << " reductions/sec\n";
        std::cout << "Tensors after reductions: " << s_reduction.get_data().size() / 3 << "\n";
        
        bool reduction_correct = check_scheme(s_reduction.get_data(), n);
        std::cout << "Reduction correctness: " << (reduction_correct ? "✓" : "✗") << "\n";
    }
    
    // 3️⃣ КОМБИНИРОВАННЫЙ ТЕСТ (FLIP + REDUCTION)
    {
        std::cout << "\n=== COMBINED FLIP + REDUCTION TEST ===\n";
        Scheme s_combined(data, 456, n);
        
        const long long OPERATIONS_LIMIT = 10'000'000;
        const long long FLIPS_LIMIT = 300;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        long long flips = 0, reductions = 0;
        for (long long i = 0; i < OPERATIONS_LIMIT; ++i) {
            // Чередуем flip и reduction
            if (i % FLIPS_LIMIT == 0) {
                if (s_combined.flip()) flips++;
            } else {
                if (s_combined.reduction()) reductions++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "Total flips: " << flips << ", rate: " << static_cast<int>(flips / elapsed) << "/sec\n";
        std::cout << "Total reductions: " << reductions << ", rate: " << static_cast<int>(reductions / elapsed) << "/sec\n";
        std::cout << "Final tensors: " << s_combined.get_data().size() / 3 << "\n";
        
        bool combined_correct = check_scheme(s_combined.get_data(), n);
        std::cout << "Combined correctness: " << (combined_correct ? "✓" : "✗") << "\n";
    }
    
    // 4️⃣ СПЕЦИАЛЬНЫЙ ТЕСТ ДЛЯ REDUCTION
    {
        std::cout << "\n=== REDUCTION SPECIFIC TESTS ===\n";
        test_reduction_scenarios(n);
    }
    
    return 0;
}