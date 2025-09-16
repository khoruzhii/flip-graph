// stress_test_flips.cpp
#include "field.h"
#include "utils.h"
#include "scheme.h"
#include "CLI11.hpp"
#include "unordered_dense.h"
#include <iostream>
#include <unordered_map>
#include <map>
#include <chrono>
#include <random>
#include <iomanip>

// Типы карт
enum class MapType {
    Standard = 0,   // std::unordered_map
    Fast = 1,       // ankerl::unordered_dense::map  
    Ordered = 2     // std::map
};

// Аргументы командной строки
struct TestArgs {
    int field_mod = 2;           // Модуль поля (2 или 3)
    int map_type_int = 0;        // Тип карты (0=std, 1=fast, 2=ordered)
    size_t num_flips = 100000;      // Количество флипов
    int matrix_size = 3;         // Размер матрицы
    
    MapType get_map_type() const {
        return static_cast<MapType>(map_type_int);
    }
};

// Результат теста
struct TestResult {
    size_t successful_flips = 0;
    size_t failed_flips = 0;
    int initial_rank = 0;
    int final_rank = 0;
    int min_rank_reached = 0;
    double elapsed_seconds = 0.0;
    double flips_per_second = 0.0;
    bool correctness_passed = false;
    uint32_t seed_used = 0;
    
    void print_results(const TestArgs& args) const {
        std::string field_name = (args.field_mod == 2) ? "F_2" : "F_3";
        std::string map_name;
        
        switch(args.get_map_type()) {
            case MapType::Standard: map_name = "std::unordered_map"; break;
            case MapType::Fast: map_name = "ankerl::unordered_dense"; break;
            case MapType::Ordered: map_name = "std::map"; break;
        }
        
        std::cout << "\n" << std::string(55, '=') << "\n";
        std::cout << "FLIP STRESS TEST RESULTS\n";
        std::cout << std::string(55, '=') << "\n";
        std::cout << "Field:            " << field_name << "\n";
        std::cout << "Map type:         " << map_name << "\n";
        std::cout << "Matrix size:      " << args.matrix_size << "×" << args.matrix_size << "\n";
        std::cout << "Total flips:      " << args.num_flips << "\n";
        std::cout << "Successful:       " << successful_flips << "\n";
        std::cout << "Failed:           " << failed_flips << "\n";
        std::cout << "Seed:             " << seed_used << "\n";
        std::cout << std::string(55, '-') << "\n";
        
        // Информация о ранге
        std::cout << "Initial rank:     " << initial_rank << "\n";
        std::cout << "Final rank:       " << final_rank << "\n";
        std::cout << "Min rank reached: " << min_rank_reached << "\n";
        std::cout << "Rank change:      " << initial_rank << " → " << final_rank;
        if (final_rank < initial_rank) {
            std::cout << " (reduced by " << (initial_rank - final_rank) << ")";
        } else if (final_rank > initial_rank) {
            std::cout << " (increased by " << (final_rank - initial_rank) << ")";
        } else {
            std::cout << " (no change)";
        }
        std::cout << "\n";
        
        std::cout << std::string(55, '-') << "\n";
        std::cout << "Time elapsed:     " << std::fixed << std::setprecision(3) 
                  << elapsed_seconds << " seconds\n";
        std::cout << "Performance:      " << std::fixed << std::setprecision(0) 
                  << flips_per_second << " flips/second\n";
        
        // Проверка корректности
        std::cout << "Correctness:      ";
        if (correctness_passed) {
            std::cout << "✅ PASSED\n";
        } else {
            std::cout << "❌ FAILED\n";
        }
        
        std::cout << std::string(55, '=') << "\n";
    }
};

// Парсинг аргументов
TestArgs parse_args(int argc, char** argv) {
    TestArgs args;
    CLI::App app{"Flip Performance Stress Tests"};
    
    // Модуль поля
    app.add_option("-m,--modulus", args.field_mod, "Field modulus (2 or 3)")
        ->check(CLI::IsMember({2, 3}));
    
    // Тип карты
    std::map<std::string, int> map_options{
        {"std", 0}, {"standard", 0},
        {"fast", 1}, {"dense", 1}, {"ankerl", 1},
        {"ordered", 2}, {"map", 2}
    };
    app.add_option("-t,--map-type", args.map_type_int, "Map type: std, fast, ordered")
        ->transform(CLI::CheckedTransformer(map_options, CLI::ignore_case))
        ->default_val(0);
    
    // Количество флипов
    app.add_option("-n,--flips", args.num_flips, "Number of flips to perform")
        ->check(CLI::Range(1ULL, 1000000000000ULL));
    
    // Размер матрицы
    app.add_option("-s,--size", args.matrix_size, "Matrix size (NxN)")
        ->check(CLI::Range(2, 8));
    
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        exit(app.exit(e));
    }
    
    return args;
}

// Генерация случайного сида
uint32_t generate_random_seed() {
    std::random_device rd;
    return rd();
}

// Основная функция теста
template<typename FieldType, typename MapContainer>
TestResult run_flip_test(const TestArgs& args) {
    TestResult result;
    result.seed_used = generate_random_seed();
    
    std::cout << "Initializing " << args.matrix_size << "×" << args.matrix_size 
              << " matrix (seed: " << result.seed_used << ")...\n";
    
    // Создаем схему с случайным сидом
    auto data = generate_trivial_decomposition<FieldType>(args.matrix_size);
    Scheme<FieldType, MapContainer> scheme(data, result.seed_used, args.matrix_size);
    
    // Запоминаем начальный ранг
    result.initial_rank = scheme.get_rank();
    result.min_rank_reached = result.initial_rank;
    
    std::cout << "Initial rank: " << result.initial_rank << "\n";
    std::cout << "Starting " << args.num_flips << " flips...\n";
    
    // Засекаем время
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Выполняем флипы
    for (size_t i = 0; i < args.num_flips; ++i) {
        if (scheme.flip()) {
            result.successful_flips++;
            
            // Отслеживаем минимальный ранг
            int current_rank = scheme.get_rank();
            result.min_rank_reached = std::min(result.min_rank_reached, current_rank);
        } else {
            result.failed_flips++;
        }
        
        // Прогресс каждые 10%
        if (args.num_flips >= 1000 && (i + 1) % (args.num_flips / 10) == 0) {
            size_t progress = (i + 1) * 100 / args.num_flips;
            std::cout << "Progress: " << progress << "%\r" << std::flush;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.flips_per_second = args.num_flips / result.elapsed_seconds;
    
    // Получаем финальный ранг
    result.final_rank = scheme.get_rank();
    
    if (args.num_flips >= 1000) {
        std::cout << "\nCompleted!                    \n";
    }
    
    std::cout << "Final rank: " << result.final_rank << "\n";
    
    // Проверяем корректность
    std::cout << "Verifying correctness...";
    result.correctness_passed = verify_scheme(scheme.get_data(), args.matrix_size);
    if (result.correctness_passed) {
        std::cout << " ✅ PASSED\n";
    } else {
        std::cout << " ❌ FAILED\n";
    }
    
    return result;
}

// Диспетчер для разных типов карт
template<int N>
TestResult dispatch_by_map_type(const TestArgs& args) {
    switch(args.get_map_type()) {
        case MapType::Standard:
            return run_flip_test<B<N>, std::unordered_map<B<N>, std::vector<int>>>(args);
            
        case MapType::Fast:
            return run_flip_test<B<N>, ankerl::unordered_dense::map<B<N>, std::vector<int>>>(args);
            
        case MapType::Ordered:
            return run_flip_test<B<N>, std::map<B<N>, std::vector<int>>>(args);
    }
    
    throw std::runtime_error("Unknown map type");
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        
        std::cout << "Flip Stress Testing\n";
        std::cout << "==================\n";
        
        // Запускаем тест
        TestResult result;
        if (args.field_mod == 2) {
            result = dispatch_by_map_type<2>(args);
        } else {
            result = dispatch_by_map_type<3>(args);
        }
        
        // Выводим результаты
        result.print_results(args);
        
        // Проверяем статус завершения
        if (!result.correctness_passed) {
            std::cout << "\n❌ TEST FAILED: Correctness verification failed!\n";
            return 1;
        }
        
        std::cout << "\n✅ Test completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}