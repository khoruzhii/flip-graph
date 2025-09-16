// utils.h
#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "field.h"

template<typename Field>
std::vector<Field> generate_trivial_decomposition(int n) {
    std::vector<Field> data;
    data.reserve(n * n * n * 3);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                Field u = Field::unit(i * n + j);
                Field v = Field::unit(j * n + k);
                Field w = Field::unit(k * n + i);
                data.push_back(u);
                data.push_back(v);
                data.push_back(w);
            }
        }
    }
    return data;
}

// Extract coefficient: primary template undefined
template<typename Field>
inline int get_coefficient(const Field& f, int idx);

// Specialization for Bn<2>
template<>
inline int get_coefficient<B<2>>(const B<2>& f, int idx) {
    return f.get_coefficient(idx);
}

// Specialization for Bn<3>
template<>
inline int get_coefficient<B<3>>(const B<3>& f, int idx) {
    return f.get_coefficient(idx);
}

template<typename Field>
bool verify_scheme(const std::vector<Field>& data, int n) {
    int r = data.size() / 3;
    constexpr int mod = field_traits<Field>::is_mod2 ? 2 : 3;
    for (int i1 = 0; i1 < n; ++i1) {
        for (int i2 = 0; i2 < n; ++i2) {
            for (int j1 = 0; j1 < n; ++j1) {
                for (int j2 = 0; j2 < n; ++j2) {
                    for (int k1 = 0; k1 < n; ++k1) {
                        for (int k2 = 0; k2 < n; ++k2) {
                            int sum = 0;
                            for (int l = 0; l < r; ++l) {
                                const Field& u = data[3*l];
                                const Field& v = data[3*l+1];
                                const Field& w = data[3*l+2];
                                if (u.is_zero()) continue;
                                int a = get_coefficient<Field>(u, i1*n + i2);
                                int b = get_coefficient<Field>(v, j1*n + j2);
                                int c = get_coefficient<Field>(w, k1*n + k2);
                                sum = (sum + a * b * c) % mod;
                            }
                            int expected = (i2==j1 && j2==k1 && k2==i1) ? 1 : 0;
                            if (sum != expected) return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

// Print single Field element
template<typename Field>
void print_field(const Field& f, int n);

// Specialization for B<2>
template<>
inline void print_field<B<2>>(const B<2>& f, int n) {
    for (int j = 0; j < n*n; ++j) {
        std::cout << f.get_coefficient(j);
    }
}

// Specialization for B<3>
template<>
inline void print_field<B<3>>(const B<3>& f, int n) {
    for (int j = 0; j < n*n; ++j) {
        std::cout << f.get_coefficient(j);
    }
}

template<typename Field>
void print_scheme(const std::vector<Field>& data, int n) {
    int n_terms = data.size() / 3;
    for (int i = 0; i < n_terms; ++i) {
        if (data[3*i].is_zero()) continue;
        std::cout << "Term " << i << ": ";
        print_field<Field>(data[3*i], n);
        std::cout << " ";
        print_field<Field>(data[3*i+1], n);
        std::cout << " ";
        print_field<Field>(data[3*i+2], n);
        std::cout << "\n";
    }
}
