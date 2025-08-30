#pragma once

#include <vector>
#include "unordered_dense.h"

template <typename T>
class indexed_set {
    std::vector<T> items_;
    ankerl::unordered_dense::map<T, int> pos_;

public:
    // Returns false if x already exists; O(1)
    bool insert(const T& x) {
        if (auto [it, ok] = pos_.try_emplace(x, items_.size()); ok) {
            items_.push_back(x);
            return true;
        }
        return false;
    }

    // Erase by value; O(1) via swap-and-pop
    bool erase(const T& x) {
        auto it = pos_.find(x);
        if (it == pos_.end()) return false;
        
        int i = it->second;
        if (i != items_.size() - 1) {
            pos_[items_[i] = std::move(items_.back())] = i;
        }
        items_.pop_back();
        pos_.erase(it);
        return true;
    }

    // O(1) queries
    int size() const noexcept { return items_.size(); }
    bool empty() const noexcept { return items_.empty(); }
    bool contains(const T& x) const { return pos_.count(x); }
    
    // O(1) access
    T& operator[](int i) { return items_[i]; }
    const T& operator[](int i) const { return items_[i]; }
};