#pragma once
#include <vector>
#include <random>
#include <cstdint>
#include <array>
#include <type_traits>

template<typename FieldType, typename MapType>
class Scheme {
public:
    Scheme(const std::vector<FieldType>& initial, uint32_t seed, int n);
    
    bool flip();
    bool plus();
    bool reduction();
    
    bool flipu() { return flip_private(0); }
    bool flipv() { return flip_private(1); }
    bool flipw() { return flip_private(2); }
    
    bool reductionuv() const { return reduction_private(0); }
    bool reductionvw() const { return reduction_private(1); }
    bool reductionwu() const { return reduction_private(2); }
    
    const std::vector<FieldType>& get_flip(int type) const { return flippable[type]; }
    const MapType& get_map(int type) const { return pos[type]; }
    const std::vector<FieldType>& get_data() const { return data; }
    
    int get_rank() const { return rank; }
    int get_n() const { return n; }

private:
    bool flip_private(int type);
    bool reduction_private(int type);
    void delete_element(size_t id);
    void add_element(size_t id, const FieldType& x);
    void remove_tensor(size_t i);
    void value_assign(size_t id, const FieldType& x);

private:
    int n = 0;
    int cap;
    int rank = 0;
    std::mt19937 rng{123456u};
    
    std::vector<FieldType> data;
    std::array<MapType, 3> pos;
    std::array<std::vector<FieldType>, 3> flippable;
    std::vector<int> idx_next;
    std::vector<int> idx_prev;
};

template<typename FieldType, typename MapType>
void Scheme<FieldType, MapType>::delete_element(size_t id) {
    int type = id % 3;
    FieldType x = data[id];

    auto& inds = pos[type][x];
    auto it1 = std::find(inds.begin(), inds.end(), id);

    assert(it1 != inds.end());
    inds.erase(it1);

    if (inds.size() == 1) {
        auto& flippable_x = flippable[type];
        auto it2 = std::find(flippable_x.begin(), flippable_x.end(), x);
        flippable_x.erase(it2);
    }
}

template<typename FieldType, typename MapType>
void Scheme<FieldType, MapType>::add_element(size_t id, const FieldType& x) {
    int type = id % 3;
    data[id] = x;

    pos[type][x].push_back(id);
    auto& inds = pos[type][x];

    if (inds.size() == 2) {
        flippable[type].push_back(x);
    }
}

template<typename FieldType, typename MapType>
void Scheme<FieldType, MapType>::value_assign(size_t id, const FieldType& x) {
    delete_element(id);
    add_element(id, x);
}

template<typename FieldType, typename MapType>
void Scheme<FieldType, MapType>::remove_tensor(size_t i) {
    size_t idu = i * 3;
    size_t idv = idu + 1;
    size_t idw = idv + 1;

    size_t lastu = rank * 3 - 3;
    size_t lastv = rank * 3 - 2;
    size_t lastw = rank * 3 - 1;

    if (idu == lastu) {
        delete_element(lastu);
        delete_element(lastv);
        delete_element(lastw);

        data.pop_back(); data.pop_back(); data.pop_back();

        rank--;

        return;
    }

    FieldType u = data[lastu], v = data[lastv], w = data[lastw];

    delete_element(lastu);
    delete_element(lastv);
    delete_element(lastw);

    delete_element(idu);
    delete_element(idv);
    delete_element(idw);

    add_element(idu, u);
    add_element(idv, v);
    add_element(idw, w);

    data.pop_back(); data.pop_back(); data.pop_back();

    rank--;
}

template<typename FieldType, typename MapType>
bool Scheme<FieldType, MapType>::flip() {
    std::array<int, 3> p{0, 1, 2};
    std::shuffle(p.begin(), p.end(), rng);

    if (flip_private(p[0])) {
        return true;
    }
    if (flip_private(p[1])) {
        return true;
    }
    if (flip_private(p[2])) {
        return true;
    }
    return false;
}

template<typename FieldType, typename MapType>
bool Scheme<FieldType, MapType>::flip_private(int type) {
    auto& pos_u             = pos[type];
    auto& flippable_u = flippable[type];

    if (flippable_u.empty()) {
        return false;
    }
    // gen random repeating element for flip
    std::uniform_int_distribution<size_t> dist1(0, flippable_u.size() - 1);
    FieldType x = flippable_u[dist1(rng)];
    
    // gen 2 random ids
    auto& p = pos_u[x];
    size_t sz = p.size();
    std::uniform_int_distribution<size_t> dist2(1, sz - 1);

    size_t id1 = dist2(rng);
    size_t id2 = (id1 + dist2(rng)) % sz;
    int idu1 = p[id1];
    int idu2 = p[id2];

    int idv1 = idx_prev[idu1];
    int idw1 = idx_next[idu1];
    int idv2 = idx_prev[idu2];
    int idw2 = idx_next[idu2];

    // geting values
    FieldType u1 = data[idu1];
    FieldType v1 = data[idv1];
    FieldType w1 = data[idw1];
    FieldType u2 = data[idu2]; // del later
    FieldType v2 = data[idv2];
    FieldType w2 = data[idw2];

    assert(u1 == u2);

    FieldType new_w1 = w1 - w2;
    FieldType new_v2 = v1 + v2;

    if (v1 == v2 && w1 == w2) {
        if (idu1 < idu2) {
            std::swap(idu1, idu2);
        }
        remove_tensor(idu1 / 3);
        remove_tensor(idu2 / 3);

        assert(idu1 != idu2);

        return true;
    }
    if (w1 == w2) {
        delete_element(idv2);
        add_element(idv2, new_v2);

        remove_tensor(idw1 / 3);

        return true;
    }
    if (v1 == v2) {
        delete_element(idw1);
        add_element(idw1, new_w1);

        remove_tensor(idv2 / 3);

        return true;
    }

    // delete old elemets
    delete_element(idw1);
    delete_element(idv2);
    
    // add new elements
    add_element(idw1, new_w1);
    add_element(idv2, new_v2);
    
    return true;
}

template<typename FieldType, typename MapType>
Scheme<FieldType, MapType>::Scheme(const std::vector<FieldType>& initial, uint32_t seed, int n) : n(n), rng(seed) {
    data = initial;
    rank = data.size() / 3;
    cap = rank * 6;

    idx_next.assign(cap, 0);
    idx_prev.assign(cap, 0);
    for (int i = 0; i < cap; i += 3) {
        idx_next[i]     = i + 2; idx_prev[i]     = i + 1;
        idx_next[i + 1] = i;     idx_prev[i + 1] = i + 2;
        idx_next[i + 2] = i + 1; idx_prev[i + 2] = i;
    }

    for (int i = 0; i < 3 * rank; i++) {
        pos[i % 3][data[i]].push_back(i);
        if (pos[i % 3][data[i]].size() == 2) {
            flippable[i % 3].push_back(data[i]);
        }
    }
}
