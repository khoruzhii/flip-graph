#include "scheme.h"
#include <cassert>
#include <algorithm>

void Scheme::delete_element(int id) {
    int type = id % 3;
    U64 x = data[id];

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

void Scheme::add_element(int id, U64 x) {
    int type = id % 3;
    data[id] = x;

    pos[type][x].push_back(id);
    auto& inds = pos[type][x];

    if (inds.size() == 2) {
        flippable[type].push_back(x);
    }
}

void Scheme::value_assign(int id, U64 x) {
    delete_element(id);
    add_element(id, x);
}

void Scheme::remove_tensor(int i) {
    int idu = i * 3;
    int idv = idu + 1;
    int idw = idv + 1;

    int lastu = rank * 3 - 3;
    int lastv = rank * 3 - 2;
    int lastw = rank * 3 - 1;

    if (idu == lastu) {
        delete_element(lastu);
        delete_element(lastv);
        delete_element(lastw);

        data.pop_back(); data.pop_back(); data.pop_back();

        rank--;

        return;
    }

    U64 u = data[lastu], v = data[lastv], w = data[lastw];

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

bool Scheme::flip() {
    // random flip-func permutation 
    std::array<int, 3> p{0, 1, 2};
    std::shuffle(p.begin(), p.end(), rng);

    if (flip_private(0)) {
        return true;
    }
    if (flip_private(1)) {
        return true;
    }
    if (flip_private(2)) {
        return true;
    }
    return false;
}

bool Scheme::flip_private(int type) {
    auto& pos_u             = pos[type];      // std::unordered_map<int,std::vector<int>>&
    auto& flippable_u = flippable[type];      // std::vector<U64>&

    if (flippable_u.empty()) {
        return false;
    }
    // gen random repeating element for flip
    std::uniform_int_distribution<size_t> dist1(0, flippable_u.size() - 1);
    U64 x = flippable_u[dist1(rng)];
    
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
    U64 u1 = data[idu1];
    U64 v1 = data[idv1];
    U64 w1 = data[idw1];
    U64 u2 = data[idu2]; // del later
    U64 v2 = data[idv2];
    U64 w2 = data[idw2];

    assert(u1 == u2);

    U64 new_w1 = w1 ^ w2;
    U64 new_v2 = v1 ^ v2;

    if (v1 == v2 && w1 == w2) {
        remove_tensor(idu1 / 3);
        remove_tensor(idu2 / 3);

        assert(idu1 != idu2);

        return true;
    }
    if (w1 == w2) {
        remove_tensor(idw1 / 3);

        delete_element(idv2);
        add_element(idv2, new_v2);

        return true;
    }
    if (v1 == v2) {
        remove_tensor(idv2 / 3);

        delete_element(idw1);
        add_element(idw1, new_w1);

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

bool Scheme::reduction() {
    std::array<int, 3> p{0, 1, 2};
    std::shuffle(p.begin(), p.end(), rng);

    return reduction_private(0);
}

bool Scheme::reduction_private(int type) {
    for (int i = 0; i < rank; ++i) {
        for (int j = i + 1; j < rank; ++j) {
            int idu1 = 3 * i;
            int idv1 = 3 * i + 1;
            int idw1 = 3 * i + 2;
            U64 u1 = data[idu1];
            U64 v1 = data[idv1];
            U64 w1 = data[idw1];

            int idu2 = 3 * j;
            int idv2 = 3 * j + 1;
            int idw2 = 3 * j + 2;
            U64 u2 = data[idu2];
            U64 v2 = data[idv2];
            U64 w2 = data[idw2];

            assert(idu1 != idu2);

            if (v1 == v2 && w1 == w2) {
                int tensor1 = idu1 / 3;
                int tensor2 = idu2 / 3;
                
                if (tensor1 > tensor2) {
                    remove_tensor(tensor1);
                    remove_tensor(tensor2);
                } else {
                    remove_tensor(tensor2);
                    remove_tensor(tensor1);
                }
                return true;
            }

            if (u1 == u2 && v1 == v2) {
                U64 new_w1 = w1 ^ w2;
                remove_tensor(idw2 / 3);

                delete_element(idw1);
                add_element(idw1, new_w1);

                return true;
            }

            if (u1 == u2 && w1 == w2) {
                U64 new_v1 = v1 ^ v2;
                remove_tensor(idv2 / 3);

                delete_element(idv1);
                add_element(idv1, new_v1);

                return true;
            }

            if (v1 == v2 && w1 == w2) {
                U64 new_u1 = u1 ^ u2;
                remove_tensor(idu2 / 3);

                delete_element(idu1);
                add_element(idu1, new_u1);

                return true;
            }
        }
    }
    return false;
}

Scheme::Scheme(const std::vector<U64>& initial, uint32_t seed, int n) : n(n), rng(seed) {
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