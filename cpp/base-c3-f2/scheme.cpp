// scheme.cpp
#include "scheme.h"
#include <cassert>

Scheme::Scheme(const std::vector<U64>& initial, int sym_in, uint32_t seed)
    : sym(sym_in), rng(seed) {
    assert(sym == 3 || sym == 6);

    data = initial;
    n = static_cast<int>(data.size());
    assert(n % 3 == 0);

    // Build index helpers for v and w components per row
    idx_next.assign(n, 0);
    idx_prev.assign(n, 0);
    for (int i = 0; i < n; i += 3) {
        idx_next[i]     = i + 2; idx_prev[i]     = i + 1;
        idx_next[i + 1] = i;     idx_prev[i + 1] = i + 2;
        idx_next[i + 2] = i + 1; idx_prev[i + 2] = i;
    }

    // Permit matrix: 0 if same orbit, 1 otherwise
    permit.assign(n, std::vector<uint8_t>(n, 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            permit[i][j] = (i / sym == j / sym) ? 0 : 1;
        }
    }

    // Reset dictionaries and lists
    unique = HashDict();
    flippable_idx = HashDict();
    flippable.clear();

    // Allocate position blocks: for each distinct value a block [len, idx1, ...]
    const int block_len = n + 1;
    pos.assign(n * block_len, 0);
    free_slots.clear();
    free_slots.reserve(n);
    for (int i = 0; i < n; ++i) {
        int b = i * block_len;
        free_slots.push_back(b);
    }

    // Initial fill of unique / flippable based on current data
    rank = 0;
    for (int i = 0; i < n; ++i) {
        U64 m = data[i];
        if (m == 0ULL) continue;

        if (unique.contains(m)) {
            int b = unique.getvaluex(m);
            int l = pos[b];
            ++l;
            pos[b + l] = i;
            pos[b] = l;
            if (!flippable_idx.contains(m)) {
                flippable_idx.addx(m, static_cast<int>(flippable.size()));
                flippable.push_back(m);
            }
        } else {
            int b = free_slots.back(); free_slots.pop_back();
            unique.addx(m, b);
            pos[b] = 1;
            pos[b + 1] = i;
        }
        rank += 1;
    }

    // Build pair selection tables (covers lengths up to 79 as in the original)
    pair_starts.clear(); pair_i.clear(); pair_j.clear();
    pair_starts.reserve(100);
    pair_i.reserve(6400);
    pair_j.reserve(6400);
    pair_starts.push_back(0);
    pair_starts.push_back(0);
    for (int x = 1; x < 80; ++x) {
        for (int y = 0; y < x; ++y) {
            pair_i.push_back(x);
            pair_j.push_back(y);
            pair_i.push_back(y);
            pair_j.push_back(x);
        }
        pair_starts.push_back(static_cast<int>(pair_i.size()));
    }
}

void Scheme::del(int r, U64 v) {
    int b = unique.getvalue(v);
    int l = pos[b];
    if (l == 2) {
        // Will drop from 2 -> 1; remove this value from flippable
        flippable_idx.lasthash = unique.lasthash;
        int idx = flippable_idx.getvaluex(v);
        U64 last_val = flippable.back();
        flippable_idx.replace(last_val, idx);
        flippable[idx] = last_val;
        flippable.pop_back();
        flippable_idx.lasthash = unique.lasthash;
        flippable_idx.removex(v);
    }
    if (l == 1) {
        // Now 0 -> release the block and drop from unique
        free_slots.push_back(b);
        unique.removex(v);
    } else {
        // Remove row r from the block by rotating from the end
        int i = b + l;
        int x = pos[i];
        while (x != r) {
            --i;
            int y = x;
            x = pos[i];
            pos[i] = y;
        }
        pos[b] = l - 1;
    }
}

void Scheme::add(int r, U64 v) {
    int present = unique.contains(v);
    if (present) {
        int b = unique.getvaluex(v);
        int l = pos[b];
        if (l == 1) {
            // Will become multiplicity 2 -> insert into flippable
            flippable_idx.lasthash = unique.lasthash;
            flippable_idx.addx(v, static_cast<int>(flippable.size()));
            flippable.push_back(v);
        }
        ++l;
        pos[b + l] = r;
        pos[b] = l;
    } else {
        int b = free_slots.back(); free_slots.pop_back();
        unique.addx(v, b);
        pos[b + 1] = r;
        pos[b] = 1;
    }
}

bool Scheme::flip() {
    if (flippable.empty()) return false;
    return (sym == 3) ? flip3() : flip6();
}

bool Scheme::plus() {
    // Find first free row (u component is zero)
    int r = -1;
    for (int i = 0; i < n; ++i) { if (data[i] == 0ULL) { r = i; break; } }
    if (r < 0) return false;
    return (sym == 3) ? plus3() : plus6();
}

bool Scheme::flip3() {
    while (true) {
        unsigned int sample = rng();
        U64 val = flippable[sample % flippable.size()];
        int b = unique.getvalue(val);
        int l = pos[b];
        ++b; // point to first index entry
        int p, q;
        if (l == 2) {
            if (sample & 65536U) {
                p = pos[b];
                q = pos[b + 1];
            } else {
                p = pos[b + 1];
                q = pos[b];
            }
        } else {
            int x = static_cast<int>((sample >> 16) % pair_starts[l]);
            p = pos[b + pair_i[x]];
            q = pos[b + pair_j[x]];
        }
        if (!permit[p][q]) continue;

        U64 pv = data[idx_next[p]];
        U64 pw = data[idx_prev[p]];
        U64 qv = data[idx_next[q]];
        U64 qw = data[idx_prev[q]];
        U64 pv_new = qv ^ pv;
        U64 qw_new = qw ^ pw;

        del(idx_next[p], pv); add(idx_next[p], pv_new); data[idx_next[p]] = pv_new;
        del(idx_prev[q], qw); add(idx_prev[q], qw_new); data[idx_prev[q]] = qw_new;

        if (pv_new == 0ULL) {
            U64 pu = data[p];
            del(p, pu);
            del(idx_next[p], pv_new);
            del(idx_prev[p], pw);
            data[p] = 0ULL;
            data[idx_prev[p]] = 0ULL;
            rank -= 3;
        }

        if (qw_new == 0ULL) {
            U64 qu = data[q];
            del(q, qu);
            del(idx_next[q], qv);
            del(idx_prev[q], qw_new);
            data[q] = 0ULL;
            data[idx_next[q]] = 0ULL;
            rank -= 3;
        }

        return true;
    }
}

bool Scheme::flip6() {
    while (true) {
        unsigned int sample = rng();
        U64 val = flippable[sample % flippable.size()];
        int b = unique.getvalue(val);
        int l = pos[b];
        ++b;
        int p, q;
        if (l == 2) {
            if (sample & 65536U) {
                p = pos[b];
                q = pos[b + 1];
            } else {
                p = pos[b + 1];
                q = pos[b];
            }
        } else {
            int x = static_cast<int>((sample >> 16) % pair_starts[l]);
            p = pos[b + pair_i[x]];
            q = pos[b + pair_j[x]];
        }
        if (!permit[p][q]) continue;

        int pp = (p % 6 < 3) ? p + 3 : p - 3;
        int qq = (q % 6 < 3) ? q + 3 : q - 3;

        U64 pu = data[p],  pv = data[idx_next[p]],  pw = data[idx_prev[p]];
        U64 qu = data[q],  qv = data[idx_next[q]],  qw = data[idx_prev[q]];
        U64 ppu = data[pp], ppv = data[idx_next[pp]], ppw = data[idx_prev[pp]];
        U64 qqu = data[qq], qqv = data[idx_next[qq]], qqw = data[idx_prev[qq]];

        U64 pv_new  = qv  ^ pv;
        U64 qw_new  = qw  ^ pw;
        U64 ppv_new = qqv ^ ppv;
        U64 qqw_new = qqw ^ ppw;

        // Apply v-updates on p and pp
        del(idx_next[p],  pv);  add(idx_next[p],  pv_new);  data[idx_next[p]]  = pv_new;
        del(idx_next[pp], ppv); add(idx_next[pp], ppv_new); data[idx_next[pp]] = ppv_new;

        // Apply w-updates on q and qq
        del(idx_prev[q],  qw);  add(idx_prev[q],  qw_new);  data[idx_prev[q]]  = qw_new;
        del(idx_prev[qq], qqw); add(idx_prev[qq], qqw_new); data[idx_prev[qq]] = qqw_new;

        if (pv_new == 0ULL || (pu == ppu && pv_new == ppv_new && pw == ppw)) {
            del(p,  pu);
            del(idx_next[p],  pv_new);
            del(idx_prev[p],  pw);
            data[p] = 0ULL;
            data[idx_prev[p]] = 0ULL;

            del(pp, ppu);
            del(idx_next[pp], ppv_new);
            del(idx_prev[pp], ppw);
            data[pp] = 0ULL;
            data[idx_prev[pp]] = 0ULL;

            if (pv_new != 0ULL) {
                data[idx_next[p]]  = 0ULL;
                data[idx_next[pp]] = 0ULL;
            }
            rank -= 6;
        }

        if (qw_new == 0ULL || (qu == qqu && qv == qqv && qw_new == qqw_new)) {
            del(q,  qu);
            del(idx_next[q],  qv);
            del(idx_prev[q],  qw_new);
            data[q] = 0ULL;
            data[idx_next[q]] = 0ULL;

            del(qq, qqu);
            del(idx_next[qq], qqv);
            del(idx_prev[qq], qqw_new);
            data[qq] = 0ULL;
            data[idx_next[qq]] = 0ULL;

            if (qw_new != 0ULL) {
                data[idx_prev[q]]  = 0ULL;
                data[idx_prev[qq]] = 0ULL;
            }
            rank -= 6;
        }

        return true;
    }
}

bool Scheme::plus3() {
    // Find first free row r
    int r = -1;
    for (int i = 0; i < n; ++i) { if (data[i] == 0ULL) { r = i; break; } }
    if (r < 0) return false;

    int p, q;
    U64 pu, pv, pw, qu, qv, qw;
    U64 pu_new, pv_new, pw_new, qu_new, qv_new, qw_new, ru_new, rv_new, rw_new;

    while (true) {
        p = static_cast<int>(rng() % n);
        q = static_cast<int>(rng() % n);

        pu = data[p];  pv = data[idx_next[p]];  pw = data[idx_prev[p]];
        qu = data[q];  qv = data[idx_next[q]];  qw = data[idx_prev[q]];

        pu_new = pu;
        pv_new = pv ^ qv;
        pw_new = pw;

        qu_new = pu;
        qv_new = qv;
        qw_new = pw ^ qw;

        ru_new = pu ^ qu;
        rv_new = qv;
        rw_new = qw;

        bool ok = true;
        if (pu == 0ULL || qu == 0ULL) ok = false;
        if (pu == qu || pv == qv || pw == qw) ok = false;
        if (!permit[p][q]) ok = false;
        if (ok) break;
    }

    del(idx_next[p], pv); add(idx_next[p], pv_new);
    del(q, qu);          add(q, pu);
    del(idx_prev[q], qw); add(idx_prev[q], qw_new);
    add(r, ru_new);
    add(idx_next[r], rv_new);
    add(idx_prev[r], rw_new);

    data[p]             = pu_new;
    data[idx_next[p]]   = pv_new;
    data[idx_prev[p]]   = pw_new;
    data[q]             = qu_new;
    data[idx_next[q]]   = qv_new;
    data[idx_prev[q]]   = qw_new;
    data[r]             = ru_new;
    data[idx_next[r]]   = rv_new;
    data[idx_prev[r]]   = rw_new;

    rank += 3;
    return true;
}

bool Scheme::plus6() {
    // Find first free row r (paired row rr = r + 3 must be valid by invariant)
    int r = -1;
    for (int i = 0; i < n; ++i) { if (data[i] == 0ULL) { r = i; break; } }
    if (r < 0 || r + 3 >= n) return false;
    int rr = r + 3;

    int p, q, pp, qq;
    U64 pu, pv, pw, qu, qv, qw;
    U64 ppu, ppv, ppw, qqu, qqv, qqw;
    U64 pu_new, pv_new, pw_new, qu_new, qv_new, qw_new, ru_new, rv_new, rw_new;
    U64 ppu_new, ppv_new, ppw_new, qqu_new, qqv_new, qqw_new, rru_new, rrv_new, rrw_new;

    while (true) {
        p = static_cast<int>(rng() % n);
        q = static_cast<int>(rng() % n);
        pp = (p % 6 < 3) ? p + 3 : p - 3;
        qq = (q % 6 < 3) ? q + 3 : q - 3;

        pu = data[p];  pv = data[idx_next[p]];  pw = data[idx_prev[p]];
        qu = data[q];  qv = data[idx_next[q]];  qw = data[idx_prev[q]];

        ppu = data[pp]; ppv = data[idx_next[pp]]; ppw = data[idx_prev[pp]];
        qqu = data[qq]; qqv = data[idx_next[qq]]; qqw = data[idx_prev[qq]];

        pu_new  = pu;
        pv_new  = pv ^ qv;
        pw_new  = pw;

        qu_new  = pu;
        qv_new  = qv;
        qw_new  = pw ^ qw;

        ru_new  = pu ^ qu;
        rv_new  = qv;
        rw_new  = qw;

        ppu_new  = ppu;
        ppv_new  = ppv ^ qqv;
        ppw_new  = ppw;

        qqu_new  = ppu;
        qqv_new  = qqv;
        qqw_new  = ppw ^ qqw;

        rru_new  = ppu ^ qqu;
        rrv_new  = qqv;
        rrw_new  = qqw;

        bool ok = true;
        if (pu == 0ULL || qu == 0ULL) ok = false;
        if (ppu == 0ULL || qqu == 0ULL) ok = false;
        if (pu == qu || pv == qv || pw == qw) ok = false;
        if (ppu == qqu || ppv == qqv || ppw == qqw) ok = false;
        if (!permit[p][q]) ok = false;
        if (ok) break;
    }

    // First triple block
    del(idx_next[p], pv);  add(idx_next[p], pv_new);
    del(q, qu);            add(q, pu);
    del(idx_prev[q], qw);  add(idx_prev[q], qw_new);
    add(r, ru_new);
    add(idx_next[r], rv_new);
    add(idx_prev[r], rw_new);

    // Second triple block
    del(idx_next[pp], ppv); add(idx_next[pp], ppv_new);
    del(qq, qqu);           add(qq, ppu);
    del(idx_prev[qq], qqw); add(idx_prev[qq], qqw_new);
    add(rr, rru_new);
    add(idx_next[rr], rrv_new);
    add(idx_prev[rr], rrw_new);

    // Write data back
    data[p]             = pu_new;
    data[idx_next[p]]   = pv_new;
    data[idx_prev[p]]   = pw_new;
    data[q]             = qu_new;
    data[idx_next[q]]   = qv_new;
    data[idx_prev[q]]   = qw_new;
    data[r]             = ru_new;
    data[idx_next[r]]   = rv_new;
    data[idx_prev[r]]   = rw_new;

    data[pp]            = ppu_new;
    data[idx_next[pp]]  = ppv_new;
    data[idx_prev[pp]]  = ppw_new;
    data[qq]            = qqu_new;
    data[idx_next[qq]]  = qqv_new;
    data[idx_prev[qq]]  = qqw_new;
    data[rr]            = rru_new;
    data[idx_next[rr]]  = rrv_new;
    data[idx_prev[rr]]  = rrw_new;

    rank += 6;
    return true;
}
