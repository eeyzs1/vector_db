#include "hnsw.h"
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectordb {
namespace algorithms {

IndexHNSW::IndexHNSW(size_t dimension, size_t M, size_t ef_construction)
    : VectorStorage(dimension), M_(M), ef_construction_(ef_construction),
      ef_search_(ef_construction), max_level_(0), enter_point_(0),
      compact_slot_size_(0), compact_ready_(false),
      rng_(42), uniform_(0.0, 1.0), insert_visit_mark_(1) {
    ml_ = 1.0 / std::log(static_cast<double>(M_));
    memset(compact_level_offsets_, 0, sizeof(compact_level_offsets_));
}

size_t IndexHNSW::random_level() {
    float r = uniform_(rng_);
    if (r <= 0.0f) r = 1e-10f;
    double level = -std::log(static_cast<double>(r)) * ml_;
    return static_cast<size_t>(level);
}

void IndexHNSW::build_compact_neighbors() const {
    if (ntotal == 0) {
        compact_ready_ = true;
        return;
    }

    size_t n_levels = max_level_ + 1;
    compact_level_offsets_[0] = 0;
    for (size_t l = 0; l < n_levels; ++l) {
        size_t slot = (l == 0) ? 2 * M_ : M_;
        compact_level_offsets_[l + 1] = compact_level_offsets_[l] + slot;
    }
    compact_slot_size_ = compact_level_offsets_[n_levels];

    neighbors_compact_.assign(ntotal * compact_slot_size_, 0xFFFFFFFF);
    neighbor_counts_.assign(ntotal * n_levels, 0);

    for (size_t i = 0; i < ntotal; ++i) {
        const auto& nb_vec = neighbors_[i];
        uint32_t* dst = neighbors_compact_.data() + i * compact_slot_size_;
        uint16_t* counts = neighbor_counts_.data() + i * n_levels;

        for (size_t l = 0; l < n_levels; ++l) {
            size_t nb_offset = l * (2 * M_);
            size_t slot = (l == 0) ? 2 * M_ : M_;
            uint32_t* dst_level = dst + compact_level_offsets_[l];
            uint16_t count = 0;

            for (size_t ni = 0; ni < slot && ni + nb_offset < nb_vec.size(); ++ni) {
                int32_t nb = nb_vec[nb_offset + ni];
                if (nb < 0) break;
                dst_level[ni] = static_cast<uint32_t>(nb);
                count++;
            }
            counts[l] = count;
        }
    }

    compact_ready_ = true;
}

void IndexHNSW::precompute_norms() const {
    const float* base = data();
    size_t dim = d;
    norms_.resize(ntotal);
    for (size_t i = 0; i < ntotal; ++i) {
        norms_[i] = distance::compute_l2_distance(base + i * dim, base + i * dim, dim);
    }
}

void IndexHNSW::search_layer_impl(const float* query,
                                   float query_norm,
                                   size_t ef,
                                   size_t level,
                                   const int32_t* entry_points,
                                   size_t n_entry,
                                   float* out_dists,
                                   int32_t* out_ids,
                                   size_t* out_count,
                                   int32_t* visited,
                                   int32_t visit_mark) const {
    const float* base = data();
    size_t dim = d;
    size_t n_levels = max_level_ + 1;
    size_t level_offset = compact_level_offsets_[level];

    size_t heap_cap = ef + 2 * M_ + 64;

    float* cand_dists = static_cast<float*>(alloca(heap_cap * sizeof(float)));
    int32_t* cand_ids = static_cast<int32_t*>(alloca(heap_cap * sizeof(int32_t)));
    size_t cand_size = 0;

    float* res_dists = static_cast<float*>(alloca((ef + 1) * sizeof(float)));
    int32_t* res_ids = static_cast<int32_t*>(alloca((ef + 1) * sizeof(int32_t)));
    size_t res_size = 0;

    uint32_t* batch_ids = static_cast<uint32_t*>(alloca((2 * M_) * sizeof(uint32_t)));
    float* batch_dists = static_cast<float*>(alloca((2 * M_) * sizeof(float)));

    for (size_t i = 0; i < n_entry; ++i) {
        int32_t ep = entry_points[i];
        if (ep < 0 || (size_t)ep >= ntotal) continue;
        float dist = distance::compute_l2_distance(query, base + ep * dim, dim);
        visited[ep] = visit_mark;

        cand_dists[cand_size] = dist;
        cand_ids[cand_size] = ep;
        cand_size++;

        res_dists[res_size] = dist;
        res_ids[res_size] = ep;
        res_size++;
    }

    auto sift_up_max = [&](size_t idx) {
        while (idx > 0) {
            size_t p = (idx - 1) / 2;
            if (cand_dists[idx] > cand_dists[p]) {
                std::swap(cand_dists[idx], cand_dists[p]);
                std::swap(cand_ids[idx], cand_ids[p]);
                idx = p;
            } else break;
        }
    };

    auto sift_down_max = [&](size_t idx) {
        while (true) {
            size_t largest = idx;
            size_t l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < cand_size && cand_dists[l] > cand_dists[largest]) largest = l;
            if (r < cand_size && cand_dists[r] > cand_dists[largest]) largest = r;
            if (largest != idx) {
                std::swap(cand_dists[idx], cand_dists[largest]);
                std::swap(cand_ids[idx], cand_ids[largest]);
                idx = largest;
            } else break;
        }
    };

    auto sift_up_min = [&](size_t idx) {
        while (idx > 0) {
            size_t p = (idx - 1) / 2;
            if (res_dists[idx] < res_dists[p]) {
                std::swap(res_dists[idx], res_dists[p]);
                std::swap(res_ids[idx], res_ids[p]);
                idx = p;
            } else break;
        }
    };

    auto sift_down_min = [&](size_t idx) {
        while (true) {
            size_t smallest = idx;
            size_t l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < res_size && res_dists[l] < res_dists[smallest]) smallest = l;
            if (r < res_size && res_dists[r] < res_dists[smallest]) smallest = r;
            if (smallest != idx) {
                std::swap(res_dists[idx], res_dists[smallest]);
                std::swap(res_ids[idx], res_ids[smallest]);
                idx = smallest;
            } else break;
        }
    };

    for (size_t i = cand_size; i > 0; --i) sift_up_max(i - 1);
    for (size_t i = res_size; i > 0; --i) sift_up_min(i - 1);

    while (cand_size > 0) {
        if (cand_dists[0] > res_dists[0] && res_size >= ef) {
            break;
        }

        int32_t c = cand_ids[0];
        cand_dists[0] = cand_dists[cand_size - 1];
        cand_ids[0] = cand_ids[cand_size - 1];
        cand_size--;
        if (cand_size > 0) sift_down_max(0);

        if (c < 0 || (size_t)c >= ntotal) continue;

        const uint32_t* nb_ptr = neighbors_compact_.data() + c * compact_slot_size_ + level_offset;
        uint16_t nb_count = neighbor_counts_[c * n_levels + level];

        size_t n_new = 0;
        for (uint16_t ni = 0; ni < nb_count; ++ni) {
            uint32_t neighbor = nb_ptr[ni];
            if (neighbor >= ntotal) continue;
            if (visited[neighbor] == visit_mark) continue;
            visited[neighbor] = visit_mark;
            batch_ids[n_new++] = neighbor;
        }

        if (n_new > 0) {
            for (size_t i = 0; i < n_new; ++i) {
                batch_dists[i] = distance::compute_l2_distance(query, base + batch_ids[i] * dim, dim);
            }
        }

        for (size_t i = 0; i < n_new; ++i) {
            float dist = batch_dists[i];

            if (res_size < ef || dist < res_dists[0]) {
                if (cand_size < heap_cap) {
                    cand_dists[cand_size] = dist;
                    cand_ids[cand_size] = static_cast<int32_t>(batch_ids[i]);
                    cand_size++;
                    sift_up_max(cand_size - 1);
                }

                if (res_size < ef + 1) {
                    res_dists[res_size] = dist;
                    res_ids[res_size] = static_cast<int32_t>(batch_ids[i]);
                    res_size++;
                    sift_up_min(res_size - 1);

                    if (res_size > ef) {
                        res_dists[0] = res_dists[res_size - 1];
                        res_ids[0] = res_ids[res_size - 1];
                        res_size--;
                        sift_down_min(0);
                    }
                }
            }
        }
    }

    size_t count = std::min(res_size, ef);
    *out_count = count;

    for (size_t i = 0; i < count; ++i) {
        out_dists[i] = res_dists[i];
        out_ids[i] = res_ids[i];
    }

    for (size_t i = 1; i < count; ++i) {
        float key_d = out_dists[i];
        int32_t key_id = out_ids[i];
        size_t j = i;
        while (j > 0 && out_dists[j - 1] > key_d) {
            out_dists[j] = out_dists[j - 1];
            out_ids[j] = out_ids[j - 1];
            j--;
        }
        out_dists[j] = key_d;
        out_ids[j] = key_id;
    }
}

void IndexHNSW::search_layer_impl_no_blas(const float* query,
                                           size_t ef,
                                           size_t level,
                                           const int32_t* entry_points,
                                           size_t n_entry,
                                           float* out_dists,
                                           int32_t* out_ids,
                                           size_t* out_count,
                                           int32_t* visited,
                                           int32_t visit_mark) const {
    const float* base = data();
    size_t dim = d;

    size_t heap_cap = ef + 2 * M_ + 64;

    float* cand_dists = static_cast<float*>(alloca(heap_cap * sizeof(float)));
    int32_t* cand_ids = static_cast<int32_t*>(alloca(heap_cap * sizeof(int32_t)));
    size_t cand_size = 0;

    float* res_dists = static_cast<float*>(alloca((ef + 1) * sizeof(float)));
    int32_t* res_ids = static_cast<int32_t*>(alloca((ef + 1) * sizeof(int32_t)));
    size_t res_size = 0;

    for (size_t i = 0; i < n_entry; ++i) {
        int32_t ep = entry_points[i];
        if (ep < 0 || (size_t)ep >= ntotal) continue;
        float dist = distance::compute_l2_distance(query, base + ep * dim, dim);
        visited[ep] = visit_mark;

        cand_dists[cand_size] = dist;
        cand_ids[cand_size] = ep;
        cand_size++;

        res_dists[res_size] = dist;
        res_ids[res_size] = ep;
        res_size++;
    }

    auto sift_up_max = [&](size_t idx) {
        while (idx > 0) {
            size_t p = (idx - 1) / 2;
            if (cand_dists[idx] > cand_dists[p]) {
                std::swap(cand_dists[idx], cand_dists[p]);
                std::swap(cand_ids[idx], cand_ids[p]);
                idx = p;
            } else break;
        }
    };

    auto sift_down_max = [&](size_t idx) {
        while (true) {
            size_t largest = idx;
            size_t l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < cand_size && cand_dists[l] > cand_dists[largest]) largest = l;
            if (r < cand_size && cand_dists[r] > cand_dists[largest]) largest = r;
            if (largest != idx) {
                std::swap(cand_dists[idx], cand_dists[largest]);
                std::swap(cand_ids[idx], cand_ids[largest]);
                idx = largest;
            } else break;
        }
    };

    auto sift_up_min = [&](size_t idx) {
        while (idx > 0) {
            size_t p = (idx - 1) / 2;
            if (res_dists[idx] < res_dists[p]) {
                std::swap(res_dists[idx], res_dists[p]);
                std::swap(res_ids[idx], res_ids[p]);
                idx = p;
            } else break;
        }
    };

    auto sift_down_min = [&](size_t idx) {
        while (true) {
            size_t smallest = idx;
            size_t l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < res_size && res_dists[l] < res_dists[smallest]) smallest = l;
            if (r < res_size && res_dists[r] < res_dists[smallest]) smallest = r;
            if (smallest != idx) {
                std::swap(res_dists[idx], res_dists[smallest]);
                std::swap(res_ids[idx], res_ids[smallest]);
                idx = smallest;
            } else break;
        }
    };

    for (size_t i = cand_size; i > 0; --i) sift_up_max(i - 1);
    for (size_t i = res_size; i > 0; --i) sift_up_min(i - 1);

    size_t nb_per_level = (level == 0) ? 2 * M_ : M_;
    size_t nb_offset = level * (2 * M_);

    while (cand_size > 0) {
        if (cand_dists[0] > res_dists[0] && res_size >= ef) {
            break;
        }

        int32_t c = cand_ids[0];
        cand_dists[0] = cand_dists[cand_size - 1];
        cand_ids[0] = cand_ids[cand_size - 1];
        cand_size--;
        if (cand_size > 0) sift_down_max(0);

        if (c < 0 || (size_t)c >= ntotal) continue;

        if ((size_t)c >= neighbors_.size()) continue;
        const auto& nb_vec = neighbors_[c];
        if (nb_offset >= nb_vec.size()) continue;

        size_t nb_end = std::min(nb_offset + nb_per_level, nb_vec.size());

        for (size_t ni = nb_offset; ni < nb_end; ++ni) {
            int32_t neighbor = nb_vec[ni];
            if (neighbor < 0) break;
            if ((size_t)neighbor >= ntotal) continue;
            if (visited[neighbor] == visit_mark) continue;
            visited[neighbor] = visit_mark;

            float dist = distance::compute_l2_distance(query, base + neighbor * dim, dim);

            if (res_size < ef || dist < res_dists[0]) {
                if (cand_size < heap_cap) {
                    cand_dists[cand_size] = dist;
                    cand_ids[cand_size] = neighbor;
                    cand_size++;
                    sift_up_max(cand_size - 1);
                }

                if (res_size < ef + 1) {
                    res_dists[res_size] = dist;
                    res_ids[res_size] = neighbor;
                    res_size++;
                    sift_up_min(res_size - 1);

                    if (res_size > ef) {
                        res_dists[0] = res_dists[res_size - 1];
                        res_ids[0] = res_ids[res_size - 1];
                        res_size--;
                        sift_down_min(0);
                    }
                }
            }
        }
    }

    size_t count = std::min(res_size, ef);
    *out_count = count;

    for (size_t i = 0; i < count; ++i) {
        out_dists[i] = res_dists[i];
        out_ids[i] = res_ids[i];
    }

    for (size_t i = 1; i < count; ++i) {
        float key_d = out_dists[i];
        int32_t key_id = out_ids[i];
        size_t j = i;
        while (j > 0 && out_dists[j - 1] > key_d) {
            out_dists[j] = out_dists[j - 1];
            out_ids[j] = out_ids[j - 1];
            j--;
        }
        out_dists[j] = key_d;
        out_ids[j] = key_id;
    }
}

void IndexHNSW::select_neighbors_heuristic(const float*,
                                            size_t,
                                            const float*,
                                            const int32_t* cand_ids,
                                            size_t n_candidates,
                                            size_t M_max,
                                            int32_t* selected,
                                            size_t* n_selected) const {
    *n_selected = 0;
    if (n_candidates == 0) return;

    size_t take = std::min(M_max, n_candidates);
    for (size_t i = 0; i < take; ++i) {
        selected[i] = cand_ids[i];
    }
    *n_selected = take;
}

void IndexHNSW::insert_node(size_t idx) {
    size_t new_level = random_level();
    levels_[idx] = static_cast<int32_t>(new_level);

    size_t node_levels = std::max(new_level, max_level_) + 1;
    neighbors_[idx].assign(node_levels * (2 * M_), -1);

    if (ntotal == 1) {
        max_level_ = new_level;
        enter_point_ = 0;
        return;
    }

    const float* query = data() + idx * d;
    int32_t cur_ep = static_cast<int32_t>(enter_point_);

    for (size_t l = max_level_; l > new_level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            if ((size_t)cur_ep >= neighbors_.size()) break;
            const auto& nb_vec = neighbors_[cur_ep];
            size_t nb_offset = l * (2 * M_);
            if (nb_offset >= nb_vec.size()) break;

            float cur_dist = distance::compute_l2_distance(query, data() + cur_ep * d, d);
            for (size_t ni = 0; ni < M_; ++ni) {
                if (nb_offset + ni >= nb_vec.size()) break;
                int32_t neighbor = nb_vec[nb_offset + ni];
                if (neighbor < 0) break;
                if ((size_t)neighbor >= ntotal) continue;
                float d_n = distance::compute_l2_distance(query, data() + neighbor * d, d);
                if (d_n < cur_dist) {
                    cur_dist = d_n;
                    cur_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    if (insert_visited_.size() < ntotal) {
        insert_visited_.assign(ntotal, 0);
    }

    size_t buf_size = ef_construction_ + 1;
    float* res_dists = static_cast<float*>(alloca(buf_size * sizeof(float)));
    int32_t* res_ids = static_cast<int32_t*>(alloca(buf_size * sizeof(int32_t)));
    size_t res_count = 0;

    int32_t* sel_ids = static_cast<int32_t*>(alloca((2 * M_) * sizeof(int32_t)));
    size_t sel_count = 0;

    for (size_t l = std::min(new_level, max_level_); l != static_cast<size_t>(-1); --l) {
        size_t M_max = (l == 0) ? 2 * M_ : M_;

        insert_visit_mark_++;
        if (insert_visit_mark_ == 0) {
            insert_visited_.assign(ntotal, 0);
            insert_visit_mark_ = 1;
        }

        search_layer_impl_no_blas(query, ef_construction_, l, &cur_ep, 1,
                                  res_dists, res_ids, &res_count,
                                  insert_visited_.data(), insert_visit_mark_);

        select_neighbors_heuristic(query, idx, res_dists, res_ids,
                                  res_count, M_max, sel_ids, &sel_count);

        int32_t* node_nb = neighbors_[idx].data() + l * (2 * M_);
        for (size_t i = 0; i < sel_count; ++i) {
            int32_t neighbor = sel_ids[i];
            node_nb[i] = neighbor;

            if ((size_t)neighbor >= neighbors_.size()) continue;
            auto& nb_of_neighbor_vec = neighbors_[neighbor];
            size_t nb_offset = l * (2 * M_);
            if (nb_offset >= nb_of_neighbor_vec.size()) {
                nb_of_neighbor_vec.resize(nb_offset + 2 * M_, -1);
            }

            int32_t* nb_of_neighbor = nb_of_neighbor_vec.data() + nb_offset;
            size_t neighbor_max = (l == 0) ? 2 * M_ : M_;

            bool already = false;
            for (size_t j = 0; j < neighbor_max; ++j) {
                if (nb_of_neighbor[j] < 0) break;
                if (nb_of_neighbor[j] == static_cast<int32_t>(idx)) { already = true; break; }
            }
            if (already) continue;

            size_t nb_count = 0;
            for (size_t j = 0; j < 2 * M_; ++j) {
                if (nb_of_neighbor[j] < 0) break;
                nb_count = j + 1;
            }

            if (nb_count < neighbor_max) {
                nb_of_neighbor[nb_count] = static_cast<int32_t>(idx);
            } else {
                const float* neighbor_vec = data() + neighbor * d;
                float d_idx = distance::compute_l2_distance(neighbor_vec, query, d);

                float max_dist = 0.0f;
                size_t max_idx = 0;
                for (size_t j = 0; j < nb_count; ++j) {
                    float d_n = distance::compute_l2_distance(neighbor_vec, data() + nb_of_neighbor[j] * d, d);
                    if (d_n > max_dist) {
                        max_dist = d_n;
                        max_idx = j;
                    }
                }

                if (d_idx < max_dist) {
                    nb_of_neighbor[max_idx] = static_cast<int32_t>(idx);
                }
            }
        }

        if (res_count > 0) {
            cur_ep = res_ids[0];
        }
    }

    if (new_level > max_level_) {
        max_level_ = new_level;
        enter_point_ = idx;
    }
}

void IndexHNSW::add(size_t n, const float* x) {
    compact_ready_ = false;
    size_t old_total = ntotal;
    VectorStorage::add(n, x);

    levels_.resize(ntotal, 0);
    neighbors_.resize(ntotal);

    for (size_t i = 0; i < n; ++i) {
        insert_node(old_total + i);
    }
}

void IndexHNSW::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    if (!compact_ready_) {
        build_compact_neighbors();
        precompute_norms();
    }

    const float* base = data();
    size_t dim = d;
    size_t n_levels = max_level_ + 1;

    if (search_states_.empty()) {
        search_states_.resize(1);
    }
    auto& state = search_states_[0];
    if (state.visited.size() < ntotal) {
        state.visited.assign(ntotal, 0);
        state.visit_mark = 1;
    }

    size_t buf_size = std::max(ef_search_, k) + 1;
    float* res_dists = new float[buf_size];
    int32_t* res_ids = new int32_t[buf_size];

    for (size_t q = 0; q < n; ++q) {
        const float* query = x + q * dim;

        if (ntotal == 0) {
            for (size_t i = 0; i < k; ++i) {
                distances[q * k + i] = 0.0f;
                labels[q * k + i] = 0;
            }
            continue;
        }

        state.visit_mark++;
        if (state.visit_mark == 0) {
            state.visited.assign(ntotal, 0);
            state.visit_mark = 1;
        }

        float query_norm = 0.0f;
        for (size_t i = 0; i < dim; ++i) query_norm += query[i] * query[i];

        int32_t cur_ep = static_cast<int32_t>(enter_point_);

        for (size_t l = max_level_; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                if ((size_t)cur_ep >= ntotal) break;

                const uint32_t* nb_ptr = neighbors_compact_.data() + cur_ep * compact_slot_size_ + compact_level_offsets_[l];
                uint16_t nb_count = neighbor_counts_[cur_ep * n_levels + l];

                float cur_dist = distance::compute_l2_distance(query, base + cur_ep * dim, dim);

                for (uint16_t ni = 0; ni < nb_count; ++ni) {
                    uint32_t neighbor = nb_ptr[ni];
                    if (neighbor >= ntotal) continue;
                    float nb_dist = distance::compute_l2_distance(query, base + neighbor * dim, dim);
                    if (nb_dist < cur_dist) {
                        cur_dist = nb_dist;
                        cur_ep = static_cast<int32_t>(neighbor);
                        changed = true;
                    }
                }
            }
        }

        size_t res_count = 0;

        search_layer_impl(query, query_norm, std::max(ef_search_, k), 0, &cur_ep, 1,
                         res_dists, res_ids, &res_count,
                         state.visited.data(), state.visit_mark);

        size_t take = std::min(k, res_count);
        for (size_t i = 0; i < take; ++i) {
            distances[q * k + i] = res_dists[i];
            labels[q * k + i] = static_cast<size_t>(res_ids[i]);
        }
        for (size_t i = take; i < k; ++i) {
            distances[q * k + i] = 0.0f;
            labels[q * k + i] = 0;
        }
    }

    delete[] res_dists;
    delete[] res_ids;
}

}
}
