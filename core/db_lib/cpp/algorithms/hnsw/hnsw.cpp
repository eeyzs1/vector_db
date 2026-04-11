#include "hnsw.h"
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace vectordb {
namespace algorithms {

IndexHNSW::IndexHNSW(size_t dimension, size_t M, size_t ef_construction)
    : VectorStorage(dimension), M_(M), ef_construction_(ef_construction),
      ef_search_(ef_construction), max_level_(0), enter_point_(0),
      rng_(42), uniform_(0.0, 1.0), insert_visit_mark_(1) {
    ml_ = 1.0 / std::log(static_cast<double>(M_));
}

size_t IndexHNSW::random_level() {
    float r = uniform_(rng_);
    if (r <= 0.0f) r = 1e-10f;
    double level = -std::log(static_cast<double>(r)) * ml_;
    return static_cast<size_t>(level);
}

void IndexHNSW::search_layer(const float* query,
                              std::vector<float>& result_dists,
                              std::vector<int32_t>& result_ids,
                              const int32_t* entry_points,
                              size_t n_entry,
                              size_t ef,
                              size_t level,
                              int32_t* visited,
                              int32_t visit_mark,
                              std::vector<std::pair<float, int32_t>>& candidates,
                              std::vector<std::pair<float, int32_t>>& results) const {
    result_dists.clear();
    result_ids.clear();
    candidates.clear();
    results.clear();

    for (size_t i = 0; i < n_entry; ++i) {
        int32_t ep = entry_points[i];
        if (ep < 0 || (size_t)ep >= ntotal) continue;
        const float* vec = data() + ep * d;
        float dist = distance::compute_l2_distance(query, vec, d);
        visited[ep] = visit_mark;
        candidates.push_back({dist, ep});
        results.push_back({dist, ep});
    }

    auto cand_cmp = [](const auto& a, const auto& b) { return a.first > b.first; };
    auto res_cmp = [](const auto& a, const auto& b) { return a.first < b.first; };

    std::make_heap(candidates.begin(), candidates.end(), cand_cmp);
    std::make_heap(results.begin(), results.end(), res_cmp);

    while (!candidates.empty()) {
        if (candidates.front().first > results.front().first && results.size() >= ef) {
            break;
        }

        std::pop_heap(candidates.begin(), candidates.end(), cand_cmp);
        auto [dist_c, c] = candidates.back();
        candidates.pop_back();

        if (c < 0 || (size_t)c >= ntotal) continue;
        if ((size_t)c >= neighbors_.size()) continue;

        const auto& nb_vec = neighbors_[c];
        size_t nb_per_level = (level == 0) ? 2 * M_ : M_;
        size_t nb_offset = level * (2 * M_);

        for (size_t ni = 0; ni < nb_per_level; ++ni) {
            if (nb_offset + ni >= nb_vec.size()) break;
            int32_t neighbor = nb_vec[nb_offset + ni];
            if (neighbor < 0) break;
            if ((size_t)neighbor >= ntotal) continue;
            if (visited[neighbor] == visit_mark) continue;
            visited[neighbor] = visit_mark;

            const float* vec = data() + neighbor * d;
            float dist = distance::compute_l2_distance(query, vec, d);

            if (results.size() < ef || dist < results.front().first) {
                candidates.push_back({dist, neighbor});
                std::push_heap(candidates.begin(), candidates.end(), cand_cmp);

                results.push_back({dist, neighbor});
                std::push_heap(results.begin(), results.end(), res_cmp);

                if (results.size() > ef) {
                    std::pop_heap(results.begin(), results.end(), res_cmp);
                    results.pop_back();
                }
            }
        }
    }

    std::sort(results.begin(), results.end());
    result_dists.resize(results.size());
    result_ids.resize(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        result_dists[i] = results[i].first;
        result_ids[i] = results[i].second;
    }
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
            size_t nb_count = M_;
            for (size_t ni = 0; ni < nb_count; ++ni) {
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

    std::vector<float> res_dists;
    std::vector<int32_t> res_ids;
    std::vector<std::pair<float, int32_t>> candidates_buf;
    std::vector<std::pair<float, int32_t>> results_buf;

    for (size_t l = std::min(new_level, max_level_); l != static_cast<size_t>(-1); --l) {
        size_t M_max = (l == 0) ? 2 * M_ : M_;

        insert_visit_mark_++;
        if (insert_visit_mark_ == 0) {
            insert_visited_.assign(ntotal, 0);
            insert_visit_mark_ = 1;
        }

        {
            search_layer(query, res_dists, res_ids, &cur_ep, 1,
                        ef_construction_, l, insert_visited_.data(), insert_visit_mark_,
                        candidates_buf, results_buf);
        }

        size_t take = std::min(M_max, res_ids.size());

        int32_t* node_nb = neighbors_[idx].data() + l * (2 * M_);
        for (size_t i = 0; i < take; ++i) {
            int32_t neighbor = res_ids[i];
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

        if (!res_ids.empty()) {
            cur_ep = res_ids[0];
        }
    }

    if (new_level > max_level_) {
        max_level_ = new_level;
        enter_point_ = idx;
    }
}

void IndexHNSW::add(size_t n, const float* x) {
    size_t old_total = ntotal;
    VectorStorage::add(n, x);

    levels_.resize(ntotal, 0);
    neighbors_.resize(ntotal);

    for (size_t i = 0; i < n; ++i) {
        insert_node(old_total + i);
    }
}

void IndexHNSW::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    #pragma omp parallel
    {
        std::vector<int32_t> visited;
        visited.assign(ntotal, 0);
        int32_t visit_mark = 1;

        std::vector<float> res_dists;
        std::vector<int32_t> res_ids;
        std::vector<std::pair<float, int32_t>> candidates_buf;
        std::vector<std::pair<float, int32_t>> results_buf;

        #pragma omp for schedule(dynamic, 1)
        for (size_t q = 0; q < n; ++q) {
            const float* query = x + q * d;

            if (ntotal == 0) {
                for (size_t i = 0; i < k; ++i) {
                    distances[q * k + i] = 0.0f;
                    labels[q * k + i] = 0;
                }
                continue;
            }

            int32_t cur_ep = static_cast<int32_t>(enter_point_);

            for (size_t l = max_level_; l > 0; --l) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    if ((size_t)cur_ep >= neighbors_.size()) break;
                    const auto& nb_vec = neighbors_[cur_ep];
                    size_t nb_offset = l * (2 * M_);
                    if (nb_offset >= nb_vec.size()) break;

                    float cur_dist = distance::compute_l2_distance(query, data() + cur_ep * d, d);
                    size_t nb_count = M_;
                    for (size_t ni = 0; ni < nb_count; ++ni) {
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

            visit_mark++;
            if (visit_mark == 0) {
                visited.assign(ntotal, 0);
                visit_mark = 1;
            }

            search_layer(query, res_dists, res_ids, &cur_ep, 1,
                        std::max(ef_search_, k), 0, visited.data(), visit_mark,
                        candidates_buf, results_buf);

            size_t take = std::min(k, res_ids.size());
            for (size_t i = 0; i < take; ++i) {
                distances[q * k + i] = res_dists[i];
                labels[q * k + i] = static_cast<size_t>(res_ids[i]);
            }

            for (size_t i = take; i < k; ++i) {
                distances[q * k + i] = 0.0f;
                labels[q * k + i] = 0;
            }
        }
    }
}

}
}
