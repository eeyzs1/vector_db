#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <atomic>

namespace vectordb {
namespace algorithms {

class IndexHNSW : public VectorStorage {
private:
    size_t M_;
    size_t ef_construction_;
    size_t ef_search_;
    size_t max_level_;
    size_t enter_point_;
    double ml_;

    std::vector<int32_t> levels_;
    std::vector<std::vector<int32_t>> neighbors_;

    mutable std::vector<uint32_t> neighbors_compact_;
    mutable std::vector<uint16_t> neighbor_counts_;
    mutable size_t compact_slot_size_;
    mutable size_t compact_level_offsets_[32];
    mutable bool compact_ready_;

    mutable std::vector<float> norms_;

    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_;

    std::vector<int32_t> insert_visited_;
    int32_t insert_visit_mark_;

    struct SearchState {
        std::vector<int32_t> visited;
        int32_t visit_mark;
    };

    mutable std::vector<SearchState> search_states_;

    size_t random_level();

    void build_compact_neighbors() const;
    void precompute_norms() const;

    void search_layer_impl(const float* query,
                           float query_norm,
                           size_t ef,
                           size_t level,
                           const int32_t* entry_points,
                           size_t n_entry,
                           float* out_dists,
                           int32_t* out_ids,
                           size_t* out_count,
                           int32_t* visited,
                           int32_t visit_mark) const;

    void search_layer_impl_no_blas(const float* query,
                                   size_t ef,
                                   size_t level,
                                   const int32_t* entry_points,
                                   size_t n_entry,
                                   float* out_dists,
                                   int32_t* out_ids,
                                   size_t* out_count,
                                   int32_t* visited,
                                   int32_t visit_mark) const;

    void select_neighbors_heuristic(const float* query,
                                    size_t idx,
                                    const float* cand_dists,
                                    const int32_t* cand_ids,
                                    size_t n_candidates,
                                    size_t M_max,
                                    int32_t* selected,
                                    size_t* n_selected) const;

    void insert_node(size_t idx);

public:
    IndexHNSW(size_t dimension, size_t M = 16,
              size_t ef_construction = 200);

    void add(size_t n, const float* x);

    void set_ef_search(size_t ef) { ef_search_ = ef; }

    void search(size_t n, const float* x, size_t k,
               float* distances, size_t* labels) const;

    size_t get_M() const { return M_; }
    size_t get_ef_construction() const { return ef_construction_; }
    size_t get_ef_search() const { return ef_search_; }
};

}
}
