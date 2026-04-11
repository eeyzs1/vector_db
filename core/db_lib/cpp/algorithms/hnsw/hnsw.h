#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>

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

    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_;

    std::vector<int32_t> insert_visited_;
    int32_t insert_visit_mark_;

    size_t random_level();

    void search_layer(const float* query,
                      std::vector<float>& result_dists,
                      std::vector<int32_t>& result_ids,
                      const int32_t* entry_points,
                      size_t n_entry,
                      size_t ef,
                      size_t level,
                      int32_t* visited,
                      int32_t visit_mark,
                      std::vector<std::pair<float, int32_t>>& candidates,
                      std::vector<std::pair<float, int32_t>>& results) const;

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
