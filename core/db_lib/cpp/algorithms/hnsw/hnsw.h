#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <cmath>
#include <algorithm>

namespace vectordb {
namespace algorithms {

struct HNSWNode {
    std::vector<std::vector<size_t>> neighbors;
    HNSWNode(size_t max_level) : neighbors(max_level + 1) {}
};

class IndexHNSW : public VectorStorage {
private:
    size_t M_;
    size_t ef_construction_;
    size_t ef_search_;
    size_t max_level_;
    size_t enter_point_;
    std::vector<HNSWNode> nodes_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_;

    size_t random_level();
    void search_layer(const float* query, std::vector<size_t>& candidates,
                     size_t level, size_t ef) const;
    void connect(size_t a, size_t b, size_t level);
    void select_neighbors(const float* query, 
                        const std::vector<size_t>& candidates,
                        std::vector<size_t>& result,
                        size_t level, bool keep_pruned) const;
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
