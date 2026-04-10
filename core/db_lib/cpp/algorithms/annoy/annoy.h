#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <memory>

namespace vectordb {
namespace algorithms {

struct AnnoyNode {
    size_t index;
    std::vector<float> hyperplane_normal;
    float hyperplane_offset;
    std::unique_ptr<AnnoyNode> left;
    std::unique_ptr<AnnoyNode> right;

    AnnoyNode() : left(nullptr), right(nullptr) {}
};

class IndexAnnoy : public VectorStorage {
private:
    size_t n_trees_;
    std::vector<std::unique_ptr<AnnoyNode>> trees_;
    std::mt19937 rng_;

    std::unique_ptr<AnnoyNode> build_tree(const std::vector<size_t>& indices);
    void get_candidates(const float* query, const AnnoyNode* node, std::unordered_set<size_t>& candidates) const;

public:
    IndexAnnoy(size_t dimension, size_t n_trees = 10);

    void add(size_t n, const float* x);

    void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const;

    size_t get_n_trees() const { return n_trees_; }
};

}
}
