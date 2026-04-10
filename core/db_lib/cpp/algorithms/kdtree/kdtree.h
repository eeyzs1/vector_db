#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <memory>
#include <queue>
#include <cmath>
#include <algorithm>

namespace vectordb {
namespace algorithms {

struct KDNode {
    size_t index;
    size_t split_axis;
    float split_value;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(size_t idx, size_t axis, float val) 
        : index(idx), split_axis(axis), split_value(val), left(nullptr), right(nullptr) {}
};

class IndexKDTree : public VectorStorage {
private:
    std::unique_ptr<KDNode> root_;
    std::vector<size_t> indices_;
    size_t leaf_size_;

    std::unique_ptr<KDNode> build_tree(size_t* indices_ptr, size_t n, size_t depth);
    void search_k_nearest(const float* query, const KDNode* node, 
                          size_t k, std::priority_queue<std::pair<float, size_t>>& heap) const;

public:
    IndexKDTree(size_t dimension, size_t leaf_size = 40);

    void add(size_t n, const float* x);

    void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const;

    size_t get_leaf_size() const { return leaf_size_; }
};

}
}
