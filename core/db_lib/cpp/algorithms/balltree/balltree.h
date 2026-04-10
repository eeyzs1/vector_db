#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <memory>
#include <queue>
#include <cmath>
#include <algorithm>

namespace vectordb {
namespace algorithms {

struct BallNode {
    size_t index;
    std::vector<size_t> points;
    std::vector<float> center;
    float radius;
    std::unique_ptr<BallNode> left;
    std::unique_ptr<BallNode> right;

    BallNode() : radius(0.0f), left(nullptr), right(nullptr) {}
};

class IndexBallTree : public VectorStorage {
private:
    std::unique_ptr<BallNode> root_;
    size_t leaf_size_;

    std::unique_ptr<BallNode> build_tree(const std::vector<size_t>& indices);
    void search_k_nearest(const float* query, const BallNode* node, 
                          size_t k, std::priority_queue<std::pair<float, size_t>>& heap) const;

public:
    IndexBallTree(size_t dimension, size_t leaf_size = 40);

    void add(size_t n, const float* x);

    void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const;

    size_t get_leaf_size() const { return leaf_size_; }
};

}
}
