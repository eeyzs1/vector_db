#include "kdtree.h"
#include <stdexcept>
#include <limits>

namespace vectordb {
namespace algorithms {

IndexKDTree::IndexKDTree(size_t dimension, size_t leaf_size)
    : VectorStorage(dimension), leaf_size_(leaf_size) {}

std::unique_ptr<KDNode> IndexKDTree::build_tree(size_t* indices_ptr, size_t n, size_t depth) {
    if (n == 0) {
        return nullptr;
    }

    size_t axis = depth % d;

    std::sort(indices_ptr, indices_ptr + n, [this, axis](size_t a, size_t b) {
        const float* va = data() + a * d;
        const float* vb = data() + b * d;
        return va[axis] < vb[axis];
    });

    size_t median_idx = n / 2;
    size_t median_vector_idx = indices_ptr[median_idx];
    const float* median_vec = data() + median_vector_idx * d;

    auto node = std::make_unique<KDNode>(median_vector_idx, axis, median_vec[axis]);

    node->left = build_tree(indices_ptr, median_idx, depth + 1);
    node->right = build_tree(indices_ptr + median_idx + 1, n - median_idx - 1, depth + 1);

    return node;
}

void IndexKDTree::search_k_nearest(const float* query, const KDNode* node, 
                                     size_t k, std::priority_queue<std::pair<float, size_t>>& heap) const {
    if (!node) {
        return;
    }

    const float* vec = data() + node->index * d;
    float dist = distance::compute_l2_distance(query, vec, d);

    if (heap.size() < k) {
        heap.push({dist, node->index});
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.push({dist, node->index});
    }

    float diff = query[node->split_axis] - node->split_value;
    const KDNode* near_subtree = (diff < 0) ? node->left.get() : node->right.get();
    const KDNode* far_subtree = (diff < 0) ? node->right.get() : node->left.get();

    search_k_nearest(query, near_subtree, k, heap);

    if (heap.size() < k || diff * diff < heap.top().first) {
        search_k_nearest(query, far_subtree, k, heap);
    }
}

void IndexKDTree::add(size_t n, const float* x) {
    size_t old_total = ntotal;
    VectorStorage::add(n, x);

    indices_.resize(ntotal);
    for (size_t i = 0; i < ntotal; ++i) {
        indices_[i] = i;
    }

    root_ = build_tree(indices_.data(), ntotal, 0);
}

void IndexKDTree::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    for (size_t q = 0; q < n; ++q) {
        const float* query = x + q * d;

        std::priority_queue<std::pair<float, size_t>> heap;
        search_k_nearest(query, root_.get(), k, heap);

        std::vector<std::pair<float, size_t>> results;
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }

        std::reverse(results.begin(), results.end());

        size_t take = std::min(k, results.size());
        for (size_t i = 0; i < take; ++i) {
            distances[q * k + i] = results[i].first;
            labels[q * k + i] = results[i].second;
        }

        for (size_t i = take; i < k; ++i) {
            distances[q * k + i] = 0.0f;
            labels[q * k + i] = 0;
        }
    }
}

}
}
