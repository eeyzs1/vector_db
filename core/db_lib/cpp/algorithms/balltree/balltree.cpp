#include "balltree.h"
#include <stdexcept>
#include <limits>

namespace vectordb {
namespace algorithms {

IndexBallTree::IndexBallTree(size_t dimension, size_t leaf_size)
    : VectorStorage(dimension), leaf_size_(leaf_size) {}

std::unique_ptr<BallNode> IndexBallTree::build_tree(const std::vector<size_t>& indices) {
    auto node = std::make_unique<BallNode>();
    node->points = indices;

    if (indices.size() <= leaf_size_) {
        node->center.resize(d);
        std::fill(node->center.begin(), node->center.end(), 0.0f);
        
        for (size_t idx : indices) {
            const float* vec = data() + idx * d;
            for (size_t j = 0; j < d; ++j) {
                node->center[j] += vec[j];
            }
        }
        
        float inv_size = 1.0f / indices.size();
        for (size_t j = 0; j < d; ++j) {
            node->center[j] *= inv_size;
        }
        
        node->radius = 0.0f;
        for (size_t idx : indices) {
            const float* vec = data() + idx * d;
            float dist = distance::compute_l2_distance(vec, node->center.data(), d);
            node->radius = std::max(node->radius, dist);
        }
        
        return node;
    }

    std::vector<float> mean(d, 0.0f);
    for (size_t idx : indices) {
        const float* vec = data() + idx * d;
        for (size_t j = 0; j < d; ++j) {
            mean[j] += vec[j];
        }
    }
    float inv_n = 1.0f / indices.size();
    for (size_t j = 0; j < d; ++j) {
        mean[j] *= inv_n;
    }

    size_t split_dim = 0;
    float max_var = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        float var = 0.0f;
        for (size_t idx : indices) {
            const float* vec = data() + idx * d;
            float diff = vec[j] - mean[j];
            var += diff * diff;
        }
        if (var > max_var) {
            max_var = var;
            split_dim = j;
        }
    }

    std::vector<size_t> sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [this, split_dim](size_t a, size_t b) {
                  const float* va = data() + a * d;
                  const float* vb = data() + b * d;
                  return va[split_dim] < vb[split_dim];
              });

    size_t median = sorted_indices.size() / 2;
    std::vector<size_t> left_indices(sorted_indices.begin(), sorted_indices.begin() + median);
    std::vector<size_t> right_indices(sorted_indices.begin() + median, sorted_indices.end());

    node->left = build_tree(left_indices);
    node->right = build_tree(right_indices);

    node->center.resize(d);
    for (size_t j = 0; j < d; ++j) {
        node->center[j] = (node->left->center[j] + node->right->center[j]) / 2.0f;
    }

    float d1 = distance::compute_l2_distance(node->left->center.data(), node->center.data(), d) + node->left->radius;
    float d2 = distance::compute_l2_distance(node->right->center.data(), node->center.data(), d) + node->right->radius;
    node->radius = std::max(d1, d2);

    return node;
}

void IndexBallTree::search_k_nearest(const float* query, const BallNode* node, 
                                       size_t k, std::priority_queue<std::pair<float, size_t>>& heap) const {
    if (!node) {
        return;
    }

    if (node->points.size() <= leaf_size_) {
        for (size_t idx : node->points) {
            const float* vec = data() + idx * d;
            float dist = distance::compute_l2_distance(query, vec, d);
            
            if (heap.size() < k) {
                heap.push({dist, idx});
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.push({dist, idx});
            }
        }
        return;
    }

    float dist_to_left = distance::compute_l2_distance(query, node->left->center.data(), d);
    float dist_to_right = distance::compute_l2_distance(query, node->right->center.data(), d);

    const BallNode* near_child = (dist_to_left < dist_to_right) ? node->left.get() : node->right.get();
    const BallNode* far_child = (dist_to_left < dist_to_right) ? node->right.get() : node->left.get();
    float dist_to_far = std::min(dist_to_left, dist_to_right);

    search_k_nearest(query, near_child, k, heap);

    if (heap.size() < k || dist_to_far - near_child->radius < heap.top().first) {
        search_k_nearest(query, far_child, k, heap);
    }
}

void IndexBallTree::add(size_t n, const float* x) {
    VectorStorage::add(n, x);

    std::vector<size_t> indices(ntotal);
    for (size_t i = 0; i < ntotal; ++i) {
        indices[i] = i;
    }

    root_ = build_tree(indices);
}

void IndexBallTree::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
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
