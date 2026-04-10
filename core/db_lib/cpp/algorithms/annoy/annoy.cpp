#include "annoy.h"
#include <stdexcept>
#include <limits>

namespace vectordb {
namespace algorithms {

IndexAnnoy::IndexAnnoy(size_t dimension, size_t n_trees)
    : VectorStorage(dimension), n_trees_(n_trees), rng_(std::random_device{}()) {}

std::unique_ptr<AnnoyNode> IndexAnnoy::build_tree(const std::vector<size_t>& indices) {
    auto node = std::make_unique<AnnoyNode>();

    if (indices.size() <= 1) {
        node->index = indices.empty() ? 0 : indices[0];
        return node;
    }

    std::uniform_int_distribution<size_t> idx_dist(0, indices.size() - 1);
    size_t i1 = indices[idx_dist(rng_)];
    size_t i2 = indices[idx_dist(rng_)];
    
    while (i1 == i2) {
        i2 = indices[idx_dist(rng_)];
    }

    const float* v1 = data() + i1 * d;
    const float* v2 = data() + i2 * d;

    node->hyperplane_normal.resize(d);
    for (size_t j = 0; j < d; ++j) {
        node->hyperplane_normal[j] = v2[j] - v1[j];
    }

    float norm = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        norm += node->hyperplane_normal[j] * node->hyperplane_normal[j];
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-6f) {
        for (size_t j = 0; j < d; ++j) {
            node->hyperplane_normal[j] /= norm;
        }
    }

    node->hyperplane_offset = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        node->hyperplane_offset += node->hyperplane_normal[j] * (v1[j] + v2[j]) / 2.0f;
    }

    std::vector<size_t> left_indices, right_indices;
    for (size_t idx : indices) {
        const float* v = data() + idx * d;
        float dot = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            dot += node->hyperplane_normal[j] * v[j];
        }
        
        if (dot < node->hyperplane_offset) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    if (left_indices.empty()) {
        left_indices.swap(right_indices);
    }
    if (right_indices.empty()) {
        node->index = left_indices[0];
        return node;
    }

    node->left = build_tree(left_indices);
    node->right = build_tree(right_indices);

    return node;
}

void IndexAnnoy::get_candidates(const float* query, const AnnoyNode* node, 
                                  std::unordered_set<size_t>& candidates) const {
    if (!node) {
        return;
    }

    if (!node->left && !node->right) {
        candidates.insert(node->index);
        return;
    }

    float dot = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        dot += node->hyperplane_normal[j] * query[j];
    }

    if (dot < node->hyperplane_offset) {
        get_candidates(query, node->left.get(), candidates);
        get_candidates(query, node->right.get(), candidates);
    } else {
        get_candidates(query, node->right.get(), candidates);
        get_candidates(query, node->left.get(), candidates);
    }
}

void IndexAnnoy::add(size_t n, const float* x) {
    VectorStorage::add(n, x);

    trees_.clear();
    trees_.reserve(n_trees_);

    std::vector<size_t> indices(ntotal);
    for (size_t i = 0; i < ntotal; ++i) {
        indices[i] = i;
    }

    for (size_t t = 0; t < n_trees_; ++t) {
        trees_.push_back(build_tree(indices));
    }
}

void IndexAnnoy::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    for (size_t q = 0; q < n; ++q) {
        const float* query = x + q * d;

        std::unordered_set<size_t> candidates;
        for (const auto& tree : trees_) {
            get_candidates(query, tree.get(), candidates);
        }

        std::vector<std::pair<float, size_t>> results;
        for (size_t idx : candidates) {
            const float* vec = data() + idx * d;
            float dist = distance::compute_l2_distance(query, vec, d);
            results.push_back({dist, idx});
        }

        std::sort(results.begin(), results.end());

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
