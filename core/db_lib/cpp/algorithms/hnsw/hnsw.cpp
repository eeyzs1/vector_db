#include "hnsw.h"
#include <stdexcept>
#include <limits>

namespace vectordb {
namespace algorithms {

IndexHNSW::IndexHNSW(size_t dimension, size_t M, size_t ef_construction)
    : VectorStorage(dimension), M_(M), ef_construction_(ef_construction), 
      ef_search_(ef_construction), max_level_(0), enter_point_(0),
      rng_(std::random_device{}()), uniform_(0.0, 1.0) {}

size_t IndexHNSW::random_level() {
    float level = -log(uniform_(rng_)) * M_;
    return static_cast<size_t>(level);
}

void IndexHNSW::search_layer(const float* query, std::vector<size_t>& candidates,
                             size_t level, size_t ef) const {
    std::priority_queue<std::pair<float, size_t>, 
                       std::vector<std::pair<float, size_t>>,
                       std::greater<std::pair<float, size_t>>> candidates_set;
    
    std::unordered_set<size_t> visited;
    
    for (size_t idx : candidates) {
        const float* vec = data() + idx * d;
        float dist = distance::compute_l2_distance(query, vec, d);
        candidates_set.push({dist, idx});
        visited.insert(idx);
    }
    
    while (!candidates_set.empty()) {
        auto [dist_c, c] = candidates_set.top();
        candidates_set.pop();
        
        float dist_f;
        size_t f;
        if (candidates_set.empty()) {
            dist_f = std::numeric_limits<float>::max();
            f = 0;
        } else {
            auto& top = candidates_set.top();
            dist_f = top.first;
            f = top.second;
        }
        
        if (dist_c > dist_f && candidates_set.size() >= ef) {
            break;
        }
        
        for (size_t neighbor : nodes_[c].neighbors[level]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                const float* vec = data() + neighbor * d;
                float dist = distance::compute_l2_distance(query, vec, d);
                
                if (candidates_set.size() < ef || dist < dist_f) {
                    candidates_set.push({dist, neighbor});
                }
            }
        }
    }
    
    candidates.clear();
    while (!candidates_set.empty()) {
        candidates.push_back(candidates_set.top().second);
        candidates_set.pop();
    }
}

void IndexHNSW::connect(size_t a, size_t b, size_t level) {
    nodes_[a].neighbors[level].push_back(b);
    nodes_[b].neighbors[level].push_back(a);
}

void IndexHNSW::select_neighbors(const float* query, 
                                const std::vector<size_t>& candidates,
                                std::vector<size_t>& result,
                                size_t level, bool keep_pruned) const {
    std::vector<std::pair<float, size_t>> dists;
    for (size_t idx : candidates) {
        const float* vec = data() + idx * d;
        float dist = distance::compute_l2_distance(query, vec, d);
        dists.push_back({dist, idx});
    }
    
    std::sort(dists.begin(), dists.end());
    
    size_t M_max = level == 0 ? 2 * M_ : M_;
    size_t take = std::min(M_max, dists.size());
    
    result.clear();
    for (size_t i = 0; i < take; ++i) {
        result.push_back(dists[i].second);
    }
}

void IndexHNSW::insert_node(size_t idx) {
    size_t new_level = random_level();
    
    nodes_.emplace_back(std::max(new_level, max_level_));
    
    std::vector<size_t> entry_points;
    if (ntotal > 0) {
        entry_points.push_back(enter_point_);
        for (size_t l = max_level_; l > new_level; --l) {
            search_layer(data() + idx * d, entry_points, l, 1);
        }
    }
    
    for (size_t l = std::min(new_level, max_level_); l != static_cast<size_t>(-1); --l) {
        search_layer(data() + idx * d, entry_points, l, ef_construction_);
        
        std::vector<size_t> neighbors;
        select_neighbors(data() + idx * d, entry_points, neighbors, l, false);
        
        for (size_t neighbor : neighbors) {
            connect(idx, neighbor, l);
        }
    }
    
    if (new_level > max_level_ || ntotal == 0) {
        enter_point_ = idx;
        max_level_ = std::max(max_level_, new_level);
    }
}

void IndexHNSW::add(size_t n, const float* x) {
    size_t old_total = ntotal;
    VectorStorage::add(n, x);
    
    for (size_t i = 0; i < n; ++i) {
        insert_node(old_total + i);
    }
}

void IndexHNSW::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    for (size_t q = 0; q < n; ++q) {
        const float* query = x + q * d;
        
        std::vector<size_t> candidates;
        if (ntotal > 0) {
            candidates.push_back(enter_point_);
            for (size_t l = max_level_; l > 0; --l) {
                search_layer(query, candidates, l, 1);
            }
            search_layer(query, candidates, 0, std::max(ef_search_, k));
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
