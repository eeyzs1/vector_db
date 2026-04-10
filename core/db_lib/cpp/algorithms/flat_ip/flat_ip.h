#pragma once

#include "../../core/vectordb_core.h"
#include <queue>
#include <utility>
#include <algorithm>
#include <span>
#include <thread>
#include <atomic>

namespace vectordb {
namespace algorithms {

class IndexFlatIP : public VectorStorage {
private:
    inline void compute_batch_distances_transposed(const float* query, size_t start_idx, size_t batch_size, float* distances) const;
    inline void compute_batch_distances(const float* query, const float* vecs, size_t batch_size, size_t dim, float* distances) const;
    inline void insert_into_top_k(float dist, size_t label, float* top_distances, size_t* top_labels, size_t k) const;
    void search_single(const float* query, size_t k, float* distances, size_t* labels) const;
    void search_parallel(const float* query, size_t k, float* distances, size_t* labels) const;

public:
    IndexFlatIP(size_t dimension);
    void add(size_t n, const float* x);
    void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const;
    
    std::vector<uint8_t> save() const { return VectorStorage::save_to_bytes(); }
    void load(const uint8_t* bytes, size_t length) { VectorStorage::load_from_bytes(bytes, length); }
    
    void save_to_file(const std::string& path) const;
    void load_from_file(const std::string& path);
};

} // namespace algorithms
} // namespace vectordb
