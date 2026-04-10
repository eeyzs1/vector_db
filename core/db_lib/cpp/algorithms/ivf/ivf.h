#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>
#include <utility>

namespace vectordb {
namespace algorithms {

class IndexIVF : public VectorStorage {
private:
    size_t nlist_;
    size_t nprobe_;
    std::vector<float> centroids_;
    std::vector<std::vector<size_t>> inverted_lists_;

    std::vector<float> cluster_vectors_;
    std::vector<size_t> cluster_vector_offsets_;
    std::vector<size_t> cluster_vector_sizes_;
    std::vector<size_t> cluster_original_ids_;
    bool layout_built_;

    void kmeans_pp_init(size_t n, const float* x, std::mt19937& gen);
    void kmeans_clustering(size_t n, const float* x, size_t max_iter = 25);
    size_t assign_to_cluster(const float* vec) const;
    void build_cluster_layout();

public:
    IndexIVF(size_t dimension, size_t nlist = 100);

    void train(size_t n, const float* x);

    void add(size_t n, const float* x);

    void set_nprobe(size_t nprobe) { nprobe_ = nprobe; }

    void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const;

    size_t get_nlist() const { return nlist_; }
    size_t get_nprobe() const { return nprobe_; }
};

}
}
