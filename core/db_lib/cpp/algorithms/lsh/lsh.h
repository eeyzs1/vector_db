#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace vectordb {
namespace algorithms {

class IndexLSH : public VectorStorage {
private:
    size_t num_hash_tables_;
    size_t num_hash_functions_;
    float r_;
    float inv_r_;
    size_t padded_dim_;
    size_t num_probes_;
    std::vector<float, AlignedAllocator<float, 64>> hash_functions_flat_;
    std::vector<float> hash_biases_;
    std::vector<std::unordered_map<size_t, std::vector<size_t>>> hash_tables_;
    std::mt19937 rng_;

    void generate_hash_functions();
    size_t hash_vector(const float* vec, size_t table_idx) const;
    void hash_vector_batch(const float* vec, size_t n, size_t* hashes) const;
    void generate_probe_sequence(size_t hash, size_t num_bits, size_t max_probes, size_t* probes, size_t& n_probes) const;

public:
    IndexLSH(size_t dimension, size_t num_hash_tables = 8,
             size_t num_hash_functions = 4, float r = 1.0);

    void add(size_t n, const float* x);

    void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const;

    void set_num_probes(size_t n) { num_probes_ = n; }
    size_t get_num_probes() const { return num_probes_; }
    size_t get_num_hash_tables() const { return num_hash_tables_; }
    size_t get_num_hash_functions() const { return num_hash_functions_; }
};

}
}
