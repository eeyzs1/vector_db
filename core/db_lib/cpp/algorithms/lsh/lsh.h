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
    size_t hash_key_space_;  // 2^num_hash_functions_
    std::vector<float, AlignedAllocator<float, 64>> hash_functions_flat_;
    std::vector<float> hash_biases_;
    // Direct-index hash tables: hash_tables_[t][hash] = vector of IDs.
    std::vector<std::vector<std::vector<size_t>>> hash_tables_;
    // Binary codes for Hamming distance pre-filtering (like FAISS).
    // Each vector gets a num_hash_tables_*num_hash_functions_-bit code.
    std::vector<uint32_t> binary_codes_;
    // Multi-index hashing: split 32-bit code into 4 substrings of 8 bits.
    // For Hamming dist <= 3, at least one substring matches (pigeonhole).
    // O(1) lookup per substring, ~100 candidates total. Much faster than full scan.
    std::vector<std::vector<std::vector<size_t>>> substring_tables_;
    std::mt19937 rng_;
    // Per-thread seen[] arrays for dedup. uint8_t = 1MB for 1M vectors (fits L2),
    // vs int = 4MB (misses L2). Allocated once and reused across search calls
    // via per-thread version counter — avoids page-fault churn per call.
    mutable std::vector<std::vector<uint8_t>> thread_seen_;
    mutable std::vector<uint8_t> thread_seen_version_;

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
