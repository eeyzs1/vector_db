#pragma once

#include "../../core/vectordb_core.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace vectordb {
namespace algorithms {

class IndexPQ : public VectorStorage {
private:
    size_t M_;
    size_t nbits_;
    size_t ksub_;
    std::vector<float> codebooks_;
    std::vector<uint8_t> codes_;

    void train_kmeans(size_t m, const float* x, size_t n, float* centroids);
    void encode_vector(const float* x, uint8_t* code) const;
    float compute_distance(const uint8_t* code_a, const uint8_t* code_b) const;

public:
    IndexPQ(size_t dimension, size_t M = 8, size_t nbits = 8);

    void train(size_t n, const float* x);

    void add(size_t n, const float* x);

    void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const;

    size_t get_M() const { return M_; }
    size_t get_nbits() const { return nbits_; }
};

}
}
