#include "pq.h"
#include <stdexcept>
#include <limits>
#include <unordered_set>

namespace vectordb {
namespace algorithms {

IndexPQ::IndexPQ(size_t dimension, size_t M, size_t nbits)
    : VectorStorage(dimension), M_(M), nbits_(nbits), ksub_(1 << nbits) {
    if (dimension % M != 0) {
        throw std::invalid_argument("Dimension must be divisible by M");
    }
}

void IndexPQ::train_kmeans(size_t m, const float* x, size_t n, float* centroids) {
    size_t dim_sub = d / M_;
    size_t max_iter = 25;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, n - 1);

    std::unordered_set<size_t> selected_indices;
    for (size_t i = 0; i < ksub_; ++i) {
        size_t idx;
        do {
            idx = dist(gen);
        } while (selected_indices.count(idx));
        selected_indices.insert(idx);

        const float* src = x + idx * d + m * dim_sub;
        float* dst = centroids + i * dim_sub;
        std::copy(src, src + dim_sub, dst);
    }

    std::vector<float> new_centroids(ksub_ * dim_sub, 0.0f);
    std::vector<size_t> counts(ksub_, 0);
    std::vector<size_t> assignments(n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < n; ++i) {
            const float* vec = x + i * d + m * dim_sub;
            size_t best_k = 0;
            float best_dist = std::numeric_limits<float>::max();

            for (size_t k = 0; k < ksub_; ++k) {
                const float* centroid = centroids + k * dim_sub;
                float dist = distance::compute_l2_distance(vec, centroid, dim_sub);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }

            assignments[i] = best_k;
            float* sum = new_centroids.data() + best_k * dim_sub;
            for (size_t j = 0; j < dim_sub; ++j) {
                sum[j] += vec[j];
            }
            counts[best_k]++;
        }

        bool converged = true;
        float max_shift = 0.0f;

        for (size_t k = 0; k < ksub_; ++k) {
            if (counts[k] > 0) {
                float* old_c = centroids + k * dim_sub;
                float* new_c = new_centroids.data() + k * dim_sub;
                float inv_count = 1.0f / counts[k];

                for (size_t j = 0; j < dim_sub; ++j) {
                    new_c[j] *= inv_count;
                    float diff = new_c[j] - old_c[j];
                    max_shift = std::max(max_shift, std::abs(diff));
                }

                std::copy(new_c, new_c + dim_sub, old_c);
            }
        }

        if (max_shift < 1e-4) {
            break;
        }
    }
}

void IndexPQ::encode_vector(const float* x, uint8_t* code) const {
    size_t dim_sub = d / M_;

    for (size_t m = 0; m < M_; ++m) {
        const float* vec_sub = x + m * dim_sub;
        const float* centroids = codebooks_.data() + m * ksub_ * dim_sub;

        size_t best_k = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (size_t k = 0; k < ksub_; ++k) {
            const float* centroid = centroids + k * dim_sub;
            float dist = distance::compute_l2_distance(vec_sub, centroid, dim_sub);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        code[m] = static_cast<uint8_t>(best_k);
    }
}

float IndexPQ::compute_distance(const uint8_t* code_a, const uint8_t* code_b) const {
    float dist = 0.0f;
    size_t dim_sub = d / M_;

    for (size_t m = 0; m < M_; ++m) {
        uint8_t k_a = code_a[m];
        uint8_t k_b = code_b[m];
        const float* centroids = codebooks_.data() + m * ksub_ * dim_sub;
        const float* c_a = centroids + k_a * dim_sub;
        const float* c_b = centroids + k_b * dim_sub;
        dist += distance::compute_l2_distance(c_a, c_b, dim_sub);
    }

    return dist;
}

void IndexPQ::train(size_t n, const float* x) {
    size_t dim_sub = d / M_;
    codebooks_.resize(M_ * ksub_ * dim_sub);

    for (size_t m = 0; m < M_; ++m) {
        train_kmeans(m, x, n, codebooks_.data() + m * ksub_ * dim_sub);
    }
}

void IndexPQ::add(size_t n, const float* x) {
    if (codebooks_.empty()) {
        throw std::runtime_error("Index not trained");
    }

    VectorStorage::add(n, x);

    size_t old_size = codes_.size();
    codes_.resize(old_size + n * M_);

    for (size_t i = 0; i < n; ++i) {
        encode_vector(x + i * d, codes_.data() + old_size + i * M_);
    }
}

void IndexPQ::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    std::vector<uint8_t> query_code(M_);

    for (size_t q = 0; q < n; ++q) {
        encode_vector(x + q * d, query_code.data());

        std::vector<std::pair<float, size_t>> results;
        for (size_t i = 0; i < ntotal; ++i) {
            const uint8_t* code = codes_.data() + i * M_;
            float dist = compute_distance(query_code.data(), code);
            results.push_back({dist, i});
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
