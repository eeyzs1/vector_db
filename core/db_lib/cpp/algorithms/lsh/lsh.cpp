#include "lsh.h"
#include <stdexcept>
#include <limits>
#include <cstring>
#include <queue>

namespace vectordb {
namespace algorithms {

IndexLSH::IndexLSH(size_t dimension, size_t num_hash_tables,
                   size_t num_hash_functions, float r)
    : VectorStorage(dimension), num_hash_tables_(num_hash_tables),
      num_hash_functions_(num_hash_functions), r_(r), inv_r_(1.0f / r),
      num_probes_(1), rng_(std::random_device{}()) {
    padded_dim_ = ((d + 15) / 16) * 16;
    hash_tables_.resize(num_hash_tables_);
    generate_hash_functions();
}

void IndexLSH::generate_hash_functions() {
    size_t total_funcs = num_hash_tables_ * num_hash_functions_;
    hash_functions_flat_.resize(total_funcs * padded_dim_, 0.0f);
    hash_biases_.resize(total_funcs);

    std::normal_distribution<float> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<float> uniform_dist(0.0, r_);

    for (size_t t = 0; t < num_hash_tables_; ++t) {
        for (size_t h = 0; h < num_hash_functions_; ++h) {
            size_t func_idx = t * num_hash_functions_ + h;
            float* weights = hash_functions_flat_.data() + func_idx * padded_dim_;
            for (size_t i = 0; i < d; ++i) {
                weights[i] = normal_dist(rng_);
            }
            for (size_t i = d; i < padded_dim_; ++i) {
                weights[i] = 0.0f;
            }
            hash_biases_[func_idx] = uniform_dist(rng_);
        }
    }
}

size_t IndexLSH::hash_vector(const float* vec, size_t table_idx) const {
    size_t hash = 0;
    const float* table_funcs = hash_functions_flat_.data() + table_idx * num_hash_functions_ * padded_dim_;

#ifdef __AVX512F__
    for (size_t h = 0; h < num_hash_functions_; ++h) {
        const float* weights = table_funcs + h * padded_dim_;
        __m512 sum = _mm512_setzero_ps();
        for (size_t i = 0; i < padded_dim_; i += 16) {
            __m512 v = _mm512_loadu_ps(vec + i);
            __m512 w = _mm512_load_ps(weights + i);
            sum = _mm512_fmadd_ps(v, w, sum);
        }
        float dot = _mm512_reduce_add_ps(sum) + hash_biases_[table_idx * num_hash_functions_ + h];
        int bit = static_cast<int>(std::floor(dot * inv_r_));
        hash = (hash << 1) | (bit & 1);
    }
#elif defined(__AVX2__)
    for (size_t h = 0; h < num_hash_functions_; ++h) {
        const float* weights = table_funcs + h * padded_dim_;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 31 < padded_dim_; i += 32) {
            __m256 v0 = _mm256_loadu_ps(vec + i);
            __m256 w0 = _mm256_load_ps(weights + i);
            sum0 = _mm256_fmadd_ps(v0, w0, sum0);
            __m256 v1 = _mm256_loadu_ps(vec + i + 8);
            __m256 w1 = _mm256_load_ps(weights + i + 8);
            sum1 = _mm256_fmadd_ps(v1, w1, sum1);
            __m256 v2 = _mm256_loadu_ps(vec + i + 16);
            __m256 w2 = _mm256_load_ps(weights + i + 16);
            sum0 = _mm256_fmadd_ps(v2, w2, sum0);
            __m256 v3 = _mm256_loadu_ps(vec + i + 24);
            __m256 w3 = _mm256_load_ps(weights + i + 24);
            sum1 = _mm256_fmadd_ps(v3, w3, sum1);
        }
        sum0 = _mm256_add_ps(sum0, sum1);
        for (; i < padded_dim_; i += 8) {
            __m256 v = _mm256_loadu_ps(vec + i);
            __m256 w = _mm256_load_ps(weights + i);
            sum0 = _mm256_fmadd_ps(v, w, sum0);
        }
        __m256 shuffled = _mm256_permute2f128_ps(sum0, sum0, 0x21);
        __m256 summed = _mm256_add_ps(sum0, shuffled);
        summed = _mm256_hadd_ps(summed, summed);
        summed = _mm256_hadd_ps(summed, summed);
        float dot = _mm256_cvtss_f32(summed) + hash_biases_[table_idx * num_hash_functions_ + h];
        int bit = static_cast<int>(std::floor(dot * inv_r_));
        hash = (hash << 1) | (bit & 1);
    }
#else
    for (size_t h = 0; h < num_hash_functions_; ++h) {
        const float* weights = table_funcs + h * padded_dim_;
        float dot = hash_biases_[table_idx * num_hash_functions_ + h];
        for (size_t i = 0; i < d; ++i) {
            dot += vec[i] * weights[i];
        }
        int bit = static_cast<int>(std::floor(dot * inv_r_));
        hash = (hash << 1) | (bit & 1);
    }
#endif
    return hash;
}

void IndexLSH::hash_vector_batch(const float* vec, size_t n, size_t* hashes) const {
    std::vector<float, AlignedAllocator<float, 64>> padded_vec(padded_dim_);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(padded_vec.data(), vec + i * d, d * sizeof(float));
        for (size_t j = d; j < padded_dim_; ++j) {
            padded_vec[j] = 0.0f;
        }
        for (size_t t = 0; t < num_hash_tables_; ++t) {
            hashes[i * num_hash_tables_ + t] = hash_vector(padded_vec.data(), t);
        }
    }
}

void IndexLSH::generate_probe_sequence(size_t hash, size_t num_bits, size_t max_probes,
                                         size_t* probes, size_t& n_probes) const {
    n_probes = 0;
    probes[n_probes++] = hash;

    if (max_probes <= 1 || num_bits == 0) return;

    size_t max_flips = std::min(num_bits, size_t(8));
    size_t total_possible = (1u << max_flips);
    size_t to_generate = std::min(max_probes - 1, total_possible - 1);

    struct ProbeEntry {
        size_t hash_val;
        int hamming_dist;
    };

    std::vector<ProbeEntry> entries;
    entries.reserve(to_generate);

    for (size_t mask = 1; mask < total_possible && entries.size() < to_generate; ++mask) {
        size_t flipped = hash;
        int hd = 0;
        for (size_t bit = 0; bit < max_flips; ++bit) {
            if (mask & (1u << bit)) {
                flipped ^= (1u << (num_bits - 1 - bit));
                ++hd;
            }
        }
        entries.push_back({flipped, hd});
    }

    std::sort(entries.begin(), entries.end(), [](const ProbeEntry& a, const ProbeEntry& b) {
        return a.hamming_dist < b.hamming_dist;
    });

    for (size_t i = 0; i < std::min(to_generate, entries.size()); ++i) {
        probes[n_probes++] = entries[i].hash_val;
    }
}

void IndexLSH::add(size_t n, const float* x) {
    size_t old_total = ntotal;
    VectorStorage::add(n, x);

    std::vector<size_t> all_hashes(n * num_hash_tables_);
    hash_vector_batch(x, n, all_hashes.data());

    for (size_t i = 0; i < n; ++i) {
        size_t idx = old_total + i;
        for (size_t t = 0; t < num_hash_tables_; ++t) {
            size_t hash = all_hashes[i * num_hash_tables_ + t];
            hash_tables_[t][hash].push_back(idx);
        }
    }
}

inline float compute_l2_dist_avx2(const float* query, const float* vec, size_t dim) {
#if defined(__AVX512F__)
    if (dim >= 16) {
        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;
        size_t end = dim - 15;
        while (i < end) {
            __m512 q = _mm512_loadu_ps(query + i);
            __m512 v = _mm512_loadu_ps(vec + i);
            __m512 diff = _mm512_sub_ps(q, v);
            sum = _mm512_fmadd_ps(diff, diff, sum);
            i += 16;
        }
        float dist = _mm512_reduce_add_ps(sum);
        for (; i < dim; ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
        return dist;
    }
#endif
#if defined(__AVX2__)
    if (dim >= 8) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 31 < dim; i += 32) {
            __m256 q0 = _mm256_loadu_ps(query + i);
            __m256 v0 = _mm256_loadu_ps(vec + i);
            __m256 diff0 = _mm256_sub_ps(q0, v0);
            sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
            __m256 q1 = _mm256_loadu_ps(query + i + 8);
            __m256 v1 = _mm256_loadu_ps(vec + i + 8);
            __m256 diff1 = _mm256_sub_ps(q1, v1);
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            __m256 q2 = _mm256_loadu_ps(query + i + 16);
            __m256 v2 = _mm256_loadu_ps(vec + i + 16);
            __m256 diff2 = _mm256_sub_ps(q2, v2);
            sum0 = _mm256_fmadd_ps(diff2, diff2, sum0);
            __m256 q3 = _mm256_loadu_ps(query + i + 24);
            __m256 v3 = _mm256_loadu_ps(vec + i + 24);
            __m256 diff3 = _mm256_sub_ps(q3, v3);
            sum1 = _mm256_fmadd_ps(diff3, diff3, sum1);
        }
        sum0 = _mm256_add_ps(sum0, sum1);
        for (; i + 7 < dim; i += 8) {
            __m256 q = _mm256_loadu_ps(query + i);
            __m256 v = _mm256_loadu_ps(vec + i);
            __m256 diff = _mm256_sub_ps(q, v);
            sum0 = _mm256_fmadd_ps(diff, diff, sum0);
        }
        __m256 shuffled = _mm256_permute2f128_ps(sum0, sum0, 0x21);
        __m256 summed = _mm256_add_ps(sum0, shuffled);
        summed = _mm256_hadd_ps(summed, summed);
        summed = _mm256_hadd_ps(summed, summed);
        float dist = _mm256_cvtss_f32(summed);
        for (; i < dim; ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
        return dist;
    }
#endif
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = query[i] - vec[i];
        dist += diff * diff;
    }
    return dist;
}

struct HeapItem {
    float dist;
    size_t idx;
    bool operator<(const HeapItem& other) const { return dist < other.dist; }
};

void IndexLSH::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    std::vector<size_t> candidates;
    candidates.reserve(1024);

    std::vector<size_t> seen(ntotal, 0);
    size_t query_id = 0;

    std::vector<float, AlignedAllocator<float, 64>> padded_query(padded_dim_);

    size_t max_probes_per_table = num_probes_;
    std::vector<size_t> probe_hashes(max_probes_per_table);

    for (size_t q = 0; q < n; ++q) {
        const float* query = x + q * d;
        std::memcpy(padded_query.data(), query, d * sizeof(float));
        for (size_t j = d; j < padded_dim_; ++j) {
            padded_query[j] = 0.0f;
        }

        candidates.clear();
        ++query_id;
        if (query_id == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            query_id = 1;
        }

        for (size_t t = 0; t < num_hash_tables_; ++t) {
            size_t base_hash = hash_vector(padded_query.data(), t);

            if (max_probes_per_table <= 1) {
                auto it = hash_tables_[t].find(base_hash);
                if (it != hash_tables_[t].end()) {
                    for (size_t idx : it->second) {
                        if (seen[idx] != query_id) {
                            seen[idx] = query_id;
                            candidates.push_back(idx);
                        }
                    }
                }
            } else {
                size_t n_probes = 0;
                generate_probe_sequence(base_hash, num_hash_functions_, max_probes_per_table,
                                       probe_hashes.data(), n_probes);
                for (size_t p = 0; p < n_probes; ++p) {
                    auto it = hash_tables_[t].find(probe_hashes[p]);
                    if (it != hash_tables_[t].end()) {
                        for (size_t idx : it->second) {
                            if (seen[idx] != query_id) {
                                seen[idx] = query_id;
                                candidates.push_back(idx);
                            }
                        }
                    }
                }
            }
        }

        size_t n_candidates = candidates.size();

        if (n_candidates == 0) {
            for (size_t i = 0; i < k; ++i) {
                distances[q * k + i] = 0.0f;
                labels[q * k + i] = 0;
            }
            continue;
        }

        const float* base_data = data();

        std::priority_queue<HeapItem> heap;
        float max_dist = std::numeric_limits<float>::max();

        for (size_t i = 0; i < n_candidates; ++i) {
            size_t idx = candidates[i];
            float dist = compute_l2_dist_avx2(query, base_data + idx * d, d);

            if (heap.size() < k) {
                heap.push({dist, idx});
                if (heap.size() == k) {
                    max_dist = heap.top().dist;
                }
            } else if (dist < max_dist) {
                heap.pop();
                heap.push({dist, idx});
                max_dist = heap.top().dist;
            }
        }

        size_t n_results = heap.size();
        for (size_t i = n_results; i > 0; --i) {
            distances[q * k + i - 1] = heap.top().dist;
            labels[q * k + i - 1] = heap.top().idx;
            heap.pop();
        }
        for (size_t i = n_results; i < k; ++i) {
            distances[q * k + i] = 0.0f;
            labels[q * k + i] = 0;
        }
    }
}

}
}
