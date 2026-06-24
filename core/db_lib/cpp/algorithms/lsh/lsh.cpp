#include "lsh.h"
#include <stdexcept>
#include <limits>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectordb {
namespace algorithms {

IndexLSH::IndexLSH(size_t dimension, size_t num_hash_tables,
                   size_t num_hash_functions, float r)
    : VectorStorage(dimension), num_hash_tables_(num_hash_tables),
      num_hash_functions_(num_hash_functions), r_(r), inv_r_(1.0f / r),
      num_probes_(1), rng_(std::random_device{}()) {
    padded_dim_ = ((d + 15) / 16) * 16;
    hash_key_space_ = size_t(1) << num_hash_functions_;
    hash_tables_.resize(num_hash_tables_);
    for (size_t t = 0; t < num_hash_tables_; ++t) {
        hash_tables_[t].resize(hash_key_space_);
    }
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
        hash = (hash << 1) | (dot > 0.0f ? 1 : 0);
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
        hash = (hash << 1) | (dot > 0.0f ? 1 : 0);
    }
#else
    for (size_t h = 0; h < num_hash_functions_; ++h) {
        const float* weights = table_funcs + h * padded_dim_;
        float dot = hash_biases_[table_idx * num_hash_functions_ + h];
        for (size_t i = 0; i < d; ++i) {
            dot += vec[i] * weights[i];
        }
        hash = (hash << 1) | (dot > 0.0f ? 1 : 0);
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

    // Resize binary codes storage
    binary_codes_.resize(ntotal);

    // Initialize substring tables (4 tables, 256 buckets each)
    size_t num_substrings = 4;
    if (substring_tables_.empty()) {
        substring_tables_.resize(num_substrings);
        for (size_t s = 0; s < num_substrings; ++s) {
            substring_tables_[s].resize(256);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        size_t idx = old_total + i;
        uint32_t code = 0;
        for (size_t t = 0; t < num_hash_tables_; ++t) {
            size_t hash = all_hashes[i * num_hash_tables_ + t];
            hash_tables_[t][hash].push_back(idx);
            code = (code << num_hash_functions_) |
                   static_cast<uint32_t>(hash & ((1u << num_hash_functions_) - 1));
        }
        binary_codes_[idx] = code;
        // Index into substring tables (8-bit substrings)
        for (size_t s = 0; s < num_substrings; ++s) {
            uint8_t substring = static_cast<uint8_t>((code >> (s * 8)) & 0xFF);
            substring_tables_[s][substring].push_back(idx);
        }
    }
}

void IndexLSH::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    if (ntotal == 0) {
        for (size_t q = 0; q < n; ++q) {
            for (size_t i = 0; i < k; ++i) {
                distances[q * k + i] = 0.0f;
                labels[q * k + i] = 0;
            }
        }
        return;
    }

    const float* base_data = data();
    size_t dim = d;
    size_t nbits = num_hash_tables_ * num_hash_functions_;
    size_t num_substrings = std::min(size_t(4), nbits / 8);
    // Refine top-30 candidates (3x FAISS's top-k=10) — balances recall vs L2 cost.
    size_t max_refine = std::min(std::max(k * 3, size_t(30)), ntotal);

    // Limit thread count for small query batches: with 32 threads and only
    // 100 queries, each thread gets ~3 queries. The OpenMP barrier sync and
    // cache contention dominates. Require 8 queries per thread for small N
    // (vectors fit in cache), 4 for large N (more memory bandwidth).
    int max_threads = omp_get_max_threads();
    int num_threads = max_threads;
    if (n > 0) {
        int min_queries_per_thread = (ntotal <= 20000) ? 8 : 4;
        int ideal_threads = (int)((n + min_queries_per_thread - 1) / min_queries_per_thread);
        num_threads = std::min(max_threads, ideal_threads);
        if (num_threads < 1) num_threads = 1;
    }

    // Ensure per-thread seen[] arrays are allocated once and reused.
    // uint8_t seen = 1MB for 1M vectors (fits L2), vs int = 4MB (misses L2).
    if (thread_seen_.size() < (size_t)num_threads) {
        thread_seen_.resize(num_threads);
        thread_seen_version_.resize(num_threads, 0);
    }
    for (int t = 0; t < num_threads; ++t) {
        if (thread_seen_[t].size() < ntotal) {
            thread_seen_[t].assign(ntotal, 0);
        }
    }

    #pragma omp parallel num_threads(num_threads) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        std::vector<uint8_t>& seen = thread_seen_[tid];
        uint8_t& version = thread_seen_version_[tid];
        std::vector<float, AlignedAllocator<float, 64>> thread_padded(padded_dim_);
        // seen[] array with version counter — O(1) dedup, faster than sort+dedup.
        std::vector<std::pair<uint32_t, size_t>> hamming_heap;
        hamming_heap.reserve(1024);
        std::vector<std::pair<float, size_t>> l2_heap;
        l2_heap.reserve(k + 1);

        #pragma omp for schedule(guided)
        for (size_t q = 0; q < n; ++q) {
            const float* query = x + q * d;
            std::memcpy(thread_padded.data(), query, d * sizeof(float));
            for (size_t j = d; j < padded_dim_; ++j) {
                thread_padded[j] = 0.0f;
            }

            // Per-thread version counter (uint8_t). Increment per query.
            // On wrap to 0, reset seen array (1MB memset ~0.1ms, rare).
            version++;
            if (version == 0) {
                std::fill(seen.begin(), seen.end(), 0);
                version = 1;
            }

            // Compute query 32-bit code
            uint32_t query_code = 0;
            for (size_t t = 0; t < num_hash_tables_; ++t) {
                size_t h = hash_vector(thread_padded.data(), t);
                query_code = (query_code << num_hash_functions_) |
                             static_cast<uint32_t>(h & ((1u << num_hash_functions_) - 1));
            }

            // Multi-index hashing with seen[] dedup + running max-heap.
            // By pigeonhole: Hamming dist <= 3 => at least one 8-bit substring matches.
            hamming_heap.clear();
            uint32_t hamming_cutoff = std::numeric_limits<uint32_t>::max();

            for (size_t s = 0; s < num_substrings; ++s) {
                uint8_t substring = static_cast<uint8_t>((query_code >> (s * 8)) & 0xFF);
                const auto& bucket = substring_tables_[s][substring];
                for (size_t idx : bucket) {
                    if (seen[idx] != version) {
                        seen[idx] = version;
                        uint32_t h = __builtin_popcount(query_code ^ binary_codes_[idx]);
                        if (hamming_heap.size() < max_refine) {
                            hamming_heap.push_back({h, idx});
                            std::push_heap(hamming_heap.begin(), hamming_heap.end(),
                                          [](const auto& a, const auto& b) { return a.first < b.first; });
                            if (hamming_heap.size() >= max_refine) {
                                hamming_cutoff = hamming_heap.front().first;
                            }
                        } else if (h < hamming_cutoff) {
                            std::pop_heap(hamming_heap.begin(), hamming_heap.end(),
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
                            hamming_heap.back() = {h, idx};
                            std::push_heap(hamming_heap.begin(), hamming_heap.end(),
                                          [](const auto& a, const auto& b) { return a.first < b.first; });
                            hamming_cutoff = hamming_heap.front().first;
                        }
                    }
                }
            }

            // 1-bit multi-probe fallback if < k candidates
            if (hamming_heap.size() < k) {
                uint8_t substring0 = static_cast<uint8_t>(query_code & 0xFF);
                for (int bit = 0; bit < 8 && hamming_heap.size() < k * 4; ++bit) {
                    uint8_t flipped = substring0 ^ (1u << bit);
                    const auto& bucket = substring_tables_[0][flipped];
                    for (size_t idx : bucket) {
                        if (seen[idx] != version) {
                            seen[idx] = version;
                            uint32_t h = __builtin_popcount(query_code ^ binary_codes_[idx]);
                            hamming_heap.push_back({h, idx});
                        }
                    }
                }
            }

            // Sort by Hamming distance (closest first) for L2 early termination
            std::sort(hamming_heap.begin(), hamming_heap.end());

            // L2 refinement with software prefetching
            l2_heap.clear();
            float l2_cutoff = std::numeric_limits<float>::max();

            size_t heap_size = hamming_heap.size();
            // Prefetch first 2 candidates
            if (heap_size > 0) {
                __builtin_prefetch(base_data + hamming_heap[0].second * dim, 0, 0);
            }
            if (heap_size > 1) {
                __builtin_prefetch(base_data + hamming_heap[1].second * dim, 0, 0);
            }

            for (size_t i = 0; i < heap_size; ++i) {
                size_t v_idx = hamming_heap[i].second;
                const float* vec = base_data + v_idx * dim;

                // Prefetch candidate 2 positions ahead (hide ~300 cycle memory latency)
                if (i + 2 < heap_size) {
                    __builtin_prefetch(base_data + hamming_heap[i + 2].second * dim, 0, 0);
                }

                float dist = distance::compute_l2_distance(query, vec, dim);

                if (l2_heap.size() < k) {
                    l2_heap.push_back({dist, v_idx});
                    std::push_heap(l2_heap.begin(), l2_heap.end());
                    if (l2_heap.size() >= k) l2_cutoff = l2_heap.front().first;
                } else if (dist < l2_cutoff) {
                    std::pop_heap(l2_heap.begin(), l2_heap.end());
                    l2_heap.back() = {dist, v_idx};
                    std::push_heap(l2_heap.begin(), l2_heap.end());
                    l2_cutoff = l2_heap.front().first;
                }
            }

            std::sort(l2_heap.begin(), l2_heap.end());

            size_t n_results = l2_heap.size();
            for (size_t i = 0; i < n_results; ++i) {
                distances[q * k + i] = l2_heap[i].first;
                labels[q * k + i] = l2_heap[i].second;
            }
            for (size_t i = n_results; i < k; ++i) {
                distances[q * k + i] = 0.0f;
                labels[q * k + i] = 0;
            }
        }
    }
}

}
}
