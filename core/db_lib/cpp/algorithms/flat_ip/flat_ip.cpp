#include "flat_ip.h"
#include <stdexcept>
#include <thread>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <memory>
#include <fstream>

namespace vectordb {
namespace algorithms {

IndexFlatIP::IndexFlatIP(size_t dimension) : VectorStorage(dimension) {}

void IndexFlatIP::add(size_t n, const float* x) {
    VectorStorage::add(n, x);
}

inline void IndexFlatIP::compute_batch_distances_transposed(const float* query, size_t start_idx, size_t batch_size, float* distances) const {
    #ifdef __AVX2__
    for (size_t vec_offset = 0; vec_offset < batch_size; vec_offset += 8) {
        __m256 dist_vec = _mm256_setzero_ps();
        const float* base_ptr = transposed_data() + start_idx + vec_offset;

        size_t dim_idx = 0;
        for (; dim_idx + 7 < d; dim_idx += 8) {
            __builtin_prefetch(base_ptr + (dim_idx + 8) * ntotal, 0, 3);
            __builtin_prefetch(base_ptr + (dim_idx + 16) * ntotal, 0, 3);

            __m256 q0 = _mm256_set1_ps(query[dim_idx]);
            __m256 v0 = _mm256_loadu_ps(base_ptr + dim_idx * ntotal);
            dist_vec = _mm256_fmadd_ps(q0, v0, dist_vec);

            __m256 q1 = _mm256_set1_ps(query[dim_idx + 1]);
            __m256 v1 = _mm256_loadu_ps(base_ptr + (dim_idx + 1) * ntotal);
            dist_vec = _mm256_fmadd_ps(q1, v1, dist_vec);

            __m256 q2 = _mm256_set1_ps(query[dim_idx + 2]);
            __m256 v2 = _mm256_loadu_ps(base_ptr + (dim_idx + 2) * ntotal);
            dist_vec = _mm256_fmadd_ps(q2, v2, dist_vec);

            __m256 q3 = _mm256_set1_ps(query[dim_idx + 3]);
            __m256 v3 = _mm256_loadu_ps(base_ptr + (dim_idx + 3) * ntotal);
            dist_vec = _mm256_fmadd_ps(q3, v3, dist_vec);

            __m256 q4 = _mm256_set1_ps(query[dim_idx + 4]);
            __m256 v4 = _mm256_loadu_ps(base_ptr + (dim_idx + 4) * ntotal);
            dist_vec = _mm256_fmadd_ps(q4, v4, dist_vec);

            __m256 q5 = _mm256_set1_ps(query[dim_idx + 5]);
            __m256 v5 = _mm256_loadu_ps(base_ptr + (dim_idx + 5) * ntotal);
            dist_vec = _mm256_fmadd_ps(q5, v5, dist_vec);

            __m256 q6 = _mm256_set1_ps(query[dim_idx + 6]);
            __m256 v6 = _mm256_loadu_ps(base_ptr + (dim_idx + 6) * ntotal);
            dist_vec = _mm256_fmadd_ps(q6, v6, dist_vec);

            __m256 q7 = _mm256_set1_ps(query[dim_idx + 7]);
            __m256 v7 = _mm256_loadu_ps(base_ptr + (dim_idx + 7) * ntotal);
            dist_vec = _mm256_fmadd_ps(q7, v7, dist_vec);
        }

        for (; dim_idx < d; ++dim_idx) {
            __m256 q_broadcast = _mm256_set1_ps(query[dim_idx]);
            __m256 v_vals = _mm256_loadu_ps(base_ptr + dim_idx * ntotal);
            dist_vec = _mm256_fmadd_ps(q_broadcast, v_vals, dist_vec);
        }

        _mm256_storeu_ps(distances + vec_offset, dist_vec);
    }
    #else
    for (size_t i = 0; i < batch_size; ++i) {
        float dist = 0.0f;
        for (size_t dim_idx = 0; dim_idx < d; ++dim_idx) {
            dist += query[dim_idx] * transposed_data()[dim_idx * ntotal + start_idx + i];
        }
        distances[i] = dist;
    }
    #endif
}

inline void IndexFlatIP::compute_batch_distances(const float* query, const float* vecs, size_t batch_size, size_t dim, float* distances) const {
    for (size_t i = 0; i < batch_size; ++i) {
        distances[i] = distance::compute_ip_distance(query, vecs + i * dim, dim);
    }
}

inline void IndexFlatIP::insert_into_top_k(float dist, size_t label, float* top_distances, size_t* top_labels, size_t k) const {
    if (dist > top_distances[k-1]) {
        size_t left = 0, right = k;
        while (left < right) {
            size_t mid = (left + right) / 2;
            if (dist > top_distances[mid]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        if (left < k) {
            std::memmove(&top_distances[left + 1], &top_distances[left], (k - left - 1) * sizeof(float));
            std::memmove(&top_labels[left + 1], &top_labels[left], (k - left - 1) * sizeof(size_t));
            top_distances[left] = dist;
            top_labels[left] = label;
        }
    }
}

void IndexFlatIP::search_single(const float* query, size_t k, float* distances, size_t* labels) const {
    std::vector<float> top_distances(k, -std::numeric_limits<float>::max());
    std::vector<size_t> top_labels(k, 0);

    if (is_transposed()) {
        size_t block_size = ntotal < 200000 ? std::min(size_t(16384), ntotal) : std::min(size_t(4096), ntotal);
        std::vector<float> batch_dists(block_size);

        for (size_t block_start = 0; block_start < ntotal; block_start += block_size) {
            size_t block_end = std::min(block_start + block_size, ntotal);
            size_t block_len = block_end - block_start;

            for (size_t i = 0; i < block_len; i += 8) {
                size_t batch = std::min(size_t(8), block_len - i);
                compute_batch_distances_transposed(query, block_start + i, batch, batch_dists.data() + i);
            }

            for (size_t i = 0; i < block_len; ++i) {
                float dist = batch_dists[i];
                insert_into_top_k(dist, block_start + i, top_distances.data(), top_labels.data(), k);
            }
        }
    } else {
        const size_t batch_size = 32;
        const float* vec = data();

        size_t i = 0;
        while (i + 7 < ntotal) {
            __builtin_prefetch(vec + d * 8, 0, 3);
            __builtin_prefetch(vec + d * 16, 0, 3);

            float dist0 = distance::compute_ip_distance(query, vec, d);
            float dist1 = distance::compute_ip_distance(query, vec + d, d);
            float dist2 = distance::compute_ip_distance(query, vec + 2 * d, d);
            float dist3 = distance::compute_ip_distance(query, vec + 3 * d, d);
            float dist4 = distance::compute_ip_distance(query, vec + 4 * d, d);
            float dist5 = distance::compute_ip_distance(query, vec + 5 * d, d);
            float dist6 = distance::compute_ip_distance(query, vec + 6 * d, d);
            float dist7 = distance::compute_ip_distance(query, vec + 7 * d, d);

            insert_into_top_k(dist0, i, top_distances.data(), top_labels.data(), k);
            insert_into_top_k(dist1, i + 1, top_distances.data(), top_labels.data(), k);
            insert_into_top_k(dist2, i + 2, top_distances.data(), top_labels.data(), k);
            insert_into_top_k(dist3, i + 3, top_distances.data(), top_labels.data(), k);
            insert_into_top_k(dist4, i + 4, top_distances.data(), top_labels.data(), k);
            insert_into_top_k(dist5, i + 5, top_distances.data(), top_labels.data(), k);
            insert_into_top_k(dist6, i + 6, top_distances.data(), top_labels.data(), k);
            insert_into_top_k(dist7, i + 7, top_distances.data(), top_labels.data(), k);

            vec += 8 * d;
            i += 8;
        }

        for (; i < ntotal; ++i) {
            float dist = distance::compute_ip_distance(query, vec, d);
            insert_into_top_k(dist, i, top_distances.data(), top_labels.data(), k);
            vec += d;
        }
    }

    for (size_t j = 0; j < k; ++j) {
        distances[j] = top_distances[j];
        labels[j] = top_labels[j];
    }
}

void IndexFlatIP::search_parallel(const float* query, size_t k, float* distances, size_t* labels) const {
    for (size_t i = 0; i < k; ++i) {
        distances[i] = -std::numeric_limits<float>::max();
        labels[i] = 0;
    }

    size_t num_threads = std::thread::hardware_concurrency();

    if (ntotal > 500000) {
        num_threads = std::min(num_threads, size_t(64));
    } else if (ntotal > 100000) {
        num_threads = std::min(num_threads, size_t(32));
    } else if (ntotal > 10000) {
        num_threads = std::min(num_threads, size_t(16));
    } else {
        num_threads = std::min(num_threads, size_t(4));
    }

    if (d > 256) {
        num_threads = std::max(size_t(1), num_threads / 4);
    } else if (d > 128) {
        num_threads = std::max(size_t(1), num_threads / 2);
    }

    num_threads = std::max(size_t(1), num_threads);

    size_t min_vectors_per_thread = 1000;
    size_t max_threads = std::min(num_threads, ntotal / min_vectors_per_thread + 1);
    max_threads = std::max(size_t(1), max_threads);

    size_t vectors_per_thread = ntotal / max_threads;
    size_t remainder = ntotal % max_threads;

    std::vector<std::vector<float>> thread_distances(max_threads, std::vector<float>(k, -std::numeric_limits<float>::max()));
    std::vector<std::vector<size_t>> thread_labels(max_threads, std::vector<size_t>(k, 0));

    std::vector<std::thread> threads;
    threads.reserve(max_threads);

    size_t current_start = 0;
    for (size_t t = 0; t < max_threads; ++t) {
        size_t thread_vectors = vectors_per_thread + (t < remainder ? 1 : 0);
        size_t start = current_start;
        size_t end = current_start + thread_vectors;
        current_start = end;

        threads.emplace_back([this, query, start, end, k, &thread_distances, &thread_labels, t]() {
            __builtin_prefetch(query, 0, 0);
            __builtin_prefetch(query + 64, 0, 0);
            __builtin_prefetch(query + 128, 0, 0);
            __builtin_prefetch(query + 192, 0, 0);

            std::vector<float> top_distances(k, -std::numeric_limits<float>::max());
            std::vector<size_t> top_labels(k, 0);

            const float* vec = data() + start * d;
            for (size_t i = start; i < end; ++i) {
                float dist = distance::compute_ip_distance(query, vec, d);

                if (dist > top_distances[k-1]) {
                    size_t idx = 0;
                    while (idx < k && dist <= top_distances[idx]) idx++;
                    if (idx < k) {
                        for (size_t l = k - 1; l > idx; --l) {
                            top_distances[l] = top_distances[l - 1];
                            top_labels[l] = top_labels[l - 1];
                        }
                        top_distances[idx] = dist;
                        top_labels[idx] = i;
                    }
                }
                vec += d;
            }

            for (size_t i = 0; i < k; ++i) {
                thread_distances[t][i] = top_distances[i];
                thread_labels[t][i] = top_labels[i];
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (size_t t = 0; t < max_threads; ++t) {
        for (size_t i = 0; i < k; ++i) {
            float dist = thread_distances[t][i];
            size_t label = thread_labels[t][i];

            if (dist > distances[k-1]) {
                size_t idx = 0;
                while (idx < k && dist <= distances[idx]) idx++;

                if (idx < k) {
                    for (size_t l = k - 1; l > idx; --l) {
                        distances[l] = distances[l - 1];
                        labels[l] = labels[l - 1];
                    }
                    distances[idx] = dist;
                    labels[idx] = label;
                }
            }
        }
    }
}

void IndexFlatIP::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    if (d == 0) {
        throw std::invalid_argument("Dimension not set");
    }

    if (ntotal == 0) {
        for (size_t i = 0; i < n * k; ++i) {
            distances[i] = 0.0f;
            labels[i] = 0;
        }
        return;
    }

    size_t num_threads = std::thread::hardware_concurrency();

    if (ntotal > 500000 && n >= 10) {
        num_threads = std::min(num_threads, size_t(32));
    } else if (ntotal > 100000 && n >= 10) {
        num_threads = std::min(num_threads, size_t(16));
    } else if (n < 4) {
        num_threads = std::min(num_threads, size_t(8));
    }

    if (d > 256) {
        num_threads = std::max(size_t(1), num_threads / 2);
    }

    num_threads = std::max(size_t(1), num_threads);

    if (num_threads > 1) {
        if (n == 1) {
            const float* query = x;
            float* dist_ptr = distances;
            size_t* label_ptr = labels;
            search_parallel(query, k, dist_ptr, label_ptr);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(num_threads);

            size_t queries_per_thread = n / num_threads;
            size_t remainder = n % num_threads;

            size_t current_start = 0;
            for (size_t t = 0; t < num_threads; ++t) {
                size_t thread_queries = queries_per_thread + (t < remainder ? 1 : 0);
                size_t start = current_start;
                size_t end = current_start + thread_queries;
                current_start = end;

                threads.emplace_back([this, start, end, x, k, distances, labels]() {
                    for (size_t i = start; i < end; ++i) {
                        const float* query = x + i * d;
                        float* dist_ptr = distances + i * k;
                        size_t* label_ptr = labels + i * k;
                        this->search_single(query, k, dist_ptr, label_ptr);
                    }
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            const float* query = x + i * d;
            float* dist_ptr = distances + i * k;
            size_t* label_ptr = labels + i * k;
            search_single(query, k, dist_ptr, label_ptr);
        }
    }
}

void IndexFlatIP::save_to_file(const std::string& path) const {
    auto bytes = save_to_bytes();
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!file) {
        throw std::runtime_error("Failed to write to file: " + path);
    }
}

void IndexFlatIP::load_from_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> bytes(file_size);
    file.read(reinterpret_cast<char*>(bytes.data()), file_size);
    if (!file) {
        throw std::runtime_error("Failed to read from file: " + path);
    }
    
    load_from_bytes(bytes.data(), bytes.size());
}

} // namespace algorithms
} // namespace vectordb
