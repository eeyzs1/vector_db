#include "ivf.h"
#include <stdexcept>
#include <limits>
#include <cstring>
#include <numeric>
#include <unordered_set>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectordb {
namespace algorithms {

IndexIVF::IndexIVF(size_t dimension, size_t nlist)
    : VectorStorage(dimension), nlist_(nlist), nprobe_(10), layout_built_(false) {
    inverted_lists_.resize(nlist_);
}

void IndexIVF::kmeans_pp_init(size_t n, const float* x, std::mt19937& gen) {
    // Random initialization (matches FAISS approach): pick nlist_ distinct points.
    // Much faster than k-means++ (O(nlist*d) vs O(nlist*n*d)) and the k-means
    // iterations with 5e-3 convergence threshold still produce good centroids.
    std::uniform_int_distribution<size_t> uniform_dist(0, n - 1);

    std::vector<size_t> chosen_indices;
    chosen_indices.reserve(nlist_);
    std::unordered_set<size_t> seen;
    seen.reserve(nlist_ * 2);

    while (chosen_indices.size() < nlist_) {
        size_t idx = uniform_dist(gen);
        if (seen.insert(idx).second) {
            chosen_indices.push_back(idx);
        }
    }

    for (size_t c = 0; c < nlist_; ++c) {
        size_t idx = chosen_indices[c];
        std::copy(x + idx * d, x + (idx + 1) * d, centroids_.data() + c * d);
    }
}

void IndexIVF::kmeans_clustering(size_t n, const float* x, size_t max_iter) {
    centroids_.resize(nlist_ * d);

    // Fixed seed for reproducible centroid initialization (matches FAISS approach).
    std::mt19937 gen(42);

    kmeans_pp_init(n, x, gen);

    std::vector<float> new_centroids(nlist_ * d, 0.0f);
    std::vector<size_t> counts(nlist_, 0);

    // Shared 2D arrays for parallel reduction (avoids serial critical section).
    int num_threads = omp_get_max_threads();
    std::vector<float> all_centroids(num_threads * nlist_ * d, 0.0f);
    std::vector<size_t> all_counts(num_threads * nlist_, 0);

    // Hoist parallel region outside the iteration loop to avoid
    // repeated allocation of thread-private buffers (25x savings).
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* thread_centroids = all_centroids.data() + tid * nlist_ * d;
        size_t* thread_counts = all_counts.data() + tid * nlist_;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            std::fill(thread_centroids, thread_centroids + nlist_ * d, 0.0f);
            std::fill(thread_counts, thread_counts + nlist_, 0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float* vec = x + i * d;
                // Inline assign_to_cluster with batch-4 SIMD for speed
                size_t best_cluster = 0;
                float best_dist = std::numeric_limits<float>::max();

                size_t ci = 0;
                #ifdef __AVX2__
                for (; ci + 3 < nlist_; ci += 4) {
                    const float* c0 = centroids_.data() + ci * d;
                    const float* c1 = centroids_.data() + (ci+1) * d;
                    const float* c2 = centroids_.data() + (ci+2) * d;
                    const float* c3 = centroids_.data() + (ci+3) * d;

                    __m256 sum0 = _mm256_setzero_ps();
                    __m256 sum1 = _mm256_setzero_ps();
                    __m256 sum2 = _mm256_setzero_ps();
                    __m256 sum3 = _mm256_setzero_ps();

                    size_t j = 0;
                    for (; j + 7 < d; j += 8) {
                        __m256 v = _mm256_loadu_ps(vec + j);
                        __m256 d0 = _mm256_sub_ps(v, _mm256_loadu_ps(c0 + j));
                        __m256 d1 = _mm256_sub_ps(v, _mm256_loadu_ps(c1 + j));
                        __m256 d2 = _mm256_sub_ps(v, _mm256_loadu_ps(c2 + j));
                        __m256 d3 = _mm256_sub_ps(v, _mm256_loadu_ps(c3 + j));
                        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
                        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
                        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
                        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
                    }

                    auto hreduce = [](__m256 v) -> float {
                        __m256 sh = _mm256_permute2f128_ps(v, v, 0x21);
                        __m256 sm = _mm256_add_ps(v, sh);
                        sm = _mm256_hadd_ps(sm, sm);
                        sm = _mm256_hadd_ps(sm, sm);
                        return _mm256_cvtss_f32(sm);
                    };

                    float d0f = hreduce(sum0);
                    float d1f = hreduce(sum1);
                    float d2f = hreduce(sum2);
                    float d3f = hreduce(sum3);

                    for (; j < d; ++j) {
                        float vj = vec[j];
                        float diff0 = vj - c0[j]; d0f += diff0 * diff0;
                        float diff1 = vj - c1[j]; d1f += diff1 * diff1;
                        float diff2 = vj - c2[j]; d2f += diff2 * diff2;
                        float diff3 = vj - c3[j]; d3f += diff3 * diff3;
                    }

                    if (d0f < best_dist) { best_dist = d0f; best_cluster = ci; }
                    if (d1f < best_dist) { best_dist = d1f; best_cluster = ci+1; }
                    if (d2f < best_dist) { best_dist = d2f; best_cluster = ci+2; }
                    if (d3f < best_dist) { best_dist = d3f; best_cluster = ci+3; }
                }
                #endif
                for (; ci < nlist_; ++ci) {
                    const float* centroid = centroids_.data() + ci * d;
                    float dist = distance::compute_l2_distance(vec, centroid, d);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = ci;
                    }
                }

                size_t cluster = best_cluster;
                float* sum = thread_centroids + cluster * d;

                for (size_t j = 0; j < d; ++j) {
                    sum[j] += vec[j];
                }
                thread_counts[cluster]++;
            }

            // Parallel reduction: sum all thread-local centroids and counts.
            // Replaces serial #pragma omp critical — much faster with 32 threads.
            #pragma omp for schedule(static)
            for (size_t i = 0; i < nlist_ * d; ++i) {
                float sum = 0.0f;
                for (int t = 0; t < num_threads; ++t) {
                    sum += all_centroids[t * nlist_ * d + i];
                }
                new_centroids[i] = sum;
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < nlist_; ++i) {
                size_t sum = 0;
                for (int t = 0; t < num_threads; ++t) {
                    sum += all_counts[t * nlist_ + i];
                }
                counts[i] = sum;
            }

            // Centroid update (serial, inside parallel region).
            // The implicit barrier at the end of `single` ensures all threads
            // see the updated centroids before the next iteration.
            #pragma omp single
            {
                for (size_t i = 0; i < nlist_; ++i) {
                    if (counts[i] > 0) {
                        float* old_c = centroids_.data() + i * d;
                        float* new_c = new_centroids.data() + i * d;
                        float inv_count = 1.0f / counts[i];
                        for (size_t j = 0; j < d; ++j) {
                            new_c[j] *= inv_count;
                        }
                        std::copy(new_c, new_c + d, old_c);
                    }
                }

                // Empty cluster handling (FAISS split_clusters approach): for
                // each empty cluster, split a donor cluster by copying its
                // centroid and applying a small symmetric perturbation (±EPS).
                // This preserves full nlist capacity and is more stable than
                // random jitter. EPS = 1/1024 (slightly above float16 epsilon).
                constexpr float EPS = 1.0f / 1024.0f;
                for (size_t i = 0; i < nlist_; ++i) {
                    if (counts[i] == 0) {
                        // Find the largest cluster as donor
                        size_t largest_cluster = 0;
                        size_t largest_count = 0;
                        for (size_t k = 0; k < nlist_; ++k) {
                            if (counts[k] > largest_count) {
                                largest_count = counts[k];
                                largest_cluster = k;
                            }
                        }
                        if (largest_count > 1) {
                            // Copy donor centroid to empty cluster
                            std::copy(centroids_.data() + largest_cluster * d,
                                      centroids_.data() + largest_cluster * d + d,
                                      centroids_.data() + i * d);
                            // Small symmetric perturbation
                            float* c_empty = centroids_.data() + i * d;
                            float* c_donor = centroids_.data() + largest_cluster * d;
                            for (size_t j = 0; j < d; ++j) {
                                if (j % 2 == 0) {
                                    c_empty[j] *= 1 + EPS;
                                    c_donor[j] *= 1 - EPS;
                                } else {
                                    c_empty[j] *= 1 - EPS;
                                    c_donor[j] *= 1 + EPS;
                                }
                            }
                            // Split assignment count
                            counts[i] = counts[largest_cluster] / 2;
                            counts[largest_cluster] -= counts[i];
                        }
                    }
                }

                // Reset for next iteration
                std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                // No early stopping: FAISS runs all iterations and gets
                // slightly better centroids. The extra iterations are cheap
                // and improve recall.
            }
            // Implicit barrier from `single` synchronizes all threads.
        }
    }
}

size_t IndexIVF::assign_to_cluster(const float* vec) const {
    size_t best_cluster = 0;
    float best_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < nlist_; ++i) {
        const float* centroid = centroids_.data() + i * d;
        float dist = distance::compute_l2_distance(vec, centroid, d);
        if (dist < best_dist) {
            best_dist = dist;
            best_cluster = i;
        }
    }

    return best_cluster;
}

void IndexIVF::build_cluster_layout() {
    cluster_vector_offsets_.resize(nlist_);
    cluster_vector_sizes_.resize(nlist_);

    size_t total = 0;
    for (size_t i = 0; i < nlist_; ++i) {
        cluster_vector_offsets_[i] = total;
        cluster_vector_sizes_[i] = inverted_lists_[i].size();
        total += cluster_vector_sizes_[i];
    }

    cluster_vectors_.resize(total * d);
    cluster_original_ids_.resize(total);
    cluster_vector_norms_.resize(total);

    for (size_t i = 0; i < nlist_; ++i) {
        size_t offset = cluster_vector_offsets_[i];
        size_t size = cluster_vector_sizes_[i];
        for (size_t j = 0; j < size; ++j) {
            size_t orig_idx = inverted_lists_[i][j];
            const float* vec = data() + orig_idx * d;
            std::copy(vec, vec + d, cluster_vectors_.data() + (offset + j) * d);
            cluster_original_ids_[offset + j] = orig_idx;

            float norm = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                norm += vec[k] * vec[k];
            }
            cluster_vector_norms_[offset + j] = norm;
        }
    }

    layout_built_ = true;
}

void IndexIVF::train(size_t n, const float* x) {
    if (n < nlist_) {
        throw std::invalid_argument("Training data is too small for nlist clusters");
    }
    kmeans_clustering(n, x);
}

void IndexIVF::add(size_t n, const float* x) {
    VectorStorage::add(n, x);

    for (size_t i = 0; i < n; ++i) {
        size_t idx = ntotal - n + i;
        const float* vec = data() + idx * d;
        size_t cluster = assign_to_cluster(vec);
        inverted_lists_[cluster].push_back(idx);
    }

    layout_built_ = false;
}

void IndexIVF::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    if (centroids_.empty()) {
        throw std::runtime_error("Index not trained");
    }

    if (!layout_built_) {
        const_cast<IndexIVF*>(this)->build_cluster_layout();
    }

    size_t nprobe = std::min(nprobe_, nlist_);
    size_t dim = d;

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

    #pragma omp parallel num_threads(num_threads)
    {
        std::vector<std::pair<float, size_t>> cluster_dists(nlist_);
        std::vector<std::pair<float, size_t>> heap;
        heap.reserve(k + 1);

        #pragma omp for schedule(dynamic, 1)
        for (size_t q = 0; q < n; ++q) {
            const float* query = x + q * dim;
            float* q_dists = distances + q * k;
            size_t* q_labels = labels + q * k;

            for (size_t i = 0; i < nlist_; ++i) {
                const float* centroid = centroids_.data() + i * dim;
                float dist = distance::compute_l2_distance(query, centroid, dim);
                cluster_dists[i] = {dist, i};
            }

            std::nth_element(cluster_dists.begin(),
                             cluster_dists.begin() + nprobe,
                             cluster_dists.end());

            heap.clear();

            for (size_t c = 0; c < nprobe; ++c) {
                size_t cluster = cluster_dists[c].second;
                size_t offset = cluster_vector_offsets_[cluster];
                size_t list_size = cluster_vector_sizes_[cluster];

                if (list_size == 0) continue;

                const float* cluster_data = cluster_vectors_.data() + offset * dim;
                const size_t* cluster_ids = cluster_original_ids_.data() + offset;
                const float* cluster_norms = cluster_vector_norms_.data() + offset;

                float cutoff = (heap.size() >= k) ? heap.front().first : std::numeric_limits<float>::max();

                for (size_t j = 0; j < list_size; ++j) {
                    float dist = distance::compute_l2_distance(query, cluster_data + j * dim, dim);

                    if (dist < cutoff) {
                        if (heap.size() < k) {
                            heap.push_back({dist, cluster_ids[j]});
                            std::push_heap(heap.begin(), heap.end());
                            if (heap.size() >= k) cutoff = heap.front().first;
                        } else {
                            std::pop_heap(heap.begin(), heap.end());
                            heap.back() = {dist, cluster_ids[j]};
                            std::push_heap(heap.begin(), heap.end());
                            cutoff = heap.front().first;
                        }
                    }
                }
            }

            std::sort(heap.begin(), heap.end());

            for (size_t i = 0; i < k; ++i) {
                if (i < heap.size()) {
                    q_dists[i] = heap[i].first;
                    q_labels[i] = heap[i].second;
                } else {
                    q_dists[i] = 0.0f;
                    q_labels[i] = 0;
                }
            }
        }
    }
}

}
}
