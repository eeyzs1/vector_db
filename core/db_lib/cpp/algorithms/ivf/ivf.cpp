#include "ivf.h"
#include <stdexcept>
#include <limits>
#include <cstring>
#include <numeric>

namespace vectordb {
namespace algorithms {

IndexIVF::IndexIVF(size_t dimension, size_t nlist)
    : VectorStorage(dimension), nlist_(nlist), nprobe_(10), layout_built_(false) {
    inverted_lists_.resize(nlist_);
}

void IndexIVF::kmeans_pp_init(size_t n, const float* x, std::mt19937& gen) {
    std::uniform_int_distribution<size_t> uniform_dist(0, n - 1);
    size_t first_idx = uniform_dist(gen);
    std::copy(x + first_idx * d, x + (first_idx + 1) * d, centroids_.data());

    std::vector<float> min_dists(n, std::numeric_limits<float>::max());

    for (size_t i = 0; i < n; ++i) {
        const float* vec = x + i * d;
        float dist = distance::compute_l2_distance(vec, centroids_.data(), d);
        min_dists[i] = dist;
    }

    for (size_t c = 1; c < nlist_; ++c) {
        float total_dist = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            total_dist += min_dists[i];
        }

        if (total_dist < 1e-10f) {
            for (size_t j = c; j < nlist_; ++j) {
                size_t idx = uniform_dist(gen);
                std::copy(x + idx * d, x + (idx + 1) * d, centroids_.data() + j * d);
            }
            break;
        }

        std::uniform_real_distribution<float> real_dist(0.0f, total_dist);
        float threshold = real_dist(gen);
        float cumulative = 0.0f;
        size_t chosen = 0;
        for (size_t i = 0; i < n; ++i) {
            cumulative += min_dists[i];
            if (cumulative >= threshold) {
                chosen = i;
                break;
            }
        }

        std::copy(x + chosen * d, x + (chosen + 1) * d, centroids_.data() + c * d);

        const float* new_centroid = centroids_.data() + c * d;
        for (size_t i = 0; i < n; ++i) {
            const float* vec = x + i * d;
            float dist = distance::compute_l2_distance(vec, new_centroid, d);
            if (dist < min_dists[i]) {
                min_dists[i] = dist;
            }
        }
    }
}

void IndexIVF::kmeans_clustering(size_t n, const float* x, size_t max_iter) {
    centroids_.resize(nlist_ * d);

    std::random_device rd;
    std::mt19937 gen(rd());

    kmeans_pp_init(n, x, gen);

    std::vector<float> new_centroids(nlist_ * d, 0.0f);
    std::vector<size_t> counts(nlist_, 0);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < n; ++i) {
            const float* vec = x + i * d;
            size_t cluster = assign_to_cluster(vec);
            float* sum = new_centroids.data() + cluster * d;

            for (size_t j = 0; j < d; ++j) {
                sum[j] += vec[j];
            }
            counts[cluster]++;
        }

        float max_shift = 0.0f;

        for (size_t i = 0; i < nlist_; ++i) {
            if (counts[i] > 0) {
                float* old_c = centroids_.data() + i * d;
                float* new_c = new_centroids.data() + i * d;
                float inv_count = 1.0f / counts[i];

                for (size_t j = 0; j < d; ++j) {
                    new_c[j] *= inv_count;
                    float diff = new_c[j] - old_c[j];
                    max_shift = std::max(max_shift, std::abs(diff));
                }

                std::copy(new_c, new_c + d, old_c);
            } else {
                size_t largest_cluster = 0;
                size_t largest_count = 0;
                for (size_t k = 0; k < nlist_; ++k) {
                    if (counts[k] > largest_count) {
                        largest_count = counts[k];
                        largest_cluster = k;
                    }
                }

                if (largest_count > 1) {
                    std::uniform_real_distribution<float> perturb(0.01f, 0.1f);
                    const float* src = centroids_.data() + largest_cluster * d;
                    float* dst = centroids_.data() + i * d;
                    for (size_t j = 0; j < d; ++j) {
                        dst[j] = src[j] + perturb(gen);
                    }
                }
            }
        }

        if (max_shift < 1e-6f) {
            break;
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

    for (size_t i = 0; i < nlist_; ++i) {
        size_t offset = cluster_vector_offsets_[i];
        size_t size = cluster_vector_sizes_[i];
        for (size_t j = 0; j < size; ++j) {
            size_t orig_idx = inverted_lists_[i][j];
            std::copy(data() + orig_idx * d,
                      data() + (orig_idx + 1) * d,
                      cluster_vectors_.data() + (offset + j) * d);
            cluster_original_ids_[offset + j] = orig_idx;
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
    size_t num_threads = 1;

    if (n > 10) {
        num_threads = std::min(std::thread::hardware_concurrency(), static_cast<unsigned int>(n));
        if (num_threads < 1) num_threads = 1;
    }

    if (num_threads <= 1 || n <= 10) {
        for (size_t q = 0; q < n; ++q) {
            const float* query = x + q * d;
            float* q_dists = distances + q * k;
            size_t* q_labels = labels + q * k;

            for (size_t i = 0; i < k; ++i) {
                q_dists[i] = std::numeric_limits<float>::max();
                q_labels[i] = 0;
            }

            std::vector<std::pair<float, size_t>> cluster_dists(nlist_);
            for (size_t i = 0; i < nlist_; ++i) {
                const float* centroid = centroids_.data() + i * d;
                float dist = distance::compute_l2_distance(query, centroid, d);
                cluster_dists[i] = {dist, i};
            }

            std::partial_sort(cluster_dists.begin(),
                              cluster_dists.begin() + nprobe,
                              cluster_dists.end());

            for (size_t c = 0; c < nprobe; ++c) {
                size_t cluster = cluster_dists[c].second;
                size_t offset = cluster_vector_offsets_[cluster];
                size_t list_size = cluster_vector_sizes_[cluster];

                const float* cluster_data = cluster_vectors_.data() + offset * d;
                const size_t* cluster_ids = cluster_original_ids_.data() + offset;

                for (size_t j = 0; j < list_size; ++j) {
                    const float* vec = cluster_data + j * d;
                    float dist = distance::compute_l2_distance(query, vec, d);

                    if (dist < q_dists[k - 1]) {
                        size_t pos = k - 1;
                        while (pos > 0 && dist < q_dists[pos - 1]) {
                            q_dists[pos] = q_dists[pos - 1];
                            q_labels[pos] = q_labels[pos - 1];
                            pos--;
                        }
                        q_dists[pos] = dist;
                        q_labels[pos] = cluster_ids[j];
                    }
                }
            }

            for (size_t i = 0; i < k; ++i) {
                if (q_dists[i] == std::numeric_limits<float>::max()) {
                    q_dists[i] = 0.0f;
                    q_labels[i] = 0;
                }
            }
        }
    } else {
        std::vector<std::thread> threads;
        size_t queries_per_thread = (n + num_threads - 1) / num_threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * queries_per_thread;
            size_t end = std::min(start + queries_per_thread, n);
            if (start >= end) break;

            threads.emplace_back([this, start, end, x, k, nprobe, distances, labels]() {
                for (size_t q = start; q < end; ++q) {
                    const float* query = x + q * d;
                    float* q_dists = distances + q * k;
                    size_t* q_labels = labels + q * k;

                    for (size_t i = 0; i < k; ++i) {
                        q_dists[i] = std::numeric_limits<float>::max();
                        q_labels[i] = 0;
                    }

                    std::vector<std::pair<float, size_t>> cluster_dists(nlist_);
                    for (size_t i = 0; i < nlist_; ++i) {
                        const float* centroid = centroids_.data() + i * d;
                        float dist = distance::compute_l2_distance(query, centroid, d);
                        cluster_dists[i] = {dist, i};
                    }

                    std::partial_sort(cluster_dists.begin(),
                                      cluster_dists.begin() + nprobe,
                                      cluster_dists.end());

                    for (size_t c = 0; c < nprobe; ++c) {
                        size_t cluster = cluster_dists[c].second;
                        size_t offset = cluster_vector_offsets_[cluster];
                        size_t list_size = cluster_vector_sizes_[cluster];

                        const float* cluster_data = cluster_vectors_.data() + offset * d;
                        const size_t* cluster_ids = cluster_original_ids_.data() + offset;

                        for (size_t j = 0; j < list_size; ++j) {
                            const float* vec = cluster_data + j * d;
                            float dist = distance::compute_l2_distance(query, vec, d);

                            if (dist < q_dists[k - 1]) {
                                size_t pos = k - 1;
                                while (pos > 0 && dist < q_dists[pos - 1]) {
                                    q_dists[pos] = q_dists[pos - 1];
                                    q_labels[pos] = q_labels[pos - 1];
                                    pos--;
                                }
                                q_dists[pos] = dist;
                                q_labels[pos] = cluster_ids[j];
                            }
                        }
                    }

                    for (size_t i = 0; i < k; ++i) {
                        if (q_dists[i] == std::numeric_limits<float>::max()) {
                            q_dists[i] = 0.0f;
                            q_labels[i] = 0;
                        }
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
}

}
}
