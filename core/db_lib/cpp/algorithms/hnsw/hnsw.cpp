#include "hnsw.h"
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectordb {
namespace algorithms {

IndexHNSW::IndexHNSW(size_t dimension, size_t M, size_t ef_construction)
    : VectorStorage(dimension), M_(M), ef_construction_(ef_construction),
      ef_search_(ef_construction), max_level_(0), enter_point_(0),
      prune_headroom_(0.0f), n_inserted_(0),
      compact_slot_size_(0), compact_ready_(false),
      rng_(42), uniform_(0.0, 1.0), rng_perm_(789), insert_visit_mark_(1) {
    ml_ = 1.0 / std::log(static_cast<double>(M_));
    memset(compact_level_offsets_, 0, sizeof(compact_level_offsets_));

    // Precompute level assignment probabilities (matches FAISS set_default_probas).
    double levelMult = ml_;
    for (int level = 0;; level++) {
        double proba = std::exp(-level / levelMult) * (1.0 - std::exp(-1.0 / levelMult));
        if (proba < 1e-9) {
            break;
        }
        assign_probas_.push_back(proba);
    }
}

size_t IndexHNSW::random_level() {
    float r = uniform_(rng_);
    if (r <= 0.0f) r = 1e-10f;
    double level = -std::log(static_cast<double>(r)) * ml_;
    return static_cast<size_t>(level);
}

void IndexHNSW::build_compact_neighbors() const {
    if (ntotal == 0) {
        compact_ready_ = true;
        return;
    }

    size_t n_levels = max_level_ + 1;
    compact_level_offsets_[0] = 0;
    for (size_t l = 0; l < n_levels; ++l) {
        size_t slot = (l == 0) ? 2 * M_ : M_;
        compact_level_offsets_[l + 1] = compact_level_offsets_[l] + slot;
    }
    compact_slot_size_ = compact_level_offsets_[n_levels];

    neighbors_compact_.assign(ntotal * compact_slot_size_, 0xFFFFFFFF);
    neighbor_counts_.assign(ntotal * n_levels, 0);

    for (size_t i = 0; i < ntotal; ++i) {
        const auto& nb_vec = neighbors_[i];
        uint32_t* dst = neighbors_compact_.data() + i * compact_slot_size_;
        uint16_t* counts = neighbor_counts_.data() + i * n_levels;

        for (size_t l = 0; l < n_levels; ++l) {
            size_t nb_offset = l * (2 * M_);
            size_t slot = (l == 0) ? 2 * M_ : M_;
            uint32_t* dst_level = dst + compact_level_offsets_[l];
            uint16_t count = 0;

            for (size_t ni = 0; ni < slot && ni + nb_offset < nb_vec.size(); ++ni) {
                int32_t nb = nb_vec[nb_offset + ni];
                if (nb < 0) break;
                dst_level[ni] = static_cast<uint32_t>(nb);
                count++;
            }
            counts[l] = count;
        }
    }

    compact_ready_ = true;
}

void IndexHNSW::precompute_norms() const {
    const float* base = data();
    size_t dim = d;
    norms_.resize(ntotal);
    for (size_t i = 0; i < ntotal; ++i) {
        norms_[i] = distance::compute_l2_distance(base + i * dim, base + i * dim, dim);
    }
}

void IndexHNSW::search_layer_impl(const float* query,
                                   float query_norm,
                                   size_t ef,
                                   size_t level,
                                   const int32_t* entry_points,
                                   size_t n_entry,
                                   float* out_dists,
                                   int32_t* out_ids,
                                   size_t* out_count,
                                   uint8_t* visited,
                                   int32_t visit_mark) const {
    const float* base = data();
    size_t dim = d;
    size_t n_levels = max_level_ + 1;
    size_t level_offset = compact_level_offsets_[level];

    // Generation-counter visited tracking (FAISS approach): no clearing needed.
    // A node is visited if visited[id] == visit_mark. Mark by writing visit_mark.
    auto is_visited = [&](uint32_t id) -> bool {
        return visited[id] == visit_mark;
    };
    auto mark_visited = [&](uint32_t id) {
        visited[id] = visit_mark;
    };

    // Candidate max-heap (FAISS MinimaxHeap approach):
    // - Root = worst candidate (largest distance) → O(1) early rejection
    // - push: if full and new >= worst, reject (O(1)). Else O(log n).
    // - pop_min: O(n) linear scan with soft deletion (id=-1).
    //   Sequential scan is cache-friendly; with ~256 entries it's ~16
    //   AVX-512 comparisons. Soft deletion preserves heap structure.
    constexpr size_t CAND_CAP = 256;
    float* cand_dists = static_cast<float*>(alloca(CAND_CAP * sizeof(float)));
    int32_t* cand_ids = static_cast<int32_t*>(alloca(CAND_CAP * sizeof(int32_t)));
    int cand_k = 0;       // heap size (including deleted)
    int cand_valid = 0;   // number of valid entries

    float* res_dists = static_cast<float*>(alloca((ef + 1) * sizeof(float)));
    int32_t* res_ids = static_cast<int32_t*>(alloca((ef + 1) * sizeof(int32_t)));
    size_t res_size = 0;

    // Max-heap sift: larger values move up, smaller down.
    auto cand_sift_up = [&](int idx) {
        while (idx > 0) {
            int p = (idx - 1) / 2;
            if (cand_dists[idx] > cand_dists[p]) {
                std::swap(cand_dists[idx], cand_dists[p]);
                std::swap(cand_ids[idx], cand_ids[p]);
                idx = p;
            } else break;
        }
    };
    auto cand_sift_down = [&](int idx) {
        while (true) {
            int largest = idx;
            int l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < cand_k && cand_dists[l] > cand_dists[largest]) largest = l;
            if (r < cand_k && cand_dists[r] > cand_dists[largest]) largest = r;
            if (largest != idx) {
                std::swap(cand_dists[idx], cand_dists[largest]);
                std::swap(cand_ids[idx], cand_ids[largest]);
                idx = largest;
            } else break;
        }
    };

    auto res_sift_up = [&](size_t idx) {
        while (idx > 0) {
            size_t p = (idx - 1) / 2;
            if (res_dists[idx] > res_dists[p]) {
                std::swap(res_dists[idx], res_dists[p]);
                std::swap(res_ids[idx], res_ids[p]);
                idx = p;
            } else break;
        }
    };
    auto res_sift_down = [&](size_t idx) {
        while (true) {
            size_t largest = idx;
            size_t l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < res_size && res_dists[l] > res_dists[largest]) largest = l;
            if (r < res_size && res_dists[r] > res_dists[largest]) largest = r;
            if (largest != idx) {
                std::swap(res_dists[idx], res_dists[largest]);
                std::swap(res_ids[idx], res_ids[largest]);
                idx = largest;
            } else break;
        }
    };

    auto add_to_results = [&](int32_t id, float dist) {
        if (res_size < ef) {
            res_dists[res_size] = dist;
            res_ids[res_size] = id;
            res_size++;
            res_sift_up(res_size - 1);
        } else if (dist < res_dists[0]) {
            res_dists[0] = dist;
            res_ids[0] = id;
            res_sift_down(0);
        }
    };

    // Max-heap push with early rejection:
    // If heap full and new value >= worst (root), reject in O(1).
    // Otherwise, pop worst and push new value.
    auto cand_push = [&](int32_t id, float dist) {
        if (cand_k < (int)CAND_CAP) {
            // Heap not full yet — just push
            cand_dists[cand_k] = dist;
            cand_ids[cand_k] = id;
            cand_k++;
            cand_valid++;
            cand_sift_up(cand_k - 1);
        } else if (dist < cand_dists[0]) {
            // Heap full but new candidate is better than worst — replace
            if (cand_ids[0] != -1) cand_valid--;
            cand_dists[0] = dist;
            cand_ids[0] = id;
            cand_sift_down(0);
            cand_valid++;
        }
        // else: reject (worse than worst candidate)
    };

    // pop_min: O(n) linear scan for minimum (skipping deleted entries).
    // Soft deletion: mark id=-1, don't modify heap structure.
    auto cand_pop_min = [&](int32_t& out_id, float& out_dist) -> bool {
        int min_idx = -1;
        float min_val = std::numeric_limits<float>::max();
        for (int i = 0; i < cand_k; ++i) {
            if (cand_ids[i] != -1 && cand_dists[i] < min_val) {
                min_val = cand_dists[i];
                min_idx = i;
            }
        }
        if (min_idx == -1) return false;
        out_id = cand_ids[min_idx];
        out_dist = cand_dists[min_idx];
        cand_ids[min_idx] = -1;  // soft delete
        cand_valid--;
        return true;
    };

    // Add entry points
    for (size_t i = 0; i < n_entry; ++i) {
        int32_t ep = entry_points[i];
        if (ep < 0 || (size_t)ep >= ntotal) continue;
        float dist = distance::compute_l2_distance(query, base + ep * dim, dim);
        mark_visited(ep);
        cand_push(ep, dist);
        add_to_results(ep, dist);
    }

    uint32_t* batch_ids = static_cast<uint32_t*>(alloca((2 * M_) * sizeof(uint32_t)));
    float* batch_dists = static_cast<float*>(alloca((2 * M_) * sizeof(float)));

    while (cand_valid > 0) {
        // Pop closest candidate (O(n) linear scan with soft deletion).
        int32_t c;
        float c_dist;
        cand_pop_min(c, c_dist);

        // Stopping condition: if result set is full and the popped candidate
        // (closest remaining) is worse than the worst result, stop.
        if (res_size >= ef && c_dist > res_dists[0]) {
            break;
        }

        if (c < 0 || (size_t)c >= ntotal) continue;

        const uint32_t* nb_ptr = neighbors_compact_.data() + c * compact_slot_size_ + level_offset;
        uint16_t nb_count = neighbor_counts_[c * n_levels + level];

        // Collect unvisited neighbors (no prefetching here — done in batch loop).
        size_t n_new = 0;
        for (uint16_t ni = 0; ni < nb_count; ++ni) {
            uint32_t neighbor = nb_ptr[ni];
            if (neighbor >= ntotal) continue;
            if (is_visited(neighbor)) continue;
            mark_visited(neighbor);
            batch_ids[n_new++] = neighbor;
        }

        // Software-pipelined batch-4 distance computation:
        // Prefetch next batch while computing current batch.
        // Only prefetch first 2 cache lines (128 bytes) — hardware prefetcher
        // handles the rest for sequential access patterns.
        auto prefetch_vec = [&](uint32_t id) {
            const char* p = reinterpret_cast<const char*>(base + id * dim);
            __builtin_prefetch(p, 0, 3);
            __builtin_prefetch(p + 64, 0, 3);
        };

        // Prefetch first 2 batches (8 vectors)
        for (size_t b = 0; b < 8 && b < n_new; ++b) {
            prefetch_vec(batch_ids[b]);
        }

        size_t i = 0;
        for (; i + 3 < n_new; i += 4) {
            // Prefetch batch 2 steps ahead (gives ~400ns lead time for RAM access)
            size_t next_batch = i + 8;
            for (size_t b = next_batch; b < next_batch + 4 && b < n_new; ++b) {
                prefetch_vec(batch_ids[b]);
            }

            const float* v0 = base + batch_ids[i] * dim;
            const float* v1 = base + batch_ids[i+1] * dim;
            const float* v2 = base + batch_ids[i+2] * dim;
            const float* v3 = base + batch_ids[i+3] * dim;

#if defined(__AVX512F__)
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            size_t j = 0;
            for (; j + 15 < dim; j += 16) {
                __m512 q = _mm512_loadu_ps(query + j);
                __m512 d0v = _mm512_sub_ps(q, _mm512_loadu_ps(v0 + j));
                __m512 d1v = _mm512_sub_ps(q, _mm512_loadu_ps(v1 + j));
                __m512 d2v = _mm512_sub_ps(q, _mm512_loadu_ps(v2 + j));
                __m512 d3v = _mm512_sub_ps(q, _mm512_loadu_ps(v3 + j));
                sum0 = _mm512_fmadd_ps(d0v, d0v, sum0);
                sum1 = _mm512_fmadd_ps(d1v, d1v, sum1);
                sum2 = _mm512_fmadd_ps(d2v, d2v, sum2);
                sum3 = _mm512_fmadd_ps(d3v, d3v, sum3);
            }

            float d0f = _mm512_reduce_add_ps(sum0);
            float d1f = _mm512_reduce_add_ps(sum1);
            float d2f = _mm512_reduce_add_ps(sum2);
            float d3f = _mm512_reduce_add_ps(sum3);

            for (; j < dim; ++j) {
                float qj = query[j];
                float diff0 = qj - v0[j]; d0f += diff0 * diff0;
                float diff1 = qj - v1[j]; d1f += diff1 * diff1;
                float diff2 = qj - v2[j]; d2f += diff2 * diff2;
                float diff3 = qj - v3[j]; d3f += diff3 * diff3;
            }

            batch_dists[i] = d0f;
            batch_dists[i+1] = d1f;
            batch_dists[i+2] = d2f;
            batch_dists[i+3] = d3f;
#elif defined(__AVX2__)
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();

            size_t j = 0;
            for (; j + 7 < dim; j += 8) {
                __m256 q = _mm256_loadu_ps(query + j);
                __m256 d0v = _mm256_sub_ps(q, _mm256_loadu_ps(v0 + j));
                __m256 d1v = _mm256_sub_ps(q, _mm256_loadu_ps(v1 + j));
                __m256 d2v = _mm256_sub_ps(q, _mm256_loadu_ps(v2 + j));
                __m256 d3v = _mm256_sub_ps(q, _mm256_loadu_ps(v3 + j));
                sum0 = _mm256_fmadd_ps(d0v, d0v, sum0);
                sum1 = _mm256_fmadd_ps(d1v, d1v, sum1);
                sum2 = _mm256_fmadd_ps(d2v, d2v, sum2);
                sum3 = _mm256_fmadd_ps(d3v, d3v, sum3);
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

            for (; j < dim; ++j) {
                float qj = query[j];
                float diff0 = qj - v0[j]; d0f += diff0 * diff0;
                float diff1 = qj - v1[j]; d1f += diff1 * diff1;
                float diff2 = qj - v2[j]; d2f += diff2 * diff2;
                float diff3 = qj - v3[j]; d3f += diff3 * diff3;
            }

            batch_dists[i] = d0f;
            batch_dists[i+1] = d1f;
            batch_dists[i+2] = d2f;
            batch_dists[i+3] = d3f;
#else
            batch_dists[i] = distance::compute_l2_distance(query, v0, dim);
            batch_dists[i+1] = distance::compute_l2_distance(query, v1, dim);
            batch_dists[i+2] = distance::compute_l2_distance(query, v2, dim);
            batch_dists[i+3] = distance::compute_l2_distance(query, v3, dim);
#endif
        }
        for (; i < n_new; ++i) {
            batch_dists[i] = distance::compute_l2_distance(query, base + batch_ids[i] * dim, dim);
        }

        // Only add to results and candidates if it could improve the result set.
        // This keeps the candidate heap small and ensures the stopping condition
        // triggers correctly.
        float threshold = (res_size >= ef) ? res_dists[0] : std::numeric_limits<float>::max();
        for (size_t bi = 0; bi < n_new; ++bi) {
            float dist = batch_dists[bi];
            int32_t id = static_cast<int32_t>(batch_ids[bi]);

            if (dist < threshold) {
                add_to_results(id, dist);
                cand_push(id, dist);
                if (res_size >= ef) threshold = res_dists[0];
            }
        }
    }

    size_t count = std::min(res_size, ef);
    *out_count = count;

    for (size_t i = 0; i < count; ++i) {
        out_dists[i] = res_dists[i];
        out_ids[i] = res_ids[i];
    }

    // Insertion sort the output (small arrays, insertion sort is fast)
    for (size_t i = 1; i < count; ++i) {
        float key_d = out_dists[i];
        int32_t key_id = out_ids[i];
        size_t j = i;
        while (j > 0 && out_dists[j - 1] > key_d) {
            out_dists[j] = out_dists[j - 1];
            out_ids[j] = out_ids[j - 1];
            j--;
        }
        out_dists[j] = key_d;
        out_ids[j] = key_id;
    }
}

void IndexHNSW::search_layer_impl_no_blas(const float* query,
                                           size_t ef,
                                           size_t level,
                                           const int32_t* entry_points,
                                           size_t n_entry,
                                           float* out_dists,
                                           int32_t* out_ids,
                                           size_t* out_count,
                                           uint8_t* visited,
                                           int32_t visit_mark) const {
    const float* base = data();
    size_t dim = d;

    // Fixed-size stack buffer for candidates (16KB, fits in L1 cache).
    // The candidate set rarely exceeds 2048 entries with the threshold check.
    // Skipping candidates when full is extremely rare and doesn't affect recall.
    constexpr size_t CAND_CAP = 2048;
    float cand_dists[CAND_CAP];
    int32_t cand_ids[CAND_CAP];
    size_t cand_size = 0;

    // Stack-allocated result buffers (bounded by ef+1, no reallocation).
    float* res_dists = static_cast<float*>(alloca((ef + 1) * sizeof(float)));
    int32_t* res_ids = static_cast<int32_t*>(alloca((ef + 1) * sizeof(int32_t)));
    size_t res_size = 0;

    for (size_t i = 0; i < n_entry; ++i) {
        int32_t ep = entry_points[i];
        if (ep < 0 || (size_t)ep >= ntotal) continue;
        float dist = distance::compute_l2_distance(query, base + ep * dim, dim);
        visited[ep] = visit_mark;

        if (cand_size < CAND_CAP) {
            cand_dists[cand_size] = dist;
            cand_ids[cand_size] = ep;
            cand_size++;
        }

        res_dists[res_size] = dist;
        res_ids[res_size] = ep;
        res_size++;
    }

    auto sift_up_max = [&](size_t idx) {
        while (idx > 0) {
            size_t p = (idx - 1) / 2;
            if (res_dists[idx] > res_dists[p]) {
                std::swap(res_dists[idx], res_dists[p]);
                std::swap(res_ids[idx], res_ids[p]);
                idx = p;
            } else break;
        }
    };

    auto sift_down_max = [&](size_t idx) {
        while (true) {
            size_t largest = idx;
            size_t l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < res_size && res_dists[l] > res_dists[largest]) largest = l;
            if (r < res_size && res_dists[r] > res_dists[largest]) largest = r;
            if (largest != idx) {
                std::swap(res_dists[idx], res_dists[largest]);
                std::swap(res_ids[idx], res_ids[largest]);
                idx = largest;
            } else break;
        }
    };

    auto sift_up_min = [&](size_t idx) {
        while (idx > 0) {
            size_t p = (idx - 1) / 2;
            if (cand_dists[idx] < cand_dists[p]) {
                std::swap(cand_dists[idx], cand_dists[p]);
                std::swap(cand_ids[idx], cand_ids[p]);
                idx = p;
            } else break;
        }
    };

    auto sift_down_min = [&](size_t idx) {
        while (true) {
            size_t smallest = idx;
            size_t l = 2 * idx + 1, r = 2 * idx + 2;
            if (l < cand_size && cand_dists[l] < cand_dists[smallest]) smallest = l;
            if (r < cand_size && cand_dists[r] < cand_dists[smallest]) smallest = r;
            if (smallest != idx) {
                std::swap(cand_dists[idx], cand_dists[smallest]);
                std::swap(cand_ids[idx], cand_ids[smallest]);
                idx = smallest;
            } else break;
        }
    };

    for (size_t i = cand_size; i > 0; --i) sift_up_min(i - 1);
    for (size_t i = res_size; i > 0; --i) sift_up_max(i - 1);

    size_t nb_per_level = (level == 0) ? 2 * M_ : M_;
    size_t nb_offset = level * (2 * M_);

    // Batch buffers for distance computation (max 2*M neighbors per level)
    uint32_t batch_ids_buf[2 * M_];
    float batch_dists_buf[2 * M_];

    while (cand_size > 0) {
        // Construction-time search: match FAISS's stopping condition exactly.
        // FAISS breaks when the closest candidate is farther than the farthest
        // result, with NO res_size >= ef check. When res_size < ef, continuing
        // to explore far candidates pollutes the result set with bad neighbors
        // that hurt graph quality and recall.
        if (cand_dists[0] > res_dists[0]) {
            break;
        }

        int32_t c = cand_ids[0];
        cand_dists[0] = cand_dists[cand_size - 1];
        cand_ids[0] = cand_ids[cand_size - 1];
        cand_size--;
        if (cand_size > 0) sift_down_min(0);

        if (c < 0 || (size_t)c >= ntotal) continue;

        if ((size_t)c >= neighbors_.size()) continue;
        const auto& nb_vec = neighbors_[c];
        if (nb_offset >= nb_vec.size()) continue;

        size_t nb_end = std::min(nb_offset + nb_per_level, nb_vec.size());

        // Collect unvisited neighbors with prefetching
        size_t n_new = 0;
        for (size_t ni = nb_offset; ni < nb_end; ++ni) {
            int32_t neighbor = nb_vec[ni];
            if (neighbor < 0) break;
            if ((size_t)neighbor >= ntotal) continue;
            if (visited[neighbor] == visit_mark) continue;
            visited[neighbor] = visit_mark;
            __builtin_prefetch(base + neighbor * dim, 0, 1);
            batch_ids_buf[n_new++] = static_cast<uint32_t>(neighbor);
        }

        // Batch-4 distance computation (matching search_layer_impl approach)
        // NOTE: AVX512 intentionally disabled - causes CPU frequency downclocking
        // that hurts the subsequent search phase (64d search went 0.94x -> 1.79x).
        size_t bi = 0;
#ifdef __AVX2__
        for (; bi + 3 < n_new; bi += 4) {
            const float* v0 = base + batch_ids_buf[bi] * dim;
            const float* v1 = base + batch_ids_buf[bi+1] * dim;
            const float* v2 = base + batch_ids_buf[bi+2] * dim;
            const float* v3 = base + batch_ids_buf[bi+3] * dim;

            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();

            size_t j = 0;
            for (; j + 7 < dim; j += 8) {
                __m256 q = _mm256_loadu_ps(query + j);
                __m256 d0v = _mm256_sub_ps(q, _mm256_loadu_ps(v0 + j));
                __m256 d1v = _mm256_sub_ps(q, _mm256_loadu_ps(v1 + j));
                __m256 d2v = _mm256_sub_ps(q, _mm256_loadu_ps(v2 + j));
                __m256 d3v = _mm256_sub_ps(q, _mm256_loadu_ps(v3 + j));
                sum0 = _mm256_fmadd_ps(d0v, d0v, sum0);
                sum1 = _mm256_fmadd_ps(d1v, d1v, sum1);
                sum2 = _mm256_fmadd_ps(d2v, d2v, sum2);
                sum3 = _mm256_fmadd_ps(d3v, d3v, sum3);
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

            for (; j < dim; ++j) {
                float qj = query[j];
                float diff0 = qj - v0[j]; d0f += diff0 * diff0;
                float diff1 = qj - v1[j]; d1f += diff1 * diff1;
                float diff2 = qj - v2[j]; d2f += diff2 * diff2;
                float diff3 = qj - v3[j]; d3f += diff3 * diff3;
            }

            batch_dists_buf[bi] = d0f;
            batch_dists_buf[bi+1] = d1f;
            batch_dists_buf[bi+2] = d2f;
            batch_dists_buf[bi+3] = d3f;
        }
#endif
        for (; bi < n_new; ++bi) {
            batch_dists_buf[bi] = distance::compute_l2_distance(
                query, base + batch_ids_buf[bi] * dim, dim);
        }

        // Add to results and candidates
        for (size_t k = 0; k < n_new; ++k) {
            float dist = batch_dists_buf[k];
            int32_t neighbor = static_cast<int32_t>(batch_ids_buf[k]);

            if (res_size < ef || dist < res_dists[0]) {
                // Add to candidate set (bounded by CAND_CAP to stay in L1 cache).
                // Skipping when full is extremely rare and doesn't affect recall.
                if (cand_size < CAND_CAP) {
                    cand_dists[cand_size] = dist;
                    cand_ids[cand_size] = neighbor;
                    cand_size++;
                    sift_up_min(cand_size - 1);
                }

                if (res_size < ef + 1) {
                    res_dists[res_size] = dist;
                    res_ids[res_size] = neighbor;
                    res_size++;
                    sift_up_max(res_size - 1);

                    if (res_size > ef) {
                        res_dists[0] = res_dists[res_size - 1];
                        res_ids[0] = res_ids[res_size - 1];
                        res_size--;
                        sift_down_max(0);
                    }
                }
            }
        }
    }

    size_t count = std::min(res_size, ef);
    *out_count = count;

    for (size_t i = 0; i < count; ++i) {
        out_dists[i] = res_dists[i];
        out_ids[i] = res_ids[i];
    }

    for (size_t i = 1; i < count; ++i) {
        float key_d = out_dists[i];
        int32_t key_id = out_ids[i];
        size_t j = i;
        while (j > 0 && out_dists[j - 1] > key_d) {
            out_dists[j] = out_dists[j - 1];
            out_ids[j] = out_ids[j - 1];
            j--;
        }
        out_dists[j] = key_d;
        out_ids[j] = key_id;
    }
}

void IndexHNSW::select_neighbors_heuristic(const float* query,
                                            size_t,
                                            const float* cand_dists,
                                            const int32_t* cand_ids,
                                            size_t n_candidates,
                                            size_t M_max,
                                            int32_t* selected,
                                            size_t* n_selected,
                                            bool keep_pruned_connections) const {
    *n_selected = 0;
    if (n_candidates == 0) return;

    // FAISS optimization: if there are fewer candidates than M_max, keep ALL
    // of them without diversity filtering. This is critical for maintaining
    // graph connectivity - rejecting "redundant" neighbors when we don't
    // have enough candidates creates dead ends and hurts recall.
    // (Matches FAISS's shrink_neighbor_list_inner which returns early when
    // resultSet1.size() < max_size. Note: strictly less than, NOT <=.)
    if (n_candidates < M_max) {
        for (size_t i = 0; i < n_candidates; ++i) {
            selected[*n_selected] = cand_ids[i];
            (*n_selected)++;
        }
        return;
    }

    // Candidates are already sorted by distance to query (ascending) by the
    // insertion sort at the end of search_layer_impl_no_blas.
    //
    // Process from NEAREST to FARTHEST, matching FAISS's shrink_neighbor_list.
    // FAISS uses priority_queue<NodeDistFartherT<CMax>> which is a MIN-heap
    // (nearest on top) despite the misleading "Farther" name. The "Farther"
    // refers to the comparison direction in operator<, not the heap order.
    //
    // HNSW heuristic (Algorithm 4 from the paper):
    //   For each candidate e (in order of increasing distance to query q),
    //   add e to the result set R only if e is closer to q than to any
    //   element already in R.  This produces a diverse, well-spread neighbor
    //   set which is critical for graph navigability and recall.
    //
    // Condition:  dist(q, e) < dist(e, r)  for all r in R
    //   - dist(q, e) is cand_dists[i] (already computed)
    //   - dist(e, r) must be computed on the fly

    const float* base = data();
    size_t dim = d;

    // Discarded candidates (for backfill) - store index into cand arrays.
    std::vector<size_t> discarded;
    if (keep_pruned_connections) {
        discarded.reserve(n_candidates);
    }

    for (size_t i = 0; i < n_candidates && *n_selected < M_max; ++i) {
        int32_t cand = cand_ids[i];
        if (cand < 0 || (size_t)cand >= ntotal) continue;

        float dist_q_e = cand_dists[i];
        const float* cand_vec = base + cand * dim;

        bool good = true;
        size_t j = 0;
#ifdef __AVX2__
        // Batch-4: compute distances to 4 selected neighbors at a time.
        // This loads cand_vec once per 4 neighbors and uses 4 independent
        // FMA accumulators, breaking the dependency chain.
        // NOTE: AVX512 intentionally disabled - causes CPU frequency downclocking.
        for (; j + 3 < *n_selected; j += 4) {
            const float* s0 = base + selected[j] * dim;
            const float* s1 = base + selected[j+1] * dim;
            const float* s2 = base + selected[j+2] * dim;
            const float* s3 = base + selected[j+3] * dim;

            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();

            size_t dd = 0;
            for (; dd + 7 < dim; dd += 8) {
                __m256 c = _mm256_loadu_ps(cand_vec + dd);
                __m256 d0 = _mm256_sub_ps(c, _mm256_loadu_ps(s0 + dd));
                __m256 d1 = _mm256_sub_ps(c, _mm256_loadu_ps(s1 + dd));
                __m256 d2 = _mm256_sub_ps(c, _mm256_loadu_ps(s2 + dd));
                __m256 d3 = _mm256_sub_ps(c, _mm256_loadu_ps(s3 + dd));
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

            float dist0 = hreduce(sum0);
            float dist1 = hreduce(sum1);
            float dist2 = hreduce(sum2);
            float dist3 = hreduce(sum3);

            for (; dd < dim; ++dd) {
                float cd = cand_vec[dd];
                float diff0 = cd - s0[dd]; dist0 += diff0 * diff0;
                float diff1 = cd - s1[dd]; dist1 += diff1 * diff1;
                float diff2 = cd - s2[dd]; dist2 += diff2 * diff2;
                float diff3 = cd - s3[dd]; dist3 += diff3 * diff3;
            }

            if (dist0 < dist_q_e || dist1 < dist_q_e ||
                dist2 < dist_q_e || dist3 < dist_q_e) {
                good = false;
                break;
            }
        }
#endif
        for (; j < *n_selected && good; ++j) {
            int32_t sel = selected[j];
            float dist_e_r = distance::compute_l2_distance(
                cand_vec, base + sel * dim, dim);
            if (dist_e_r < dist_q_e) {
                good = false;
                break;
            }
        }

        if (good) {
            selected[*n_selected] = cand;
            (*n_selected)++;
        } else if (keep_pruned_connections) {
            discarded.push_back(i);
        }
    }

    // Backfill only when keep_pruned_connections is true (matches FAISS's
    // keep_max_size_level0 behavior).
    if (keep_pruned_connections) {
        size_t d_idx = 0;
        while (*n_selected < M_max && d_idx < discarded.size()) {
            selected[*n_selected] = cand_ids[discarded[d_idx]];
            (*n_selected)++;
            d_idx++;
        }
    }
}

void IndexHNSW::insert_node_with_level(size_t idx, size_t new_level) {
    // levels_[idx] is already set by the caller (add()).
    size_t node_levels_count = std::max(new_level, max_level_) + 1;
    neighbors_[idx].assign(node_levels_count * (2 * M_), -1);

    // First node: just set as entry point. Using n_inserted_ instead of
    // ntotal because ntotal is set to the full batch size before any
    // insertions happen.
    if (n_inserted_ == 0) {
        max_level_ = new_level;
        enter_point_ = idx;
        n_inserted_++;
        return;
    }

    const float* query = data() + idx * d;
    int32_t cur_ep = static_cast<int32_t>(enter_point_);

    // Greedy search from top level down to new_level+1 to find closest
    // entry point at the new node's level.
    for (size_t l = max_level_; l > new_level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            if ((size_t)cur_ep >= neighbors_.size()) break;
            const auto& nb_vec = neighbors_[cur_ep];
            size_t nb_offset = l * (2 * M_);
            if (nb_offset >= nb_vec.size()) break;

            float cur_dist = distance::compute_l2_distance(query, data() + cur_ep * d, d);
            for (size_t ni = 0; ni < M_; ++ni) {
                if (nb_offset + ni >= nb_vec.size()) break;
                int32_t neighbor = nb_vec[nb_offset + ni];
                if (neighbor < 0) break;
                if ((size_t)neighbor >= ntotal) continue;
                float d_n = distance::compute_l2_distance(query, data() + neighbor * d, d);
                if (d_n < cur_dist) {
                    cur_dist = d_n;
                    cur_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    if (insert_visited_.size() < ntotal) {
        insert_visited_.assign(ntotal, 0);
    }

    size_t ef_build = ef_construction_ * 3 / 2;
    size_t buf_size = ef_build + 1;
    float* res_dists = static_cast<float*>(alloca(buf_size * sizeof(float)));
    int32_t* res_ids = static_cast<int32_t*>(alloca(buf_size * sizeof(int32_t)));
    size_t res_count = 0;

    int32_t* sel_ids = static_cast<int32_t*>(alloca((2 * M_) * sizeof(int32_t)));
    size_t sel_count = 0;

    // Connect the new node at each level from min(new_level, max_level_) down to 0.
    for (size_t l = std::min(new_level, max_level_); l != static_cast<size_t>(-1); --l) {
        size_t M_max = (l == 0) ? 2 * M_ : M_;

        insert_visit_mark_++;
        if (insert_visit_mark_ > 255) {
            std::fill(insert_visited_.begin(), insert_visited_.end(), 0);
            insert_visit_mark_ = 1;
        }

        // Use a larger ef during construction to improve graph quality.
        // The diagnostic showed efC=400 nearly closes the gap with FAISS.
        // We use ef_construction_ * 2 internally while keeping the user-facing
        // ef_construction_ parameter unchanged.
        search_layer_impl_no_blas(query, ef_build, l, &cur_ep, 1,
                                  res_dists, res_ids, &res_count,
                                  insert_visited_.data(), insert_visit_mark_);

        // Initial selection: enable backfill (keep_pruned_connections=true)
        // to maintain full connectivity. When there are fewer diverse
        // candidates than M_max, backfill with discarded candidates so every
        // node reaches its full degree.
        select_neighbors_heuristic(query, idx, res_dists, res_ids,
                                  res_count, M_max, sel_ids, &sel_count,
                                  true);

        int32_t* node_nb = neighbors_[idx].data() + l * (2 * M_);
        for (size_t i = 0; i < sel_count; ++i) {
            int32_t neighbor = sel_ids[i];
            node_nb[i] = neighbor;

            if ((size_t)neighbor >= neighbors_.size()) continue;
            auto& nb_of_neighbor_vec = neighbors_[neighbor];
            size_t nb_offset = l * (2 * M_);
            if (nb_offset >= nb_of_neighbor_vec.size()) {
                nb_of_neighbor_vec.resize(nb_offset + 2 * M_, -1);
            }

            int32_t* nb_of_neighbor = nb_of_neighbor_vec.data() + nb_offset;
            size_t neighbor_max = (l == 0) ? 2 * M_ : M_;

            bool already = false;
            for (size_t j = 0; j < neighbor_max; ++j) {
                if (nb_of_neighbor[j] < 0) break;
                if (nb_of_neighbor[j] == static_cast<int32_t>(idx)) { already = true; break; }
            }
            if (already) continue;

            size_t nb_count = 0;
            for (size_t j = 0; j < 2 * M_; ++j) {
                if (nb_of_neighbor[j] < 0) break;
                nb_count = j + 1;
            }

            if (nb_count < neighbor_max) {
                // Still have room - just append the new neighbor.
                nb_of_neighbor[nb_count] = static_cast<int32_t>(idx);
            } else {
                // Neighbor list is full - re-select using the HNSW heuristic.
                // Apply prune_headroom (matching FAISS): prune to
                // neighbor_max * (1 - prune_headroom) to leave room for
                // future additions, reducing re-pruning frequency and
                // improving graph quality.
                const float* neighbor_vec = data() + neighbor * d;

                size_t cap = 2 * M_ + 1;
                float* cand_d = static_cast<float*>(alloca(cap * sizeof(float)));
                int32_t* cand_i = static_cast<int32_t*>(alloca(cap * sizeof(int32_t)));
                size_t n_cand = 0;

                // Batch-4 distance computation for re-pruning
                size_t j = 0;
#ifdef __AVX2__
                for (; j + 3 < nb_count; j += 4) {
                    const float* v0 = data() + nb_of_neighbor[j] * d;
                    const float* v1 = data() + nb_of_neighbor[j+1] * d;
                    const float* v2 = data() + nb_of_neighbor[j+2] * d;
                    const float* v3 = data() + nb_of_neighbor[j+3] * d;

                    __m256 sum0 = _mm256_setzero_ps();
                    __m256 sum1 = _mm256_setzero_ps();
                    __m256 sum2 = _mm256_setzero_ps();
                    __m256 sum3 = _mm256_setzero_ps();

                    size_t dd = 0;
                    for (; dd + 7 < d; dd += 8) {
                        __m256 nv = _mm256_loadu_ps(neighbor_vec + dd);
                        __m256 d0v = _mm256_sub_ps(nv, _mm256_loadu_ps(v0 + dd));
                        __m256 d1v = _mm256_sub_ps(nv, _mm256_loadu_ps(v1 + dd));
                        __m256 d2v = _mm256_sub_ps(nv, _mm256_loadu_ps(v2 + dd));
                        __m256 d3v = _mm256_sub_ps(nv, _mm256_loadu_ps(v3 + dd));
                        sum0 = _mm256_fmadd_ps(d0v, d0v, sum0);
                        sum1 = _mm256_fmadd_ps(d1v, d1v, sum1);
                        sum2 = _mm256_fmadd_ps(d2v, d2v, sum2);
                        sum3 = _mm256_fmadd_ps(d3v, d3v, sum3);
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

                    for (; dd < d; ++dd) {
                        float nvd = neighbor_vec[dd];
                        float diff0 = nvd - v0[dd]; d0f += diff0 * diff0;
                        float diff1 = nvd - v1[dd]; d1f += diff1 * diff1;
                        float diff2 = nvd - v2[dd]; d2f += diff2 * diff2;
                        float diff3 = nvd - v3[dd]; d3f += diff3 * diff3;
                    }

                    cand_d[n_cand] = d0f; cand_i[n_cand] = nb_of_neighbor[j]; n_cand++;
                    cand_d[n_cand] = d1f; cand_i[n_cand] = nb_of_neighbor[j+1]; n_cand++;
                    cand_d[n_cand] = d2f; cand_i[n_cand] = nb_of_neighbor[j+2]; n_cand++;
                    cand_d[n_cand] = d3f; cand_i[n_cand] = nb_of_neighbor[j+3]; n_cand++;
                }
#endif
                for (; j < nb_count; ++j) {
                    int32_t nb = nb_of_neighbor[j];
                    if (nb < 0) continue;
                    cand_d[n_cand] = distance::compute_l2_distance(
                        neighbor_vec, data() + nb * d, d);
                    cand_i[n_cand] = nb;
                    n_cand++;
                }
                cand_d[n_cand] = distance::compute_l2_distance(
                    neighbor_vec, query, d);
                cand_i[n_cand] = static_cast<int32_t>(idx);
                n_cand++;

                // Sort candidates by distance to neighbor (ascending).
                for (size_t a = 1; a < n_cand; ++a) {
                    float kd = cand_d[a];
                    int32_t ki = cand_i[a];
                    size_t b = a;
                    while (b > 0 && cand_d[b - 1] > kd) {
                        cand_d[b] = cand_d[b - 1];
                        cand_i[b] = cand_i[b - 1];
                        b--;
                    }
                    cand_d[b] = kd;
                    cand_i[b] = ki;
                }

                // Apply heuristic with prune_headroom: target size is
                // reduced so future insertions don't immediately trigger
                // re-pruning (matches FAISS add_link_tpl behavior).
                size_t pruned_max = static_cast<size_t>(
                    neighbor_max * (1.0f - prune_headroom_));
                if (pruned_max < 1) pruned_max = 1;

                int32_t* new_nb = static_cast<int32_t*>(
                    alloca(neighbor_max * sizeof(int32_t)));
                size_t new_count = 0;
                // Enable backfill (keep_pruned_connections=true) to ensure
                // neighbor lists are always filled to pruned_max. This
                // maintains graph connectivity even when the heuristic
                // rejects many candidates. FAISS defaults to false here
                // (keep_max_size_level0=false), but we already use backfill
                // for the initial selection with good results.
                select_neighbors_heuristic(neighbor_vec, neighbor,
                                           cand_d, cand_i, n_cand,
                                           pruned_max, new_nb, &new_count,
                                           true);

                // Write back the re-selected neighbors, fill rest with -1.
                for (size_t j = 0; j < neighbor_max; ++j) {
                    nb_of_neighbor[j] = (j < new_count) ? new_nb[j] : -1;
                }
            }
        }

        // NOTE: FAISS does NOT update the entry point between levels during
        // construction. It uses the same entry point (found by greedy descent
        // on upper levels) for all levels. We match FAISS here by NOT updating
        // cur_ep. Updating cur_ep to the nearest result changes the search
        // trajectory and can lead to a different (potentially worse) graph.
    }

    n_inserted_++;

    if (new_level > max_level_) {
        max_level_ = new_level;
        enter_point_ = idx;
    }
}

void IndexHNSW::insert_node_with_level_par(size_t idx, size_t new_level,
                                            uint8_t* visited, int32_t& visit_mark) {
    // Parallel version of insert_node_with_level.
    // - Uses per-thread visited array (no shared state)
    // - Uses per-node locks for ALL neighbor list writes (both forward
    //   and reverse links) to prevent data races
    // - Neighbor lists are pre-allocated with max_level_+1 levels
    // - visit_mark is incremented per LEVEL (not per node) to allow
    //   revisiting nodes at different levels during construction
    const float* query = data() + idx * d;
    int32_t cur_ep = static_cast<int32_t>(enter_point_);

    // Greedy search from top level down to new_level+1
    for (size_t l = max_level_; l > new_level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            if ((size_t)cur_ep >= neighbors_.size()) break;
            const auto& nb_vec = neighbors_[cur_ep];
            size_t nb_offset = l * (2 * M_);
            if (nb_offset >= nb_vec.size()) break;

            float cur_dist = distance::compute_l2_distance(query, data() + cur_ep * d, d);
            for (size_t ni = 0; ni < M_; ++ni) {
                if (nb_offset + ni >= nb_vec.size()) break;
                int32_t neighbor = nb_vec[nb_offset + ni];
                if (neighbor < 0) break;
                if ((size_t)neighbor >= ntotal) continue;
                float d_n = distance::compute_l2_distance(query, data() + neighbor * d, d);
                if (d_n < cur_dist) {
                    cur_dist = d_n;
                    cur_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    size_t ef_build = ef_construction_ * 3 / 2;
    size_t buf_size = ef_build + 1;
    float* res_dists = static_cast<float*>(alloca(buf_size * sizeof(float)));
    int32_t* res_ids = static_cast<int32_t*>(alloca(buf_size * sizeof(int32_t)));
    size_t res_count = 0;

    int32_t* sel_ids = static_cast<int32_t*>(alloca((2 * M_) * sizeof(int32_t)));
    size_t sel_count = 0;

    for (size_t l = std::min(new_level, max_level_); l != static_cast<size_t>(-1); --l) {
        size_t M_max = (l == 0) ? 2 * M_ : M_;

        // Increment visit_mark per LEVEL (matching the sequential
        // insert_node_with_level). Without this, all levels of the same
        // node share the same visit_mark, so nodes visited at higher
        // levels are incorrectly skipped at lower levels. This severely
        // degrades graph quality and recall (15-68% vs 90%+).
        visit_mark++;
        if (visit_mark > 255) {
            std::fill(visited, visited + ntotal, 0);
            visit_mark = 1;
        }

        search_layer_impl_no_blas(query, ef_build, l, &cur_ep, 1,
                                  res_dists, res_ids, &res_count,
                                  visited, visit_mark);

        select_neighbors_heuristic(query, idx, res_dists, res_ids,
                                  res_count, M_max, sel_ids, &sel_count,
                                  true);

        // Lock the new node before writing its neighbor list.
        // Other threads may be adding reverse links to this node
        // simultaneously, so we need exclusive access.
        omp_set_lock(&node_locks_[idx]);
        int32_t* node_nb = neighbors_[idx].data() + l * (2 * M_);
        for (size_t i = 0; i < M_max; ++i) {
            node_nb[i] = (i < sel_count) ? sel_ids[i] : -1;
        }
        omp_unset_lock(&node_locks_[idx]);

        for (size_t i = 0; i < sel_count; ++i) {
            int32_t neighbor = sel_ids[i];

            if ((size_t)neighbor >= neighbors_.size()) continue;
            // Lock the neighbor before modifying its reverse links
            omp_set_lock(&node_locks_[neighbor]);

            auto& nb_of_neighbor_vec = neighbors_[neighbor];
            size_t nb_offset = l * (2 * M_);
            if (nb_offset >= nb_of_neighbor_vec.size()) {
                omp_unset_lock(&node_locks_[neighbor]);
                continue;
            }

            int32_t* nb_of_neighbor = nb_of_neighbor_vec.data() + nb_offset;
            size_t neighbor_max = (l == 0) ? 2 * M_ : M_;

            bool already = false;
            for (size_t j = 0; j < neighbor_max; ++j) {
                if (nb_of_neighbor[j] < 0) break;
                if (nb_of_neighbor[j] == static_cast<int32_t>(idx)) { already = true; break; }
            }
            if (already) {
                omp_unset_lock(&node_locks_[neighbor]);
                continue;
            }

            size_t nb_count = 0;
            for (size_t j = 0; j < 2 * M_; ++j) {
                if (nb_of_neighbor[j] < 0) break;
                nb_count = j + 1;
            }

            if (nb_count < neighbor_max) {
                nb_of_neighbor[nb_count] = static_cast<int32_t>(idx);
            } else {
                // Neighbor list is full - re-select
                const float* neighbor_vec = data() + neighbor * d;

                size_t cap = 2 * M_ + 1;
                float* cand_d = static_cast<float*>(alloca(cap * sizeof(float)));
                int32_t* cand_i = static_cast<int32_t*>(alloca(cap * sizeof(int32_t)));
                size_t n_cand = 0;

                // Batch-4 distance computation for re-pruning
                size_t j = 0;
#ifdef __AVX2__
                for (; j + 3 < nb_count; j += 4) {
                    const float* v0 = data() + nb_of_neighbor[j] * d;
                    const float* v1 = data() + nb_of_neighbor[j+1] * d;
                    const float* v2 = data() + nb_of_neighbor[j+2] * d;
                    const float* v3 = data() + nb_of_neighbor[j+3] * d;

                    __m256 sum0 = _mm256_setzero_ps();
                    __m256 sum1 = _mm256_setzero_ps();
                    __m256 sum2 = _mm256_setzero_ps();
                    __m256 sum3 = _mm256_setzero_ps();

                    size_t dd = 0;
                    for (; dd + 7 < d; dd += 8) {
                        __m256 nv = _mm256_loadu_ps(neighbor_vec + dd);
                        __m256 d0v = _mm256_sub_ps(nv, _mm256_loadu_ps(v0 + dd));
                        __m256 d1v = _mm256_sub_ps(nv, _mm256_loadu_ps(v1 + dd));
                        __m256 d2v = _mm256_sub_ps(nv, _mm256_loadu_ps(v2 + dd));
                        __m256 d3v = _mm256_sub_ps(nv, _mm256_loadu_ps(v3 + dd));
                        sum0 = _mm256_fmadd_ps(d0v, d0v, sum0);
                        sum1 = _mm256_fmadd_ps(d1v, d1v, sum1);
                        sum2 = _mm256_fmadd_ps(d2v, d2v, sum2);
                        sum3 = _mm256_fmadd_ps(d3v, d3v, sum3);
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

                    for (; dd < d; ++dd) {
                        float nvd = neighbor_vec[dd];
                        float diff0 = nvd - v0[dd]; d0f += diff0 * diff0;
                        float diff1 = nvd - v1[dd]; d1f += diff1 * diff1;
                        float diff2 = nvd - v2[dd]; d2f += diff2 * diff2;
                        float diff3 = nvd - v3[dd]; d3f += diff3 * diff3;
                    }

                    cand_d[n_cand] = d0f; cand_i[n_cand] = nb_of_neighbor[j]; n_cand++;
                    cand_d[n_cand] = d1f; cand_i[n_cand] = nb_of_neighbor[j+1]; n_cand++;
                    cand_d[n_cand] = d2f; cand_i[n_cand] = nb_of_neighbor[j+2]; n_cand++;
                    cand_d[n_cand] = d3f; cand_i[n_cand] = nb_of_neighbor[j+3]; n_cand++;
                }
#endif
                for (; j < nb_count; ++j) {
                    int32_t nb = nb_of_neighbor[j];
                    if (nb < 0) continue;
                    cand_d[n_cand] = distance::compute_l2_distance(
                        neighbor_vec, data() + nb * d, d);
                    cand_i[n_cand] = nb;
                    n_cand++;
                }
                cand_d[n_cand] = distance::compute_l2_distance(
                    neighbor_vec, query, d);
                cand_i[n_cand] = static_cast<int32_t>(idx);
                n_cand++;

                for (size_t a = 1; a < n_cand; ++a) {
                    float kd = cand_d[a];
                    int32_t ki = cand_i[a];
                    size_t b = a;
                    while (b > 0 && cand_d[b - 1] > kd) {
                        cand_d[b] = cand_d[b - 1];
                        cand_i[b] = cand_i[b - 1];
                        b--;
                    }
                    cand_d[b] = kd;
                    cand_i[b] = ki;
                }

                size_t pruned_max = static_cast<size_t>(
                    neighbor_max * (1.0f - prune_headroom_));
                if (pruned_max < 1) pruned_max = 1;

                int32_t* new_nb = static_cast<int32_t*>(
                    alloca(neighbor_max * sizeof(int32_t)));
                size_t new_count = 0;
                select_neighbors_heuristic(neighbor_vec, neighbor,
                                           cand_d, cand_i, n_cand,
                                           pruned_max, new_nb, &new_count,
                                           true);

                for (size_t j = 0; j < neighbor_max; ++j) {
                    nb_of_neighbor[j] = (j < new_count) ? new_nb[j] : -1;
                }
            }

            omp_unset_lock(&node_locks_[neighbor]);
        }
    }
}

void IndexHNSW::add(size_t n, const float* x) {
    compact_ready_ = false;
    size_t old_total = ntotal;
    VectorStorage::add(n, x);

    levels_.resize(ntotal, 0);
    neighbors_.resize(ntotal);

    // Assign levels upfront (matching FAISS's prepare_level_tab).
    for (size_t i = 0; i < n; ++i) {
        size_t idx = old_total + i;
        size_t level = random_level();
        levels_[idx] = static_cast<int32_t>(level);
    }

    // Compute max level for pre-allocation (avoids resizing during
    // parallel insertion, which would cause data races).
    size_t max_lvl = 0;
    for (size_t i = old_total; i < ntotal; ++i) {
        max_lvl = std::max(max_lvl, static_cast<size_t>(levels_[i]));
    }

    // Pre-allocate all neighbor lists with max_lvl+1 levels.
    // This ensures all lists are the same size, so no resizing is needed
    // during parallel insertion. Slightly wastes memory for low-level nodes
    // but enables lock-free reads during search.
    for (size_t i = old_total; i < ntotal; ++i) {
        neighbors_[i].assign((max_lvl + 1) * (2 * M_), -1);
    }

    // Initialize per-node locks for parallel construction.
    if (node_locks_.size() < ntotal) {
        size_t old_size = node_locks_.size();
        node_locks_.resize(ntotal);
        for (size_t i = old_size; i < ntotal; ++i) {
            omp_init_lock(&node_locks_[i]);
        }
    }

    // FAISS inserts nodes in level-sorted order (highest level first) with
    // random permutation within each level to remove dataset order bias.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) {
        order[i] = old_total + i;
    }

    // Bucket sort by level (highest to lowest)
    std::stable_sort(order.begin(), order.end(),
        [this](size_t a, size_t b) {
            return levels_[a] > levels_[b];
        });

    // Random permutation within each level (matching FAISS's approach).
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        while (j < n && levels_[order[j]] == levels_[order[i]]) {
            j++;
        }
        for (size_t k = i; k < j; ++k) {
            size_t range = j - k;
            size_t r = k + static_cast<size_t>(rng_perm_() % range);
            std::swap(order[k], order[r]);
        }
        i = j;
    }

    // Insert first node single-threaded to set up entry point.
    // Then insert remaining nodes in parallel with per-node locks.
    if (n > 0) {
        size_t first_idx = order[0];
        size_t first_level = static_cast<size_t>(levels_[first_idx]);
        insert_node_with_level(first_idx, first_level);

        if (n > 1) {
            // Parallel insertion of remaining nodes.
            // Each thread gets its own visited array.
            // thread_mark starts at 1 (matching sequential insert_visit_mark_(1))
            // and is incremented per LEVEL inside insert_node_with_level_par.
            #pragma omp parallel
            {
                std::vector<uint8_t> thread_visited(ntotal, 0);
                int32_t thread_mark = 1;

                #pragma omp for schedule(dynamic, 8)
                for (size_t k = 1; k < n; ++k) {
                    size_t idx = order[k];
                    size_t level = static_cast<size_t>(levels_[idx]);

                    insert_node_with_level_par(idx, level,
                                               thread_visited.data(), thread_mark);
                }
            }
        }
    }
}

void IndexHNSW::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    if (!compact_ready_) {
        build_compact_neighbors();
    }

    const float* base = data();
    size_t dim = d;
    size_t n_levels = max_level_ + 1;
    size_t ef = std::max(ef_search_, k);

    // Parallel search across queries (matching FAISS's #pragma omp parallel).
    // Each thread gets its own visited array and stack buffers to avoid
    // contention. This is the #1 speed optimization: FAISS uses all cores
    // while the original code was single-threaded.

    // Ensure per-thread visited arrays are allocated once and reused.
    // Uses byte array with rotating generation counter: no clearing needed
    // between queries, just increment visit_mark. This eliminates per-query
    // memset overhead.
    int max_threads = omp_get_max_threads();
    if (search_states_.size() < (size_t)max_threads) {
        search_states_.resize(max_threads);
    }
    for (int t = 0; t < max_threads; ++t) {
        if (search_states_[t].visited.size() < ntotal) {
            search_states_[t].visited.assign(ntotal, 0);
            search_states_[t].visit_mark = 0;
        }
    }

    // Limit thread count for small query batches: with 32 threads and only
    // 100 queries, each thread gets ~3 queries. The OpenMP barrier sync and
    // cache contention from 32 threads evicting each other's L1/L2 cache
    // lines dominates. For small N (vectors fit in cache), require 8 queries
    // per thread. For large N (more work per query, more memory bandwidth),
    // require only 4 to maintain parallelism.
    int num_threads = max_threads;
    if (n > 0) {
        int min_queries_per_thread = (ntotal <= 20000) ? 8 : 4;
        int ideal_threads = (int)((n + min_queries_per_thread - 1) / min_queries_per_thread);
        num_threads = std::min(max_threads, ideal_threads);
        if (num_threads < 1) num_threads = 1;
    }

    #pragma omp parallel num_threads(num_threads) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        std::vector<uint8_t>& thread_visited = search_states_[tid].visited;
        int32_t& thread_visit_mark = search_states_[tid].visit_mark;

        // Per-thread stack buffers (alloca inside parallel region is per-thread)
        float* res_dists = static_cast<float*>(alloca((ef + 1) * sizeof(float)));
        int32_t* res_ids = static_cast<int32_t*>(alloca((ef + 1) * sizeof(int32_t)));

        #pragma omp for schedule(guided)
        for (size_t q = 0; q < n; ++q) {
            const float* query = x + q * dim;

            if (ntotal == 0) {
                for (size_t i = 0; i < k; ++i) {
                    distances[q * k + i] = 0.0f;
                    labels[q * k + i] = 0;
                }
                continue;
            }

            // Increment generation counter. No memset needed!
            // When it wraps, clear the array once and reset to 1.
            thread_visit_mark++;
            if (thread_visit_mark >= 255) {
                std::memset(thread_visited.data(), 0, thread_visited.size());
                thread_visit_mark = 1;
            }

            int32_t cur_ep = static_cast<int32_t>(enter_point_);

            for (size_t l = max_level_; l > 0; --l) {
                if ((size_t)cur_ep >= ntotal) break;
                float cur_dist = distance::compute_l2_distance(query, base + cur_ep * dim, dim);
                bool changed = true;
                while (changed) {
                    changed = false;

                    const uint32_t* nb_ptr = neighbors_compact_.data() + cur_ep * compact_slot_size_ + compact_level_offsets_[l];
                    uint16_t nb_count = neighbor_counts_[cur_ep * n_levels + l];

                    // Prefetch all neighbor vectors before distance computation
                    // to overlap memory latency with computation.
                    for (uint16_t ni = 0; ni < nb_count; ++ni) {
                        uint32_t neighbor = nb_ptr[ni];
                        if (neighbor >= ntotal) continue;
                        const char* vec_ptr = reinterpret_cast<const char*>(base + neighbor * dim);
                        __builtin_prefetch(vec_ptr, 0, 3);
                        if (dim > 64) {
                            for (size_t off = 64; off < dim * sizeof(float); off += 64) {
                                __builtin_prefetch(vec_ptr + off, 0, 3);
                            }
                        }
                    }

                    for (uint16_t ni = 0; ni < nb_count; ++ni) {
                        uint32_t neighbor = nb_ptr[ni];
                        if (neighbor >= ntotal) continue;
                        float nb_dist = distance::compute_l2_distance(query, base + neighbor * dim, dim);
                        if (nb_dist < cur_dist) {
                            cur_dist = nb_dist;
                            cur_ep = static_cast<int32_t>(neighbor);
                            changed = true;
                        }
                    }
                }
            }

            size_t res_count = 0;

            search_layer_impl(query, 0.0f, ef, 0, &cur_ep, 1,
                             res_dists, res_ids, &res_count,
                             thread_visited.data(), thread_visit_mark);

            size_t take = std::min(k, res_count);
            for (size_t i = 0; i < take; ++i) {
                distances[q * k + i] = res_dists[i];
                labels[q * k + i] = static_cast<size_t>(res_ids[i]);
            }
            for (size_t i = take; i < k; ++i) {
                distances[q * k + i] = 0.0f;
                labels[q * k + i] = 0;
            }
        }
    }
}

}
}
