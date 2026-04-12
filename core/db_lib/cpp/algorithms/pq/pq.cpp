#include "pq.h"
#include <stdexcept>
#include <limits>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

namespace vectordb {
namespace algorithms {

IndexPQ::IndexPQ(size_t dimension, size_t M, size_t nbits)
    : VectorStorage(dimension), M_(M), nbits_(nbits), ksub_(1 << nbits), codes_dirty_(false) {
    if (dimension % M != 0) {
        throw std::invalid_argument("Dimension must be divisible by M");
    }
}

static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}

#ifdef __AVX512F__
static inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

static void train_kmeans_single(size_t dim_sub, size_t ksub, size_t max_iter,
                                 const float* sub_vectors, size_t n,
                                 float* centroids) {
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(perm.begin(), perm.end(), gen);

    for (size_t i = 0; i < ksub && i < n; ++i) {
        std::copy(sub_vectors + perm[i] * dim_sub,
                  sub_vectors + perm[i] * dim_sub + dim_sub,
                  centroids + i * dim_sub);
    }

    for (size_t iter = 0; iter < max_iter; ++iter) {
        std::vector<float> new_centroids(ksub * dim_sub, 0.0f);
        std::vector<size_t> counts(ksub, 0);

#ifdef __AVX512F__
        if (dim_sub == 8) {
            for (size_t i = 0; i < n; ++i) {
                const float* vec = sub_vectors + i * 8;
                __m256 v = _mm256_loadu_ps(vec);
                size_t best_k = 0;
                float best_dist = std::numeric_limits<float>::max();

                for (size_t k = 0; k < ksub; ++k) {
                    __m256 c = _mm256_loadu_ps(centroids + k * 8);
                    __m256 diff = _mm256_sub_ps(v, c);
                    __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
                    float dist = hsum_avx2(sq);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                float* sum = new_centroids.data() + best_k * 8;
                __m256 ns = _mm256_loadu_ps(sum);
                _mm256_storeu_ps(sum, _mm256_add_ps(ns, v));
                counts[best_k]++;
            }
        } else if (dim_sub == 16) {
            for (size_t i = 0; i < n; ++i) {
                const float* vec = sub_vectors + i * 16;
                __m256 v0 = _mm256_loadu_ps(vec);
                __m256 v1 = _mm256_loadu_ps(vec + 8);
                size_t best_k = 0;
                float best_dist = std::numeric_limits<float>::max();

                for (size_t k = 0; k < ksub; ++k) {
                    __m256 c0 = _mm256_loadu_ps(centroids + k * 16);
                    __m256 c1 = _mm256_loadu_ps(centroids + k * 16 + 8);
                    __m256 diff0 = _mm256_sub_ps(v0, c0);
                    __m256 diff1 = _mm256_sub_ps(v1, c1);
                    __m256 sq0 = _mm256_fmadd_ps(diff0, diff0, _mm256_setzero_ps());
                    __m256 sq1 = _mm256_fmadd_ps(diff1, diff1, _mm256_setzero_ps());
                    float dist = hsum_avx2(_mm256_add_ps(sq0, sq1));
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                float* sum = new_centroids.data() + best_k * 16;
                __m256 ns0 = _mm256_loadu_ps(sum);
                __m256 ns1 = _mm256_loadu_ps(sum + 8);
                _mm256_storeu_ps(sum, _mm256_add_ps(ns0, v0));
                _mm256_storeu_ps(sum + 8, _mm256_add_ps(ns1, v1));
                counts[best_k]++;
            }
        } else
#elif defined(__AVX2__)
        if (dim_sub == 8) {
            for (size_t i = 0; i < n; ++i) {
                const float* vec = sub_vectors + i * 8;
                __m256 v = _mm256_loadu_ps(vec);
                size_t best_k = 0;
                float best_dist = std::numeric_limits<float>::max();

                for (size_t k = 0; k < ksub; ++k) {
                    __m256 c = _mm256_loadu_ps(centroids + k * 8);
                    __m256 diff = _mm256_sub_ps(v, c);
                    __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
                    float dist = hsum_avx2(sq);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                float* sum = new_centroids.data() + best_k * 8;
                __m256 ns = _mm256_loadu_ps(sum);
                _mm256_storeu_ps(sum, _mm256_add_ps(ns, v));
                counts[best_k]++;
            }
        } else if (dim_sub == 16) {
            for (size_t i = 0; i < n; ++i) {
                const float* vec = sub_vectors + i * 16;
                __m256 v0 = _mm256_loadu_ps(vec);
                __m256 v1 = _mm256_loadu_ps(vec + 8);
                size_t best_k = 0;
                float best_dist = std::numeric_limits<float>::max();

                for (size_t k = 0; k < ksub; ++k) {
                    __m256 c0 = _mm256_loadu_ps(centroids + k * 16);
                    __m256 c1 = _mm256_loadu_ps(centroids + k * 16 + 8);
                    __m256 diff0 = _mm256_sub_ps(v0, c0);
                    __m256 diff1 = _mm256_sub_ps(v1, c1);
                    __m256 sq0 = _mm256_fmadd_ps(diff0, diff0, _mm256_setzero_ps());
                    __m256 sq1 = _mm256_fmadd_ps(diff1, diff1, _mm256_setzero_ps());
                    float dist = hsum_avx2(_mm256_add_ps(sq0, sq1));
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                float* sum = new_centroids.data() + best_k * 16;
                __m256 ns0 = _mm256_loadu_ps(sum);
                __m256 ns1 = _mm256_loadu_ps(sum + 8);
                _mm256_storeu_ps(sum, _mm256_add_ps(ns0, v0));
                _mm256_storeu_ps(sum + 8, _mm256_add_ps(ns1, v1));
                counts[best_k]++;
            }
        } else
#endif
        {
            for (size_t i = 0; i < n; ++i) {
                const float* vec = sub_vectors + i * dim_sub;
                size_t best_k = 0;
                float best_dist = std::numeric_limits<float>::max();

                for (size_t k = 0; k < ksub; ++k) {
                    float dist = distance::compute_l2_distance(vec, centroids + k * dim_sub, dim_sub);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                float* sum = new_centroids.data() + best_k * dim_sub;
                for (size_t j = 0; j < dim_sub; ++j) {
                    sum[j] += vec[j];
                }
                counts[best_k]++;
            }
        }

        float max_shift = 0.0f;
        for (size_t k = 0; k < ksub; ++k) {
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

        if (max_shift < 1e-3) {
            break;
        }
    }
}

void IndexPQ::train_kmeans(size_t m, const float* x, size_t n, float* centroids) {
    size_t dim_sub = d / M_;
    std::vector<float> sub_vectors(n * dim_sub);
    for (size_t i = 0; i < n; ++i) {
        std::copy(x + i * d + m * dim_sub,
                  x + i * d + (m + 1) * dim_sub,
                  sub_vectors.data() + i * dim_sub);
    }
    train_kmeans_single(dim_sub, ksub_, 12, sub_vectors.data(), n, centroids);
}

void IndexPQ::encode_vector(const float* x, uint8_t* code) const {
    size_t dim_sub = d / M_;
    for (size_t m = 0; m < M_; ++m) {
        const float* vec_sub = x + m * dim_sub;
        const float* centroids = codebooks_.data() + m * ksub_ * dim_sub;
        size_t best_k = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t k = 0; k < ksub_; ++k) {
            float dist = distance::compute_l2_distance(vec_sub, centroids + k * dim_sub, dim_sub);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }
        code[m] = static_cast<uint8_t>(best_k);
    }
}

void IndexPQ::build_transposed_codes() const {
    codes_transposed_.resize(M_ * ntotal);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ntotal; ++i) {
        for (size_t m = 0; m < M_; ++m) {
            codes_transposed_[m * ntotal + i] = codes_[i * M_ + m];
        }
    }
    codes_dirty_ = false;
}

void IndexPQ::train(size_t n, const float* x) {
    size_t dim_sub = d / M_;
    codebooks_.resize(M_ * ksub_ * dim_sub);

    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M_; ++m) {
        std::vector<float> sub_vectors(n * dim_sub);
        for (size_t i = 0; i < n; ++i) {
            std::copy(x + i * d + m * dim_sub,
                      x + i * d + (m + 1) * dim_sub,
                      sub_vectors.data() + i * dim_sub);
        }
        train_kmeans_single(dim_sub, ksub_, 10, sub_vectors.data(), n,
                            codebooks_.data() + m * ksub_ * dim_sub);
    }

    centroid_norms_.resize(M_ * ksub_);
    for (size_t m = 0; m < M_; ++m) {
        const float* centroids = codebooks_.data() + m * ksub_ * dim_sub;
        for (size_t k = 0; k < ksub_; ++k) {
            float norm = 0.0f;
            for (size_t j = 0; j < dim_sub; ++j) {
                norm += centroids[k * dim_sub + j] * centroids[k * dim_sub + j];
            }
            centroid_norms_[m * ksub_ + k] = norm;
        }
    }

    codebooks_t_.resize(M_ * dim_sub * ksub_);
    for (size_t m = 0; m < M_; ++m) {
        const float* centroids = codebooks_.data() + m * ksub_ * dim_sub;
        float* ct = codebooks_t_.data() + m * dim_sub * ksub_;
        for (size_t j = 0; j < dim_sub; ++j) {
            for (size_t k = 0; k < ksub_; ++k) {
                ct[j * ksub_ + k] = centroids[k * dim_sub + j];
            }
        }
    }

    codes_dirty_ = true;
}

void IndexPQ::add(size_t n, const float* x) {
    if (codebooks_.empty()) {
        throw std::runtime_error("Index not trained");
    }

    ntotal += n;

    size_t old_size = codes_.size();
    codes_.resize(old_size + n * M_);

    size_t dim_sub = d / M_;

    for (size_t m = 0; m < M_; ++m) {
#ifdef __AVX512F__
        if (dim_sub == 8 && !codebooks_t_.empty()) {
            const float* ct = codebooks_t_.data() + m * 8 * ksub_;
            const float* cn = centroid_norms_.data() + m * ksub_;

            #pragma omp parallel for schedule(static, 256)
            for (size_t i = 0; i < n; ++i) {
                const float* vec_sub = x + i * d + m * 8;
                uint8_t* code = codes_.data() + old_size + i * M_;

                __m512 best_vals = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
                __m512i best_idxs = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

                for (size_t k = 0; k + 15 < ksub_; k += 16) {
                    __m512 dot_acc = _mm512_setzero_ps();
                    for (size_t j = 0; j < 8; ++j) {
                        __m512 vj = _mm512_set1_ps(vec_sub[j]);
                        __m512 cj = _mm512_loadu_ps(ct + j * ksub_ + k);
                        dot_acc = _mm512_fmadd_ps(vj, cj, dot_acc);
                    }
                    __m512 scores = _mm512_sub_ps(
                        _mm512_mul_ps(dot_acc, _mm512_set1_ps(2.0f)),
                        _mm512_loadu_ps(cn + k));

                    __mmask16 gt_mask = _mm512_cmp_ps_mask(scores, best_vals, _MM_CMPINT_GT);
                    best_vals = _mm512_mask_blend_ps(gt_mask, best_vals, scores);
                    __m512i new_idxs = _mm512_add_epi32(
                        _mm512_set1_epi32(k),
                        _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0));
                    best_idxs = _mm512_mask_blend_epi32(gt_mask, best_idxs, new_idxs);
                }

                float max_val = _mm512_reduce_max_ps(best_vals);
                __mmask16 mask = _mm512_cmp_ps_mask(best_vals, _mm512_set1_ps(max_val), _MM_CMPINT_EQ);
                alignas(64) int idx_arr[16];
                _mm512_store_si512(reinterpret_cast<__m512i*>(idx_arr), best_idxs);
                code[m] = static_cast<uint8_t>(idx_arr[__builtin_ctz(mask)]);
            }
        } else if (dim_sub == 16 && !codebooks_t_.empty()) {
            const float* ct = codebooks_t_.data() + m * 16 * ksub_;
            const float* cn = centroid_norms_.data() + m * ksub_;

            #pragma omp parallel for schedule(static, 256)
            for (size_t i = 0; i < n; ++i) {
                const float* vec_sub = x + i * d + m * 16;
                uint8_t* code = codes_.data() + old_size + i * M_;

                __m512 best_vals = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
                __m512i best_idxs = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

                for (size_t k = 0; k + 15 < ksub_; k += 16) {
                    __m512 dot_acc = _mm512_setzero_ps();
                    for (size_t j = 0; j < 16; ++j) {
                        __m512 vj = _mm512_set1_ps(vec_sub[j]);
                        __m512 cj = _mm512_loadu_ps(ct + j * ksub_ + k);
                        dot_acc = _mm512_fmadd_ps(vj, cj, dot_acc);
                    }
                    __m512 scores = _mm512_sub_ps(
                        _mm512_mul_ps(dot_acc, _mm512_set1_ps(2.0f)),
                        _mm512_loadu_ps(cn + k));

                    __mmask16 gt_mask = _mm512_cmp_ps_mask(scores, best_vals, _MM_CMPINT_GT);
                    best_vals = _mm512_mask_blend_ps(gt_mask, best_vals, scores);
                    __m512i new_idxs = _mm512_add_epi32(
                        _mm512_set1_epi32(k),
                        _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0));
                    best_idxs = _mm512_mask_blend_epi32(gt_mask, best_idxs, new_idxs);
                }

                float max_val = _mm512_reduce_max_ps(best_vals);
                __mmask16 mask = _mm512_cmp_ps_mask(best_vals, _mm512_set1_ps(max_val), _MM_CMPINT_EQ);
                alignas(64) int idx_arr[16];
                _mm512_store_si512(reinterpret_cast<__m512i*>(idx_arr), best_idxs);
                code[m] = static_cast<uint8_t>(idx_arr[__builtin_ctz(mask)]);
            }
        } else
#endif
        {
            const float* centroids = codebooks_.data() + m * ksub_ * dim_sub;

#ifdef __AVX2__
            if (dim_sub == 8 && !codebooks_t_.empty()) {
                const float* ct = codebooks_t_.data() + m * 8 * ksub_;
                const float* cn = centroid_norms_.data() + m * ksub_;

                #pragma omp parallel for schedule(static, 256)
                for (size_t i = 0; i < n; ++i) {
                    const float* vec_sub = x + i * d + m * 8;
                    uint8_t* code = codes_.data() + old_size + i * M_;

                    size_t best_k = 0;
                    float best_s = -std::numeric_limits<float>::infinity();

                    for (size_t k = 0; k + 7 < ksub_; k += 8) {
                        __m256 dot_acc = _mm256_setzero_ps();
                        for (size_t j = 0; j < 8; ++j) {
                            __m256 vj = _mm256_set1_ps(vec_sub[j]);
                            __m256 cj = _mm256_loadu_ps(ct + j * ksub_ + k);
                            dot_acc = _mm256_fmadd_ps(vj, cj, dot_acc);
                        }
                        __m256 scores = _mm256_sub_ps(
                            _mm256_mul_ps(dot_acc, _mm256_set1_ps(2.0f)),
                            _mm256_loadu_ps(cn + k));

                        alignas(32) float buf[8];
                        _mm256_store_ps(buf, scores);
                        for (int j = 0; j < 8; ++j) {
                            if (buf[j] > best_s) {
                                best_s = buf[j];
                                best_k = k + j;
                            }
                        }
                    }
                    for (size_t k = (ksub_ / 8) * 8; k < ksub_; ++k) {
                        __m256 c = _mm256_loadu_ps(centroids + k * 8);
                        __m256 v = _mm256_loadu_ps(vec_sub);
                        __m256 diff = _mm256_sub_ps(v, c);
                        __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
                        float dist = hsum_avx2(sq);
                        float s = -dist;
                        if (s > best_s) {
                            best_s = s;
                            best_k = k;
                        }
                    }
                    code[m] = static_cast<uint8_t>(best_k);
                }
            } else if (dim_sub == 16 && !codebooks_t_.empty()) {
                const float* ct = codebooks_t_.data() + m * 16 * ksub_;
                const float* cn = centroid_norms_.data() + m * ksub_;

                #pragma omp parallel for schedule(static, 256)
                for (size_t i = 0; i < n; ++i) {
                    const float* vec_sub = x + i * d + m * 16;
                    uint8_t* code = codes_.data() + old_size + i * M_;

                    size_t best_k = 0;
                    float best_s = -std::numeric_limits<float>::infinity();

                    for (size_t k = 0; k + 7 < ksub_; k += 8) {
                        __m256 dot_acc = _mm256_setzero_ps();
                        for (size_t j = 0; j < 16; ++j) {
                            __m256 vj = _mm256_set1_ps(vec_sub[j]);
                            __m256 cj = _mm256_loadu_ps(ct + j * ksub_ + k);
                            dot_acc = _mm256_fmadd_ps(vj, cj, dot_acc);
                        }
                        __m256 scores = _mm256_sub_ps(
                            _mm256_mul_ps(dot_acc, _mm256_set1_ps(2.0f)),
                            _mm256_loadu_ps(cn + k));

                        alignas(32) float buf[8];
                        _mm256_store_ps(buf, scores);
                        for (int j = 0; j < 8; ++j) {
                            if (buf[j] > best_s) {
                                best_s = buf[j];
                                best_k = k + j;
                            }
                        }
                    }
                    for (size_t k = (ksub_ / 8) * 8; k < ksub_; ++k) {
                        __m256 v0 = _mm256_loadu_ps(vec_sub);
                        __m256 v1 = _mm256_loadu_ps(vec_sub + 8);
                        __m256 c0 = _mm256_loadu_ps(centroids + k * 16);
                        __m256 c1 = _mm256_loadu_ps(centroids + k * 16 + 8);
                        __m256 diff0 = _mm256_sub_ps(v0, c0);
                        __m256 diff1 = _mm256_sub_ps(v1, c1);
                        __m256 sq0 = _mm256_fmadd_ps(diff0, diff0, _mm256_setzero_ps());
                        __m256 sq1 = _mm256_fmadd_ps(diff1, diff1, _mm256_setzero_ps());
                        float dist = hsum_avx2(_mm256_add_ps(sq0, sq1));
                        float s = -dist;
                        if (s > best_s) {
                            best_s = s;
                            best_k = k;
                        }
                    }
                    code[m] = static_cast<uint8_t>(best_k);
                }
            } else if (dim_sub == 8) {
                #pragma omp parallel for schedule(static, 256)
                for (size_t i = 0; i < n; ++i) {
                    const float* vec_sub = x + i * d + m * 8;
                    uint8_t* code = codes_.data() + old_size + i * M_;
                    __m256 v = _mm256_loadu_ps(vec_sub);
                    size_t best_k = 0;
                    float best_dist = std::numeric_limits<float>::max();

                    for (size_t k = 0; k < ksub_; ++k) {
                        __m256 c = _mm256_loadu_ps(centroids + k * 8);
                        __m256 diff = _mm256_sub_ps(v, c);
                        __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
                        float dist = hsum_avx2(sq);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }

                    code[m] = static_cast<uint8_t>(best_k);
                }
            } else if (dim_sub == 16) {
                #pragma omp parallel for schedule(static, 256)
                for (size_t i = 0; i < n; ++i) {
                    const float* vec_sub = x + i * d + m * 16;
                    uint8_t* code = codes_.data() + old_size + i * M_;
                    __m256 v0 = _mm256_loadu_ps(vec_sub);
                    __m256 v1 = _mm256_loadu_ps(vec_sub + 8);
                    size_t best_k = 0;
                    float best_dist = std::numeric_limits<float>::max();

                    for (size_t k = 0; k < ksub_; ++k) {
                        __m256 c0 = _mm256_loadu_ps(centroids + k * 16);
                        __m256 c1 = _mm256_loadu_ps(centroids + k * 16 + 8);
                        __m256 diff0 = _mm256_sub_ps(v0, c0);
                        __m256 diff1 = _mm256_sub_ps(v1, c1);
                        __m256 sq0 = _mm256_fmadd_ps(diff0, diff0, _mm256_setzero_ps());
                        __m256 sq1 = _mm256_fmadd_ps(diff1, diff1, _mm256_setzero_ps());
                        float dist = hsum_avx2(_mm256_add_ps(sq0, sq1));
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }

                    code[m] = static_cast<uint8_t>(best_k);
                }
            } else
#endif
            {
                #pragma omp parallel for schedule(static, 256)
                for (size_t i = 0; i < n; ++i) {
                    const float* vec_sub = x + i * d + m * dim_sub;
                    uint8_t* code = codes_.data() + old_size + i * M_;

                    size_t best_k = 0;
                    float best_dist = std::numeric_limits<float>::max();
                    for (size_t k = 0; k < ksub_; ++k) {
                        float dist = distance::compute_l2_distance(vec_sub, centroids + k * dim_sub, dim_sub);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }

                    code[m] = static_cast<uint8_t>(best_k);
                }
            }
        }
    }

    codes_dirty_ = true;
}

void IndexPQ::search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const {
    if (codes_dirty_) {
        build_transposed_codes();
    }

    size_t dim_sub = d / M_;

    #pragma omp parallel
    {
        std::vector<float> all_dists(ntotal);
        std::vector<std::pair<float, size_t>> heap;
        heap.reserve(k + 1);

        #pragma omp for schedule(dynamic, 1)
        for (size_t q = 0; q < n; ++q) {
            const float* query = x + q * d;

            float dis_table_buf[16 * 256];
            float* dis_table = dis_table_buf;

#ifdef __AVX512F__
            if (dim_sub == 8 && !codebooks_t_.empty()) {
                for (size_t m = 0; m < M_; ++m) {
                    const float* query_sub = query + m * 8;
                    const float* ct = codebooks_t_.data() + m * 8 * ksub_;
                    const float* cn = centroid_norms_.data() + m * ksub_;

                    __m256 qv = _mm256_loadu_ps(query_sub);
                    __m256 q_sq = _mm256_fmadd_ps(qv, qv, _mm256_setzero_ps());
                    float q_norm = hsum_avx2(q_sq);

                    for (size_t c = 0; c + 15 < ksub_; c += 16) {
                        __m512 dot_acc = _mm512_setzero_ps();
                        for (size_t j = 0; j < 8; ++j) {
                            __m512 vj = _mm512_set1_ps(query_sub[j]);
                            __m512 cj = _mm512_loadu_ps(ct + j * ksub_ + c);
                            dot_acc = _mm512_fmadd_ps(vj, cj, dot_acc);
                        }
                        __m512 dist = _mm512_sub_ps(
                            _mm512_add_ps(_mm512_set1_ps(q_norm), _mm512_loadu_ps(cn + c)),
                            _mm512_mul_ps(dot_acc, _mm512_set1_ps(2.0f)));
                        _mm512_storeu_ps(dis_table + m * ksub_ + c, dist);
                    }
                    for (size_t c = (ksub_ / 16) * 16; c < ksub_; ++c) {
                        const float* centroids = codebooks_.data() + m * ksub_ * 8;
                        __m256 cv = _mm256_loadu_ps(centroids + c * 8);
                        __m256 diff = _mm256_sub_ps(qv, cv);
                        __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
                        dis_table[m * ksub_ + c] = hsum_avx2(sq);
                    }
                }
            } else if (dim_sub == 16 && !codebooks_t_.empty()) {
                for (size_t m = 0; m < M_; ++m) {
                    const float* query_sub = query + m * 16;
                    const float* ct = codebooks_t_.data() + m * 16 * ksub_;
                    const float* cn = centroid_norms_.data() + m * ksub_;

                    float q_norm = 0.0f;
                    for (size_t j = 0; j < 16; ++j) q_norm += query_sub[j] * query_sub[j];

                    for (size_t c = 0; c + 15 < ksub_; c += 16) {
                        __m512 dot_acc = _mm512_setzero_ps();
                        for (size_t j = 0; j < 16; ++j) {
                            __m512 vj = _mm512_set1_ps(query_sub[j]);
                            __m512 cj = _mm512_loadu_ps(ct + j * ksub_ + c);
                            dot_acc = _mm512_fmadd_ps(vj, cj, dot_acc);
                        }
                        __m512 dist = _mm512_sub_ps(
                            _mm512_add_ps(_mm512_set1_ps(q_norm), _mm512_loadu_ps(cn + c)),
                            _mm512_mul_ps(dot_acc, _mm512_set1_ps(2.0f)));
                        _mm512_storeu_ps(dis_table + m * ksub_ + c, dist);
                    }
                    for (size_t c = (ksub_ / 16) * 16; c < ksub_; ++c) {
                        const float* centroids = codebooks_.data() + m * ksub_ * 16;
                        __m256 qv0 = _mm256_loadu_ps(query_sub);
                        __m256 qv1 = _mm256_loadu_ps(query_sub + 8);
                        __m256 cv0 = _mm256_loadu_ps(centroids + c * 16);
                        __m256 cv1 = _mm256_loadu_ps(centroids + c * 16 + 8);
                        __m256 diff0 = _mm256_sub_ps(qv0, cv0);
                        __m256 diff1 = _mm256_sub_ps(qv1, cv1);
                        __m256 sq0 = _mm256_fmadd_ps(diff0, diff0, _mm256_setzero_ps());
                        __m256 sq1 = _mm256_fmadd_ps(diff1, diff1, _mm256_setzero_ps());
                        dis_table[m * ksub_ + c] = hsum_avx2(_mm256_add_ps(sq0, sq1));
                    }
                }
            } else
#elif defined(__AVX2__)
            if (dim_sub == 8) {
                for (size_t m = 0; m < M_; ++m) {
                    const float* query_sub = query + m * 8;
                    const float* centroids = codebooks_.data() + m * ksub_ * 8;
                    __m256 qv = _mm256_loadu_ps(query_sub);
                    for (size_t c = 0; c < ksub_; ++c) {
                        __m256 cv = _mm256_loadu_ps(centroids + c * 8);
                        __m256 diff = _mm256_sub_ps(qv, cv);
                        __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
                        dis_table[m * ksub_ + c] = hsum_avx2(sq);
                    }
                }
            } else if (dim_sub == 16) {
                for (size_t m = 0; m < M_; ++m) {
                    const float* query_sub = query + m * 16;
                    const float* centroids = codebooks_.data() + m * ksub_ * 16;
                    __m256 qv0 = _mm256_loadu_ps(query_sub);
                    __m256 qv1 = _mm256_loadu_ps(query_sub + 8);
                    for (size_t c = 0; c < ksub_; ++c) {
                        __m256 cv0 = _mm256_loadu_ps(centroids + c * 16);
                        __m256 cv1 = _mm256_loadu_ps(centroids + c * 16 + 8);
                        __m256 diff0 = _mm256_sub_ps(qv0, cv0);
                        __m256 diff1 = _mm256_sub_ps(qv1, cv1);
                        __m256 sq0 = _mm256_fmadd_ps(diff0, diff0, _mm256_setzero_ps());
                        __m256 sq1 = _mm256_fmadd_ps(diff1, diff1, _mm256_setzero_ps());
                        dis_table[m * ksub_ + c] = hsum_avx2(_mm256_add_ps(sq0, sq1));
                    }
                }
            } else
#endif
            {
                for (size_t m = 0; m < M_; ++m) {
                    const float* query_sub = query + m * dim_sub;
                    const float* centroids = codebooks_.data() + m * ksub_ * dim_sub;
                    for (size_t c = 0; c < ksub_; ++c) {
                        dis_table[m * ksub_ + c] = distance::compute_l2_distance(query_sub, centroids + c * dim_sub, dim_sub);
                    }
                }
            }

            if (!codes_transposed_.empty()) {
#ifdef __AVX512F__
                if (M_ == 16) {
                    heap.clear();
                    const size_t N = ntotal;
                    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
                    size_t i = 0;
                    for (; i + 31 < N; i += 32) {
                        __m512 d0 = _mm512_setzero_ps(); __m512 d1 = _mm512_setzero_ps();
                        __m512 d2 = _mm512_setzero_ps(); __m512 d3 = _mm512_setzero_ps();
                        __m512 d4 = _mm512_setzero_ps(); __m512 d5 = _mm512_setzero_ps();
                        __m512 d6 = _mm512_setzero_ps(); __m512 d7 = _mm512_setzero_ps();
                        __m512 e0 = _mm512_setzero_ps(); __m512 e1 = _mm512_setzero_ps();
                        __m512 e2 = _mm512_setzero_ps(); __m512 e3 = _mm512_setzero_ps();
                        __m512 e4 = _mm512_setzero_ps(); __m512 e5 = _mm512_setzero_ps();
                        __m512 e6 = _mm512_setzero_ps(); __m512 e7 = _mm512_setzero_ps();
                        for (size_t m = 0; m < 16; ++m) {
                            const float* tab = dis_table + m * ksub_;
                            const uint8_t* cm = codes_transposed_.data() + m * N;
                            __m128i c16_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm + i));
                            __m128i c16_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm + i + 16));
                            __m512i c32_a = _mm512_cvtepu8_epi32(c16_a);
                            __m512i c32_b = _mm512_cvtepu8_epi32(c16_b);
                            __m512 ga = _mm512_i32gather_ps(c32_a, tab, 4);
                            __m512 gb = _mm512_i32gather_ps(c32_b, tab, 4);
                            switch (m >> 1) {
                            case 0: d0=_mm512_add_ps(d0,ga); e0=_mm512_add_ps(e0,gb); break;
                            case 1: d1=_mm512_add_ps(d1,ga); e1=_mm512_add_ps(e1,gb); break;
                            case 2: d2=_mm512_add_ps(d2,ga); e2=_mm512_add_ps(e2,gb); break;
                            case 3: d3=_mm512_add_ps(d3,ga); e3=_mm512_add_ps(e3,gb); break;
                            case 4: d4=_mm512_add_ps(d4,ga); e4=_mm512_add_ps(e4,gb); break;
                            case 5: d5=_mm512_add_ps(d5,ga); e5=_mm512_add_ps(e5,gb); break;
                            case 6: d6=_mm512_add_ps(d6,ga); e6=_mm512_add_ps(e6,gb); break;
                            case 7: d7=_mm512_add_ps(d7,ga); e7=_mm512_add_ps(e7,gb); break;
                            }
                        }
                        d0=_mm512_add_ps(d0,d1); d2=_mm512_add_ps(d2,d3);
                        d4=_mm512_add_ps(d4,d5); d6=_mm512_add_ps(d6,d7);
                        d0=_mm512_add_ps(d0,d2); d4=_mm512_add_ps(d4,d6);
                        __m512 ta=_mm512_add_ps(d0,d4);
                        e0=_mm512_add_ps(e0,e1); e2=_mm512_add_ps(e2,e3);
                        e4=_mm512_add_ps(e4,e5); e6=_mm512_add_ps(e6,e7);
                        e0=_mm512_add_ps(e0,e2); e4=_mm512_add_ps(e4,e6);
                        __m512 tb=_mm512_add_ps(e0,e4);
                        alignas(64) float ba[16], bb[16];
                        _mm512_store_ps(ba, ta); _mm512_store_ps(bb, tb);
                        for (int j = 0; j < 16; ++j) {
                            if (heap.size() < k) { heap.push_back({ba[j], i+j}); std::push_heap(heap.begin(), heap.end(), cmp); }
                            else if (ba[j] < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={ba[j], i+j}; std::push_heap(heap.begin(), heap.end(), cmp); }
                        }
                        for (int j = 0; j < 16; ++j) {
                            if (heap.size() < k) { heap.push_back({bb[j], i+16+j}); std::push_heap(heap.begin(), heap.end(), cmp); }
                            else if (bb[j] < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={bb[j], i+16+j}; std::push_heap(heap.begin(), heap.end(), cmp); }
                        }
                    }
                    for (; i + 15 < N; i += 16) {
                        __m512 d0 = _mm512_setzero_ps(); __m512 d1 = _mm512_setzero_ps();
                        __m512 d2 = _mm512_setzero_ps(); __m512 d3 = _mm512_setzero_ps();
                        __m512 d4 = _mm512_setzero_ps(); __m512 d5 = _mm512_setzero_ps();
                        __m512 d6 = _mm512_setzero_ps(); __m512 d7 = _mm512_setzero_ps();
                        for (size_t m = 0; m < 16; ++m) {
                            const float* tab = dis_table + m * ksub_;
                            const uint8_t* cm = codes_transposed_.data() + m * N;
                            __m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm + i));
                            __m512i c32 = _mm512_cvtepu8_epi32(c16);
                            __m512 g = _mm512_i32gather_ps(c32, tab, 4);
                            switch (m >> 1) {
                            case 0: d0=_mm512_add_ps(d0,g); break;
                            case 1: d1=_mm512_add_ps(d1,g); break;
                            case 2: d2=_mm512_add_ps(d2,g); break;
                            case 3: d3=_mm512_add_ps(d3,g); break;
                            case 4: d4=_mm512_add_ps(d4,g); break;
                            case 5: d5=_mm512_add_ps(d5,g); break;
                            case 6: d6=_mm512_add_ps(d6,g); break;
                            case 7: d7=_mm512_add_ps(d7,g); break;
                            }
                        }
                        d0=_mm512_add_ps(d0,d1); d2=_mm512_add_ps(d2,d3);
                        d4=_mm512_add_ps(d4,d5); d6=_mm512_add_ps(d6,d7);
                        d0=_mm512_add_ps(d0,d2); d4=_mm512_add_ps(d4,d6);
                        __m512 total=_mm512_add_ps(d0,d4);
                        alignas(64) float buf[16];
                        _mm512_store_ps(buf, total);
                        for (int j = 0; j < 16; ++j) {
                            if (heap.size() < k) { heap.push_back({buf[j], i+j}); std::push_heap(heap.begin(), heap.end(), cmp); }
                            else if (buf[j] < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={buf[j], i+j}; std::push_heap(heap.begin(), heap.end(), cmp); }
                        }
                    }
                    for (; i < N; ++i) {
                        float dist = 0.0f;
                        for (size_t m = 0; m < 16; ++m) dist += dis_table[m * ksub_ + codes_transposed_[m * N + i]];
                        if (heap.size() < k) { heap.push_back({dist, i}); std::push_heap(heap.begin(), heap.end(), cmp); }
                        else if (dist < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={dist, i}; std::push_heap(heap.begin(), heap.end(), cmp); }
                    }
                } else if (M_ == 8) {
                    heap.clear();
                    const size_t N = ntotal;
                    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
                    size_t i = 0;
                    for (; i + 31 < N; i += 32) {
                        __m512 d0 = _mm512_setzero_ps(); __m512 d1 = _mm512_setzero_ps();
                        __m512 d2 = _mm512_setzero_ps(); __m512 d3 = _mm512_setzero_ps();
                        __m512 e0 = _mm512_setzero_ps(); __m512 e1 = _mm512_setzero_ps();
                        __m512 e2 = _mm512_setzero_ps(); __m512 e3 = _mm512_setzero_ps();
                        for (size_t m = 0; m < 8; ++m) {
                            const float* tab = dis_table + m * ksub_;
                            const uint8_t* cm = codes_transposed_.data() + m * N;
                            __m128i c16_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm + i));
                            __m128i c16_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm + i + 16));
                            __m512i c32_a = _mm512_cvtepu8_epi32(c16_a);
                            __m512i c32_b = _mm512_cvtepu8_epi32(c16_b);
                            __m512 ga = _mm512_i32gather_ps(c32_a, tab, 4);
                            __m512 gb = _mm512_i32gather_ps(c32_b, tab, 4);
                            switch (m >> 1) {
                            case 0: d0=_mm512_add_ps(d0,ga); e0=_mm512_add_ps(e0,gb); break;
                            case 1: d1=_mm512_add_ps(d1,ga); e1=_mm512_add_ps(e1,gb); break;
                            case 2: d2=_mm512_add_ps(d2,ga); e2=_mm512_add_ps(e2,gb); break;
                            case 3: d3=_mm512_add_ps(d3,ga); e3=_mm512_add_ps(e3,gb); break;
                            }
                        }
                        d0=_mm512_add_ps(d0,d1); d2=_mm512_add_ps(d2,d3);
                        __m512 ta=_mm512_add_ps(d0,d2);
                        e0=_mm512_add_ps(e0,e1); e2=_mm512_add_ps(e2,e3);
                        __m512 tb=_mm512_add_ps(e0,e2);
                        alignas(64) float ba[16], bb[16];
                        _mm512_store_ps(ba, ta); _mm512_store_ps(bb, tb);
                        for (int j = 0; j < 16; ++j) {
                            if (heap.size() < k) { heap.push_back({ba[j], i+j}); std::push_heap(heap.begin(), heap.end(), cmp); }
                            else if (ba[j] < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={ba[j], i+j}; std::push_heap(heap.begin(), heap.end(), cmp); }
                        }
                        for (int j = 0; j < 16; ++j) {
                            if (heap.size() < k) { heap.push_back({bb[j], i+16+j}); std::push_heap(heap.begin(), heap.end(), cmp); }
                            else if (bb[j] < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={bb[j], i+16+j}; std::push_heap(heap.begin(), heap.end(), cmp); }
                        }
                    }
                    for (; i + 15 < N; i += 16) {
                        __m512 d0 = _mm512_setzero_ps(); __m512 d1 = _mm512_setzero_ps();
                        __m512 d2 = _mm512_setzero_ps(); __m512 d3 = _mm512_setzero_ps();
                        for (size_t m = 0; m < 8; ++m) {
                            const float* tab = dis_table + m * ksub_;
                            const uint8_t* cm = codes_transposed_.data() + m * N;
                            __m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm + i));
                            __m512i c32 = _mm512_cvtepu8_epi32(c16);
                            __m512 g = _mm512_i32gather_ps(c32, tab, 4);
                            switch (m >> 1) {
                            case 0: d0=_mm512_add_ps(d0,g); break;
                            case 1: d1=_mm512_add_ps(d1,g); break;
                            case 2: d2=_mm512_add_ps(d2,g); break;
                            case 3: d3=_mm512_add_ps(d3,g); break;
                            }
                        }
                        d0=_mm512_add_ps(d0,d1); d2=_mm512_add_ps(d2,d3);
                        __m512 total=_mm512_add_ps(d0,d2);
                        alignas(64) float buf[16];
                        _mm512_store_ps(buf, total);
                        for (int j = 0; j < 16; ++j) {
                            if (heap.size() < k) { heap.push_back({buf[j], i+j}); std::push_heap(heap.begin(), heap.end(), cmp); }
                            else if (buf[j] < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={buf[j], i+j}; std::push_heap(heap.begin(), heap.end(), cmp); }
                        }
                    }
                    for (; i < N; ++i) {
                        float dist = 0.0f;
                        for (size_t m = 0; m < 8; ++m) dist += dis_table[m * ksub_ + codes_transposed_[m * N + i]];
                        if (heap.size() < k) { heap.push_back({dist, i}); std::push_heap(heap.begin(), heap.end(), cmp); }
                        else if (dist < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={dist, i}; std::push_heap(heap.begin(), heap.end(), cmp); }
                    }
                } else
#endif
                {
#ifdef __AVX512F__
                    const uint8_t* codes_m0 = codes_transposed_.data();
                    const float* tab0 = dis_table;
                    size_t ii = 0;
                    for (; ii + 31 < ntotal; ii += 32) {
                        __m128i c16_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes_m0 + ii));
                        __m128i c16_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes_m0 + ii + 16));
                        __m512i c32_0 = _mm512_cvtepu8_epi32(c16_0);
                        __m512i c32_1 = _mm512_cvtepu8_epi32(c16_1);
                        _mm512_storeu_ps(all_dists.data() + ii, _mm512_i32gather_ps(c32_0, tab0, 4));
                        _mm512_storeu_ps(all_dists.data() + ii + 16, _mm512_i32gather_ps(c32_1, tab0, 4));
                    }
                    for (; ii + 15 < ntotal; ii += 16) {
                        __m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes_m0 + ii));
                        __m512i c32 = _mm512_cvtepu8_epi32(c16);
                        _mm512_storeu_ps(all_dists.data() + ii, _mm512_i32gather_ps(c32, tab0, 4));
                    }
                    for (; ii + 7 < ntotal; ii += 8) {
                        __m128i c8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes_m0 + ii));
                        __m256i c32_8 = _mm256_cvtepu8_epi32(c8);
                        _mm256_storeu_ps(all_dists.data() + ii, _mm256_i32gather_ps(tab0, c32_8, 4));
                    }
                    for (; ii < ntotal; ++ii) all_dists[ii] = tab0[codes_m0[ii]];
                    for (size_t m = 1; m < M_; m += 4) {
                        int n_sub = (int)std::min((size_t)4, M_ - m);
                        const uint8_t* cm[4]; const float* tb[4];
                        for (int s = 0; s < n_sub; ++s) { cm[s] = codes_transposed_.data() + (m + s) * ntotal; tb[s] = dis_table + (m + s) * ksub_; }
                        size_t j = 0;
                        for (; j + 31 < ntotal; j += 32) {
                            __m512 acc0 = _mm512_setzero_ps(); __m512 acc1 = _mm512_setzero_ps();
                            for (int s = 0; s < n_sub; ++s) {
                                __m128i c16_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm[s] + j));
                                __m128i c16_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm[s] + j + 16));
                                acc0 = _mm512_add_ps(acc0, _mm512_i32gather_ps(_mm512_cvtepu8_epi32(c16_0), tb[s], 4));
                                acc1 = _mm512_add_ps(acc1, _mm512_i32gather_ps(_mm512_cvtepu8_epi32(c16_1), tb[s], 4));
                            }
                            __m512 cur0 = _mm512_loadu_ps(all_dists.data() + j);
                            __m512 cur1 = _mm512_loadu_ps(all_dists.data() + j + 16);
                            _mm512_storeu_ps(all_dists.data() + j, _mm512_add_ps(cur0, acc0));
                            _mm512_storeu_ps(all_dists.data() + j + 16, _mm512_add_ps(cur1, acc1));
                        }
                        for (; j + 15 < ntotal; j += 16) {
                            __m512 acc = _mm512_setzero_ps();
                            for (int s = 0; s < n_sub; ++s) {
                                __m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cm[s] + j));
                                acc = _mm512_add_ps(acc, _mm512_i32gather_ps(_mm512_cvtepu8_epi32(c16), tb[s], 4));
                            }
                            __m512 cur = _mm512_loadu_ps(all_dists.data() + j);
                            _mm512_storeu_ps(all_dists.data() + j, _mm512_add_ps(cur, acc));
                        }
                        for (; j < ntotal; ++j) {
                            float v = 0.0f;
                            for (int s = 0; s < n_sub; ++s) v += tb[s][cm[s][j]];
                            all_dists[j] += v;
                        }
                    }
#else
                    const uint8_t* codes_m0 = codes_transposed_.data();
                    const float* tab0 = dis_table;
                    size_t ii = 0;
#ifdef __AVX2__
                    for (; ii + 7 < ntotal; ii += 8) {
                        __m128i c8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes_m0 + ii));
                        __m256i c32 = _mm256_cvtepu8_epi32(c8);
                        _mm256_storeu_ps(all_dists.data() + ii, _mm256_i32gather_ps(tab0, c32, 4));
                    }
#endif
                    for (; ii < ntotal; ++ii) all_dists[ii] = tab0[codes_m0[ii]];
                    for (size_t m = 1; m < M_; m += 2) {
                        int n_sub = (int)std::min((size_t)2, M_ - m);
                        const uint8_t* cm[2]; const float* tb[2];
                        for (int s = 0; s < n_sub; ++s) { cm[s] = codes_transposed_.data() + (m + s) * ntotal; tb[s] = dis_table + (m + s) * ksub_; }
                        size_t j = 0;
#ifdef __AVX2__
                        for (; j + 7 < ntotal; j += 8) {
                            __m256 acc = _mm256_setzero_ps();
                            for (int s = 0; s < n_sub; ++s) {
                                __m128i c8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(cm[s] + j));
                                __m256i c32 = _mm256_cvtepu8_epi32(c8);
                                acc = _mm256_add_ps(acc, _mm256_i32gather_ps(tb[s], c32, 4));
                            }
                            __m256 cur = _mm256_loadu_ps(all_dists.data() + j);
                            _mm256_storeu_ps(all_dists.data() + j, _mm256_add_ps(cur, acc));
                        }
#endif
                        for (; j < ntotal; ++j) {
                            float v = 0.0f;
                            for (int s = 0; s < n_sub; ++s) v += tb[s][cm[s][j]];
                            all_dists[j] += v;
                        }
                    }
#endif
                    heap.clear();
                    for (size_t ii = 0; ii < ntotal; ++ii) {
                        if (heap.size() < k) { heap.push_back({all_dists[ii], ii}); std::push_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; }); }
                        else if (all_dists[ii] < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; }); heap.back()={all_dists[ii], ii}; std::push_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; }); }
                    }
                }
            } else {
                const uint8_t* codes = codes_.data();
                size_t code_size = M_;
                heap.clear();
                auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
                for (size_t i = 0; i < ntotal; ++i) {
                    const uint8_t* code = codes + i * code_size;
                    float dist = 0.0f;
                    for (size_t m = 0; m < code_size; ++m) dist += dis_table[m * ksub_ + code[m]];
                    if (heap.size() < k) { heap.push_back({dist, i}); std::push_heap(heap.begin(), heap.end(), cmp); }
                    else if (dist < heap.front().first) { std::pop_heap(heap.begin(), heap.end(), cmp); heap.back()={dist, i}; std::push_heap(heap.begin(), heap.end(), cmp); }
                }
            }

            std::sort(heap.begin(), heap.end());

            for (size_t i = 0; i < k && i < heap.size(); ++i) {
                distances[q * k + i] = heap[i].first;
                labels[q * k + i] = heap[i].second;
            }
            for (size_t i = heap.size(); i < k; ++i) {
                distances[q * k + i] = std::numeric_limits<float>::max();
                labels[q * k + i] = ntotal;
            }
        }
    }
}

}
}
