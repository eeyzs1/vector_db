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
                __m512 v = _mm512_loadu_ps(vec);
                size_t best_k = 0;
                float best_dist = std::numeric_limits<float>::max();

                for (size_t k = 0; k < ksub; ++k) {
                    __m512 c = _mm512_loadu_ps(centroids + k * 16);
                    __m512 diff = _mm512_sub_ps(v, c);
                    __m512 sq = _mm512_fmadd_ps(diff, diff, _mm512_setzero_ps());
                    float dist = _mm512_reduce_add_ps(sq);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                float* sum = new_centroids.data() + best_k * 16;
                __m512 ns = _mm512_loadu_ps(sum);
                _mm512_storeu_ps(sum, _mm512_add_ps(ns, v));
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
    train_kmeans_single(dim_sub, ksub_, 25, sub_vectors.data(), n, centroids);
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
        train_kmeans_single(dim_sub, ksub_, 25, sub_vectors.data(), n,
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

    VectorStorage::add(n, x);

    size_t old_size = codes_.size();
    codes_.resize(old_size + n * M_);

    size_t dim_sub = d / M_;

#ifdef __AVX512F__
    if (dim_sub == 8) {
        if (!codebooks_t_.empty()) {
            #pragma omp parallel for schedule(static, 256)
            for (size_t i = 0; i < n; ++i) {
                const float* vec = x + i * d;
                uint8_t* code = codes_.data() + old_size + i * M_;

                for (size_t m = 0; m < M_; ++m) {
                    const float* ct = codebooks_t_.data() + m * 8 * ksub_;
                    const float* cn = centroid_norms_.data() + m * ksub_;

                    __m512 best_scores = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
                    __m512i best_indices = _mm512_setzero_epi32();

                    for (size_t k = 0; k + 15 < ksub_; k += 16) {
                        __m512 dot_acc = _mm512_setzero_ps();
                        for (size_t j = 0; j < 8; ++j) {
                            __m512 vj = _mm512_set1_ps(vec[m * 8 + j]);
                            __m512 cj = _mm512_loadu_ps(ct + j * ksub_ + k);
                            dot_acc = _mm512_fmadd_ps(vj, cj, dot_acc);
                        }
                        __m512 scores = _mm512_sub_ps(_mm512_mul_ps(dot_acc, _mm512_set1_ps(2.0f)), _mm512_loadu_ps(cn + k));

                        __mmask16 better = _mm512_cmp_ps_mask(scores, best_scores, _MM_CMPINT_GT);
                        best_scores = _mm512_mask_blend_ps(better, best_scores, scores);
                        best_indices = _mm512_mask_blend_epi32(better, best_indices, _mm512_add_epi32(_mm512_set1_epi32(k), _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)));
                    }

                    float bs[16]; _mm512_storeu_ps(bs, best_scores);
                    int bi[16]; _mm512_storeu_si512((__m512i*)bi, best_indices);
                    size_t best_k = bi[0]; float best_s = bs[0];
                    for (int j = 1; j < 16; ++j) {
                        if (bs[j] > best_s) { best_s = bs[j]; best_k = bi[j]; }
                    }
                    for (size_t k = (ksub_ / 16) * 16; k < ksub_; ++k) {
                        float dot = 0.0f;
                        for (size_t j = 0; j < 8; ++j) {
                            dot += vec[m * 8 + j] * ct[j * ksub_ + k];
                        }
                        float score = 2.0f * dot - cn[k];
                        if (score > best_s) { best_s = score; best_k = k; }
                    }
                    code[m] = static_cast<uint8_t>(best_k);
                }
            }
        } else {
            #pragma omp parallel for schedule(static, 256)
            for (size_t i = 0; i < n; ++i) {
                const float* vec = x + i * d;
                uint8_t* code = codes_.data() + old_size + i * M_;

                for (size_t m = 0; m < M_; ++m) {
                    const float* centroids = codebooks_.data() + m * ksub_ * 8;
                    __m256 v = _mm256_loadu_ps(vec + m * 8);
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
            }
        }
    } else if (dim_sub == 16) {
        #pragma omp parallel
        {
            for (size_t m = 0; m < M_; ++m) {
                const float* centroids = codebooks_.data() + m * ksub_ * 16;

                #pragma omp for schedule(static, 512)
                for (size_t i = 0; i < n; ++i) {
                    const float* vec_sub = x + i * d + m * 16;
                    uint8_t* code = codes_.data() + old_size + i * M_;

                    __m512 v = _mm512_loadu_ps(vec_sub);
                    size_t best_k = 0;
                    float best_dist = std::numeric_limits<float>::max();

                    for (size_t k = 0; k < ksub_; ++k) {
                        __m512 c = _mm512_loadu_ps(centroids + k * 16);
                        __m512 diff = _mm512_sub_ps(v, c);
                        __m512 sq = _mm512_fmadd_ps(diff, diff, _mm512_setzero_ps());
                        float dist = _mm512_reduce_add_ps(sq);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }

                    code[m] = static_cast<uint8_t>(best_k);
                }
            }
        }
    } else
#elif defined(__AVX2__)
    if (dim_sub == 8) {
        #pragma omp parallel for schedule(static, 256)
        for (size_t i = 0; i < n; ++i) {
            const float* vec = x + i * d;
            uint8_t* code = codes_.data() + old_size + i * M_;

            for (size_t m = 0; m < M_; ++m) {
                const float* centroids = codebooks_.data() + m * ksub_ * 8;
                __m256 v = _mm256_loadu_ps(vec + m * 8);
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
        }
    } else if (dim_sub == 16) {
        #pragma omp parallel for schedule(static, 256)
        for (size_t i = 0; i < n; ++i) {
            const float* vec = x + i * d;
            uint8_t* code = codes_.data() + old_size + i * M_;

            for (size_t m = 0; m < M_; ++m) {
                const float* centroids = codebooks_.data() + m * ksub_ * 16;
                __m256 v0 = _mm256_loadu_ps(vec + m * 16);
                __m256 v1 = _mm256_loadu_ps(vec + m * 16 + 8);
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
        }
    } else
#endif
    {
        #pragma omp parallel for schedule(static, 256)
        for (size_t i = 0; i < n; ++i) {
            const float* vec = x + i * d;
            uint8_t* code = codes_.data() + old_size + i * M_;

            for (size_t m = 0; m < M_; ++m) {
                const float* vec_sub = vec + m * dim_sub;
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
            if (dim_sub == 8) {
                for (size_t m = 0; m < M_; ++m) {
                    const float* query_sub = query + m * 8;
                    const float* centroids = codebooks_.data() + m * ksub_ * 8;
                    __m256 qv = _mm256_loadu_ps(query_sub);
                    for (size_t c = 0; c < ksub_; ++c) {
                        __m256 cv = _mm256_loadu_ps(centroids + c * 8);
                        __m256 diff = _mm256_sub_ps(qv, cv);
                        __m256 sq = _mm256_fmadd_ps(diff, diff, _mm256_setzero_ps());
                        dis_table[m * ksub_ + c] = _mm512_reduce_add_ps(_mm512_castps256_ps512(sq));
                    }
                }
            } else if (dim_sub == 16) {
                for (size_t m = 0; m < M_; ++m) {
                    const float* query_sub = query + m * 16;
                    const float* centroids = codebooks_.data() + m * ksub_ * 16;
                    __m512 qv = _mm512_loadu_ps(query_sub);
                    for (size_t c = 0; c < ksub_; ++c) {
                        __m512 cv = _mm512_loadu_ps(centroids + c * 16);
                        __m512 diff = _mm512_sub_ps(qv, cv);
                        __m512 sq = _mm512_fmadd_ps(diff, diff, _mm512_setzero_ps());
                        dis_table[m * ksub_ + c] = _mm512_reduce_add_ps(sq);
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

            std::fill(all_dists.begin(), all_dists.end(), 0.0f);

            if (!codes_transposed_.empty()) {
#ifdef __AVX512F__
                for (size_t m = 0; m < M_; ++m) {
                    const uint8_t* codes_m = codes_transposed_.data() + m * ntotal;
                    const float* tab = dis_table + m * ksub_;

                    size_t i = 0;
                    for (; i + 15 < ntotal; i += 16) {
                        __m128i code16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes_m + i));
                        __m512i code32_lo = _mm512_cvtepu8_epi32(code16);
                        __m512 gathered_lo = _mm512_i32gather_ps(code32_lo, tab, 4);
                        __m512 current_lo = _mm512_loadu_ps(all_dists.data() + i);
                        _mm512_storeu_ps(all_dists.data() + i, _mm512_add_ps(current_lo, gathered_lo));
                    }
                    for (; i + 7 < ntotal; i += 8) {
                        __m128i code8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes_m + i));
                        __m256i code32 = _mm256_cvtepu8_epi32(code8);
                        __m256 gathered = _mm256_i32gather_ps(tab, code32, 4);
                        __m256 current = _mm256_loadu_ps(all_dists.data() + i);
                        _mm256_storeu_ps(all_dists.data() + i, _mm256_add_ps(current, gathered));
                    }
                    for (; i < ntotal; ++i) {
                        all_dists[i] += tab[codes_m[i]];
                    }
                }
#else
                for (size_t m = 0; m < M_; ++m) {
                    const uint8_t* codes_m = codes_transposed_.data() + m * ntotal;
                    const float* tab = dis_table + m * ksub_;

                    size_t i = 0;
#ifdef __AVX2__
                    for (; i + 7 < ntotal; i += 8) {
                        __m128i code8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes_m + i));
                        __m256i code32 = _mm256_cvtepu8_epi32(code8);
                        __m256 gathered = _mm256_i32gather_ps(tab, code32, 4);
                        __m256 current = _mm256_loadu_ps(all_dists.data() + i);
                        _mm256_storeu_ps(all_dists.data() + i, _mm256_add_ps(current, gathered));
                    }
#endif
                    for (; i < ntotal; ++i) {
                        all_dists[i] += tab[codes_m[i]];
                    }
                }
#endif
            } else {
                const uint8_t* codes = codes_.data();
                size_t code_size = M_;
                for (size_t i = 0; i < ntotal; ++i) {
                    const uint8_t* code = codes + i * code_size;
                    float dist = 0.0f;
                    for (size_t m = 0; m < code_size; ++m) {
                        dist += dis_table[m * ksub_ + code[m]];
                    }
                    all_dists[i] = dist;
                }
            }

            heap.clear();
            for (size_t i = 0; i < ntotal; ++i) {
                if (heap.size() < k) {
                    heap.push_back({all_dists[i], i});
                    std::push_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                } else if (all_dists[i] < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                    heap.back() = {all_dists[i], i};
                    std::push_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
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
