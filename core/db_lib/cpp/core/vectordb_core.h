#pragma once

#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <new>

template <typename T, size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) noexcept {
        free(ptr);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

#ifdef USE_OPENBLAS
extern "C" {
#include <cblas.h>
}
#endif

namespace vectordb {

namespace distance {

inline float compute_l2_distance(const float* query, const float* vec, size_t dim) {
    float dist = 0.0f;

#ifdef __AVX512F__
    if (dim >= 16) {
        size_t i = 0;
        __m512 sum = _mm512_setzero_ps();

        size_t end = dim - 15;
        while (i < end) {
            __m512 q = _mm512_loadu_ps(query + i);
            __m512 v = _mm512_loadu_ps(vec + i);
            __m512 diff = _mm512_sub_ps(q, v);
            sum = _mm512_fmadd_ps(diff, diff, sum);
            i += 16;
        }

        dist = _mm512_reduce_add_ps(sum);

        for (; i < dim; ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
    } else {
        for (size_t i = 0; i < dim; ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
    }
#elif defined(__AVX2__)
    if (dim >= 8) {
        size_t i = 0;
        __m256 sum = _mm256_setzero_ps();

        size_t end = dim - 7;
        while (i < end) {
            __m256 q = _mm256_loadu_ps(query + i);
            __m256 v = _mm256_loadu_ps(vec + i);
            __m256 diff = _mm256_sub_ps(q, v);
            sum = _mm256_fmadd_ps(diff, diff, sum);
            i += 8;
        }

        __m256 shuffled = _mm256_permute2f128_ps(sum, sum, 0x21);
        __m256 summed = _mm256_add_ps(sum, shuffled);
        summed = _mm256_hadd_ps(summed, summed);
        summed = _mm256_hadd_ps(summed, summed);
        dist = _mm256_cvtss_f32(summed);

        for (; i < dim; ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
    } else {
        for (size_t i = 0; i < dim; ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        float diff = query[i] - vec[i];
        dist += diff * diff;
    }
#endif

    return dist;
}

inline float compute_ip_distance(const float* query, const float* vec, size_t dim) {
    float dist = 0.0f;

#ifdef __AVX512F__
    if (dim >= 16) {
        size_t i = 0;
        __m512 sum = _mm512_setzero_ps();

        size_t end = dim - 15;
        while (i < end) {
            __m512 q = _mm512_loadu_ps(query + i);
            __m512 v = _mm512_loadu_ps(vec + i);
            sum = _mm512_fmadd_ps(q, v, sum);
            i += 16;
        }

        dist = _mm512_reduce_add_ps(sum);

        for (; i < dim; ++i) {
            dist += query[i] * vec[i];
        }
    } else {
        for (size_t i = 0; i < dim; ++i) {
            dist += query[i] * vec[i];
        }
    }
#elif defined(__AVX2__)
    if (dim >= 8) {
        size_t i = 0;
        __m256 sum = _mm256_setzero_ps();

        size_t end = dim - 7;
        while (i < end) {
            __m256 q = _mm256_loadu_ps(query + i);
            __m256 v = _mm256_loadu_ps(vec + i);
            sum = _mm256_fmadd_ps(q, v, sum);
            i += 8;
        }

        __m256 shuffled = _mm256_permute2f128_ps(sum, sum, 0x21);
        __m256 summed = _mm256_add_ps(sum, shuffled);
        summed = _mm256_hadd_ps(summed, summed);
        summed = _mm256_hadd_ps(summed, summed);
        dist = _mm256_cvtss_f32(summed);

        for (; i < dim; ++i) {
            dist += query[i] * vec[i];
        }
    } else {
        for (size_t i = 0; i < dim; ++i) {
            dist += query[i] * vec[i];
        }
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        dist += query[i] * vec[i];
    }
#endif

    return dist;
}

#ifdef USE_OPENBLAS
inline void compute_batch_distances_l2(float* distances, const float* queries, size_t nq,
                                        const float* vectors, size_t nv, size_t dim) {
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < nv; ++j) {
            float dot = cblas_sdot(dim, queries + i * dim, 1, vectors + j * dim, 1);
            float norm_q = cblas_sdot(dim, queries + i * dim, 1, queries + i * dim, 1);
            float norm_v = cblas_sdot(dim, vectors + j * dim, 1, vectors + j * dim, 1);
            distances[i * nv + j] = norm_q + norm_v - 2.0f * dot;
        }
    }
}

inline void compute_batch_distances_ip(float* distances, const float* queries, size_t nq,
                                        const float* vectors, size_t nv, size_t dim) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, nv, dim,
                 -1.0f, queries, dim, vectors, dim, 0.0f, distances, nv);
}
#endif

}

inline void normalize_vector(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
}

inline void normalize_vectors(float* vectors, size_t num_vectors, size_t dim) {
    for (size_t i = 0; i < num_vectors; ++i) {
        normalize_vector(vectors + i * dim, dim);
    }
}

class VectorStorage {
protected:
    std::vector<float> xb;
    std::vector<float> xb_transposed;
    size_t d;
    size_t ntotal;
    bool transposed;

    void transpose() {
        if (ntotal == 0 || d == 0) return;
        
        xb_transposed.resize(ntotal * d);
        for (size_t i = 0; i < ntotal; ++i) {
            for (size_t j = 0; j < d; ++j) {
                xb_transposed[j * ntotal + i] = xb[i * d + j];
            }
        }
        transposed = true;
    }

public:
    VectorStorage(size_t dimension) 
        : d(dimension), ntotal(0), transposed(false) {
        size_t reserve_size = std::min(size_t(100000) * dimension, size_t(256) * 1024 * 1024 / sizeof(float));
        xb.reserve(reserve_size);
        xb_transposed.reserve(reserve_size);
    }

    virtual ~VectorStorage() = default;

    virtual void add(size_t n, const float* x) {
        if (d == 0) {
            throw std::invalid_argument("Dimension not set");
        }
        
        size_t old_size = xb.size();
        xb.resize(old_size + n * d);
        std::memcpy(xb.data() + old_size, x, n * d * sizeof(float));
        ntotal += n;
        
        if (transposed) {
            transposed = false;
            xb_transposed.clear();
        }
    }

    const float* data() const { return xb.data(); }
    const float* transposed_data() const { 
        if (!transposed) {
            const_cast<VectorStorage*>(this)->transpose();
        }
        return xb_transposed.data(); 
    }
    bool is_transposed() const { return transposed; }
    size_t get_ntotal() const { return ntotal; }
    size_t get_dimension() const { return d; }
    
    void clear_transposed() {
        transposed = false;
        xb_transposed.clear();
    }

    std::vector<uint8_t> save_to_bytes() const {
        std::vector<uint8_t> result;
        
        size_t d_val = d;
        size_t ntotal_val = ntotal;
        
        result.resize(sizeof(size_t) * 2 + xb.size() * sizeof(float));
        size_t offset = 0;
        
        std::memcpy(result.data() + offset, &d_val, sizeof(size_t));
        offset += sizeof(size_t);
        
        std::memcpy(result.data() + offset, &ntotal_val, sizeof(size_t));
        offset += sizeof(size_t);
        
        std::memcpy(result.data() + offset, xb.data(), xb.size() * sizeof(float));
        
        return result;
    }

    void load_from_bytes(const uint8_t* bytes, size_t length) {
        if (length < sizeof(size_t) * 2) {
            throw std::runtime_error("Invalid data: too short");
        }
        
        size_t offset = 0;
        
        std::memcpy(&d, bytes + offset, sizeof(size_t));
        offset += sizeof(size_t);
        
        std::memcpy(&ntotal, bytes + offset, sizeof(size_t));
        offset += sizeof(size_t);
        
        size_t expected_data_len = ntotal * d * sizeof(float);
        if (length - offset != expected_data_len) {
            throw std::runtime_error("Invalid data: length mismatch");
        }
        
        xb.resize(ntotal * d);
        std::memcpy(xb.data(), bytes + offset, expected_data_len);
        
        transposed = false;
        xb_transposed.clear();
    }
};

class IndexInterface {
public:
    virtual ~IndexInterface() = default;
    
    virtual void add(size_t n, const float* x) = 0;
    virtual void search(size_t n, const float* x, size_t k, float* distances, size_t* labels) const = 0;
    
    virtual void train(size_t n, const float* x) {}
    virtual size_t get_ntotal() const = 0;
    virtual size_t get_dimension() const = 0;
    
    virtual std::vector<uint8_t> save() const { return {}; }
    virtual void load(const uint8_t* /*bytes*/, size_t /*length*/) {}
};

}
