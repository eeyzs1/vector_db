use std::arch::x86_64::*;

pub mod distance {
    use super::*;

    #[inline]
    pub fn compute_l2_distance(query: &[f32], vec: &[f32]) -> f32 {
        let dim = query.len();
        unsafe {
            #[cfg(target_feature = "avx512f")]
            {
                compute_l2_avx512(query, vec, dim)
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                #[cfg(target_arch = "x86_64")]
                {
                    if dim == 128 {
                        compute_l2_avx2_128(query, vec)
                    } else {
                        compute_l2_avx2(query, vec, dim)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    compute_l2_scalar(query, vec, dim)
                }
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    unsafe fn compute_l2_avx2_128(query: &[f32], vec: &[f32]) -> f32 {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let q0 = _mm256_loadu_ps(query.as_ptr());
        let v0 = _mm256_loadu_ps(vec.as_ptr());
        let diff0 = _mm256_sub_ps(q0, v0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        let q1 = _mm256_loadu_ps(query.as_ptr().add(8));
        let v1 = _mm256_loadu_ps(vec.as_ptr().add(8));
        let diff1 = _mm256_sub_ps(q1, v1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        let q2 = _mm256_loadu_ps(query.as_ptr().add(16));
        let v2 = _mm256_loadu_ps(vec.as_ptr().add(16));
        let diff2 = _mm256_sub_ps(q2, v2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        let q3 = _mm256_loadu_ps(query.as_ptr().add(24));
        let v3 = _mm256_loadu_ps(vec.as_ptr().add(24));
        let diff3 = _mm256_sub_ps(q3, v3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);

        let q4 = _mm256_loadu_ps(query.as_ptr().add(32));
        let v4 = _mm256_loadu_ps(vec.as_ptr().add(32));
        let diff4 = _mm256_sub_ps(q4, v4);
        sum0 = _mm256_fmadd_ps(diff4, diff4, sum0);

        let q5 = _mm256_loadu_ps(query.as_ptr().add(40));
        let v5 = _mm256_loadu_ps(vec.as_ptr().add(40));
        let diff5 = _mm256_sub_ps(q5, v5);
        sum1 = _mm256_fmadd_ps(diff5, diff5, sum1);

        let q6 = _mm256_loadu_ps(query.as_ptr().add(48));
        let v6 = _mm256_loadu_ps(vec.as_ptr().add(48));
        let diff6 = _mm256_sub_ps(q6, v6);
        sum2 = _mm256_fmadd_ps(diff6, diff6, sum2);

        let q7 = _mm256_loadu_ps(query.as_ptr().add(56));
        let v7 = _mm256_loadu_ps(vec.as_ptr().add(56));
        let diff7 = _mm256_sub_ps(q7, v7);
        sum3 = _mm256_fmadd_ps(diff7, diff7, sum3);

        let q8 = _mm256_loadu_ps(query.as_ptr().add(64));
        let v8 = _mm256_loadu_ps(vec.as_ptr().add(64));
        let diff8 = _mm256_sub_ps(q8, v8);
        sum0 = _mm256_fmadd_ps(diff8, diff8, sum0);

        let q9 = _mm256_loadu_ps(query.as_ptr().add(72));
        let v9 = _mm256_loadu_ps(vec.as_ptr().add(72));
        let diff9 = _mm256_sub_ps(q9, v9);
        sum1 = _mm256_fmadd_ps(diff9, diff9, sum1);

        let q10 = _mm256_loadu_ps(query.as_ptr().add(80));
        let v10 = _mm256_loadu_ps(vec.as_ptr().add(80));
        let diff10 = _mm256_sub_ps(q10, v10);
        sum2 = _mm256_fmadd_ps(diff10, diff10, sum2);

        let q11 = _mm256_loadu_ps(query.as_ptr().add(88));
        let v11 = _mm256_loadu_ps(vec.as_ptr().add(88));
        let diff11 = _mm256_sub_ps(q11, v11);
        sum3 = _mm256_fmadd_ps(diff11, diff11, sum3);

        let q12 = _mm256_loadu_ps(query.as_ptr().add(96));
        let v12 = _mm256_loadu_ps(vec.as_ptr().add(96));
        let diff12 = _mm256_sub_ps(q12, v12);
        sum0 = _mm256_fmadd_ps(diff12, diff12, sum0);

        let q13 = _mm256_loadu_ps(query.as_ptr().add(104));
        let v13 = _mm256_loadu_ps(vec.as_ptr().add(104));
        let diff13 = _mm256_sub_ps(q13, v13);
        sum1 = _mm256_fmadd_ps(diff13, diff13, sum1);

        let q14 = _mm256_loadu_ps(query.as_ptr().add(112));
        let v14 = _mm256_loadu_ps(vec.as_ptr().add(112));
        let diff14 = _mm256_sub_ps(q14, v14);
        sum2 = _mm256_fmadd_ps(diff14, diff14, sum2);

        let q15 = _mm256_loadu_ps(query.as_ptr().add(120));
        let v15 = _mm256_loadu_ps(vec.as_ptr().add(120));
        let diff15 = _mm256_sub_ps(q15, v15);
        sum3 = _mm256_fmadd_ps(diff15, diff15, sum3);

        let sum_a = _mm256_add_ps(sum0, sum1);
        let sum_b = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum_a, sum_b);

        let shuffled = _mm256_permute2f128_ps(sum, sum, 0x21);
        let summed = _mm256_add_ps(sum, shuffled);
        let summed = _mm256_hadd_ps(summed, summed);
        let summed = _mm256_hadd_ps(summed, summed);
        _mm256_cvtss_f32(summed)
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    unsafe fn compute_l2_avx2(query: &[f32], vec: &[f32], dim: usize) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= dim {
            let q = _mm256_loadu_ps(query.as_ptr().add(i));
            let v = _mm256_loadu_ps(vec.as_ptr().add(i));
            let diff = _mm256_sub_ps(q, v);
            sum = _mm256_fmadd_ps(diff, diff, sum);
            i += 8;
        }

        let mut result = horizontal_sum_avx2(sum);

        while i < dim {
            let diff = query[i] - vec[i];
            result += diff * diff;
            i += 1;
        }

        result
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
        let shuffled = _mm256_permute2f128_ps(v, v, 0x21);
        let summed = _mm256_add_ps(v, shuffled);
        let summed = _mm256_hadd_ps(summed, summed);
        let summed = _mm256_hadd_ps(summed, summed);
        _mm256_cvtss_f32(summed)
    }

    #[inline]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe fn compute_l2_avx512(query: &[f32], vec: &[f32], dim: usize) -> f32 {
        let mut sum = _mm512_setzero_ps();
        let mut i = 0;
        
        while i + 16 <= dim {
            let q = _mm512_loadu_ps(query.as_ptr().add(i));
            let v = _mm512_loadu_ps(vec.as_ptr().add(i));
            let diff = _mm512_sub_ps(q, v);
            sum = _mm512_fmadd_ps(diff, diff, sum);
            i += 16;
        }
        
        let mut total = _mm512_reduce_add_ps(sum);
        
        while i < dim {
            let diff = query[i] - vec[i];
            total += diff * diff;
            i += 1;
        }
        
        total
    }

    #[inline]
    fn compute_l2_scalar(query: &[f32], vec: &[f32], dim: usize) -> f32 {
        let mut dist = 0.0f32;
        for i in 0..dim {
            let diff = query[i] - vec[i];
            dist += diff * diff;
        }
        dist
    }

    #[inline]
    pub fn compute_ip_distance(query: &[f32], vec: &[f32]) -> f32 {
        let dim = query.len();
        unsafe {
            #[cfg(target_feature = "avx512f")]
            {
                compute_ip_avx512(query, vec, dim)
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                #[cfg(target_arch = "x86_64")]
                {
                    if dim == 128 {
                        compute_ip_avx2_128(query, vec)
                    } else {
                        compute_ip_avx2(query, vec, dim)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    compute_ip_scalar(query, vec, dim)
                }
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    unsafe fn compute_ip_avx2_128(query: &[f32], vec: &[f32]) -> f32 {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let q0 = _mm256_loadu_ps(query.as_ptr());
        let v0 = _mm256_loadu_ps(vec.as_ptr());
        sum0 = _mm256_fmadd_ps(q0, v0, sum0);

        let q1 = _mm256_loadu_ps(query.as_ptr().add(8));
        let v1 = _mm256_loadu_ps(vec.as_ptr().add(8));
        sum1 = _mm256_fmadd_ps(q1, v1, sum1);

        let q2 = _mm256_loadu_ps(query.as_ptr().add(16));
        let v2 = _mm256_loadu_ps(vec.as_ptr().add(16));
        sum2 = _mm256_fmadd_ps(q2, v2, sum2);

        let q3 = _mm256_loadu_ps(query.as_ptr().add(24));
        let v3 = _mm256_loadu_ps(vec.as_ptr().add(24));
        sum3 = _mm256_fmadd_ps(q3, v3, sum3);

        let q4 = _mm256_loadu_ps(query.as_ptr().add(32));
        let v4 = _mm256_loadu_ps(vec.as_ptr().add(32));
        sum0 = _mm256_fmadd_ps(q4, v4, sum0);

        let q5 = _mm256_loadu_ps(query.as_ptr().add(40));
        let v5 = _mm256_loadu_ps(vec.as_ptr().add(40));
        sum1 = _mm256_fmadd_ps(q5, v5, sum1);

        let q6 = _mm256_loadu_ps(query.as_ptr().add(48));
        let v6 = _mm256_loadu_ps(vec.as_ptr().add(48));
        sum2 = _mm256_fmadd_ps(q6, v6, sum2);

        let q7 = _mm256_loadu_ps(query.as_ptr().add(56));
        let v7 = _mm256_loadu_ps(vec.as_ptr().add(56));
        sum3 = _mm256_fmadd_ps(q7, v7, sum3);

        let q8 = _mm256_loadu_ps(query.as_ptr().add(64));
        let v8 = _mm256_loadu_ps(vec.as_ptr().add(64));
        sum0 = _mm256_fmadd_ps(q8, v8, sum0);

        let q9 = _mm256_loadu_ps(query.as_ptr().add(72));
        let v9 = _mm256_loadu_ps(vec.as_ptr().add(72));
        sum1 = _mm256_fmadd_ps(q9, v9, sum1);

        let q10 = _mm256_loadu_ps(query.as_ptr().add(80));
        let v10 = _mm256_loadu_ps(vec.as_ptr().add(80));
        sum2 = _mm256_fmadd_ps(q10, v10, sum2);

        let q11 = _mm256_loadu_ps(query.as_ptr().add(88));
        let v11 = _mm256_loadu_ps(vec.as_ptr().add(88));
        sum3 = _mm256_fmadd_ps(q11, v11, sum3);

        let q12 = _mm256_loadu_ps(query.as_ptr().add(96));
        let v12 = _mm256_loadu_ps(vec.as_ptr().add(96));
        sum0 = _mm256_fmadd_ps(q12, v12, sum0);

        let q13 = _mm256_loadu_ps(query.as_ptr().add(104));
        let v13 = _mm256_loadu_ps(vec.as_ptr().add(104));
        sum1 = _mm256_fmadd_ps(q13, v13, sum1);

        let q14 = _mm256_loadu_ps(query.as_ptr().add(112));
        let v14 = _mm256_loadu_ps(vec.as_ptr().add(112));
        sum2 = _mm256_fmadd_ps(q14, v14, sum2);

        let q15 = _mm256_loadu_ps(query.as_ptr().add(120));
        let v15 = _mm256_loadu_ps(vec.as_ptr().add(120));
        sum3 = _mm256_fmadd_ps(q15, v15, sum3);

        let sum_a = _mm256_add_ps(sum0, sum1);
        let sum_b = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum_a, sum_b);

        let shuffled = _mm256_permute2f128_ps(sum, sum, 0x21);
        let summed = _mm256_add_ps(sum, shuffled);
        let summed = _mm256_hadd_ps(summed, summed);
        let summed = _mm256_hadd_ps(summed, summed);
        _mm256_cvtss_f32(summed)
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    unsafe fn compute_ip_avx2(query: &[f32], vec: &[f32], dim: usize) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= dim {
            let q = _mm256_loadu_ps(query.as_ptr().add(i));
            let v = _mm256_loadu_ps(vec.as_ptr().add(i));
            sum = _mm256_fmadd_ps(q, v, sum);
            i += 8;
        }

        let mut result = horizontal_sum_avx2(sum);

        while i < dim {
            result += query[i] * vec[i];
            i += 1;
        }

        result
    }

    #[inline]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe fn compute_ip_avx512(query: &[f32], vec: &[f32], dim: usize) -> f32 {
        let mut sum = _mm512_setzero_ps();
        let mut i = 0;
        
        while i + 16 <= dim {
            let q = _mm512_loadu_ps(query.as_ptr().add(i));
            let v = _mm512_loadu_ps(vec.as_ptr().add(i));
            sum = _mm512_fmadd_ps(q, v, sum);
            i += 16;
        }
        
        let mut total = _mm512_reduce_add_ps(sum);
        
        while i < dim {
            total += query[i] * vec[i];
            i += 1;
        }
        
        total
    }

    #[inline]
    fn compute_ip_scalar(query: &[f32], vec: &[f32], dim: usize) -> f32 {
        let mut dist = 0.0f32;
        for i in 0..dim {
            dist += query[i] * vec[i];
        }
        dist
    }
}

pub trait IndexTrait {
    fn add_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), String>;
    fn search_vectors(&self, queries: Vec<Vec<f32>>, k: usize) -> Result<(Vec<Vec<f32>>, Vec<Vec<i64>>), String>;
    fn ntotal(&self) -> usize;
    fn dimension(&self) -> usize;
}

pub struct VectorStorage {
    vectors: Vec<f32>,
    dimension: usize,
    size: usize,
}

impl VectorStorage {
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension,
            size: 0,
        }
    }

    pub fn add(&mut self, n: usize, x: &[f32]) {
        let total_elements = n * self.dimension;
        let current_len = self.vectors.len();
        self.vectors.resize(current_len + total_elements, 0.0f32);
        self.vectors[current_len..current_len + total_elements].copy_from_slice(x);
        self.size += n;
    }

    pub fn data(&self) -> &[f32] {
        &self.vectors
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn get_vector(&self, idx: usize) -> &[f32] {
        let dim = self.dimension;
        &self.vectors[idx * dim..(idx + 1) * dim]
    }

    pub fn save_to_bytes(&self) -> Vec<u8> {
        let header_size = 16;
        let vector_bytes = self.vectors.len() * 4;
        let mut result = Vec::with_capacity(header_size + vector_bytes);

        result.extend_from_slice(&self.dimension.to_le_bytes());
        result.extend_from_slice(&self.size.to_le_bytes());

        let vector_byte_len = self.vectors.len() * std::mem::size_of::<f32>();
        result.reserve(vector_byte_len);
        unsafe {
            let ptr = result.as_mut_ptr().add(result.len());
            std::ptr::copy_nonoverlapping(
                self.vectors.as_ptr() as *const u8,
                ptr,
                vector_byte_len,
            );
            result.set_len(result.len() + vector_byte_len);
        }

        result
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut offset = 0;

        if bytes.len() < 16 {
            return Err("Invalid data: too short");
        }

        let dimension = usize::from_le_bytes(
            bytes[offset..offset+8].try_into().map_err(|_| "Invalid dimension bytes")?
        );
        offset += 8;

        let size = usize::from_le_bytes(
            bytes[offset..offset+8].try_into().map_err(|_| "Invalid size bytes")?
        );
        offset += 8;

        let expected_bytes = size * dimension * 4;
        if bytes.len() - offset != expected_bytes {
            return Err("Invalid data: length mismatch");
        }

        let num_floats = size * dimension;
        let mut vectors = vec![0.0f32; num_floats];
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().add(offset),
                vectors.as_mut_ptr() as *mut u8,
                num_floats * 4,
            );
        }

        Ok(Self {
            vectors,
            dimension,
            size,
        })
    }
}
