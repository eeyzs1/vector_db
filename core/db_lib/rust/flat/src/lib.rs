//! Flat L2 index implementation for vector database
//! 
//! This module provides a brute-force search index using L2 (Euclidean) distance.
//! It supports both single-threaded and parallel search for improved performance.
//! 
//! # Example
//! 
//! ```rust
//! let index = FlatIndex::new(128);
//! index.add_buf(&some_numpy_array);
//! let (distances, labels) = index.search(query_vector, 10);
//! ```

use pyo3::prelude::*;
use rayon::prelude::*;
use vectordb_core::distance::compute_l2_distance;
use vectordb_core::VectorStorage;

fn validate_buffer_contiguous(buffer: &pyo3::buffer::PyBuffer<f32>) -> PyResult<()> {
    if (buffer.buf_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array is not properly aligned for f32",
        ));
    }

    let shape = buffer.shape();
    let strides = buffer.strides();
    let item_size = std::mem::size_of::<f32>() as isize;

    if shape.len() == 2 {
        let expected_stride_1 = item_size;
        let expected_stride_0 = (shape[1] as isize) * item_size;
        if strides.len() < 2
            || strides[1] != expected_stride_1
            || strides[0] != expected_stride_0
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input array must be C-contiguous. Use numpy.ascontiguousarray() to convert.",
            ));
        }
    }

    Ok(())
}

/// Flat L2 index for vector similarity search using brute-force approach
/// 
/// This index calculates exact nearest neighbors by comparing the query vector
/// against every vector in the database using L2 (Euclidean) distance.
/// It supports parallel search for better performance with large datasets.
#[pyclass]
struct FlatIndex {
    /// Vector storage holding all the indexed vectors
    storage: VectorStorage,
}

#[pymethods]
impl FlatIndex {
    #[new]
    fn new(dimension: usize) -> Self {
        Self {
            storage: VectorStorage::new(dimension),
        }
    }

    fn add(&mut self, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let first_dim = vectors[0].len();
        if self.storage.dimension() == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dimension not set, use new(dimension) instead",
            ));
        } else if self.storage.dimension() != first_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All vectors must have the same dimension",
            ));
        }

        let num_vectors = vectors.len();
        let mut flat_data = Vec::with_capacity(num_vectors * self.storage.dimension());
        for vec in vectors {
            if vec.len() != self.storage.dimension() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "All vectors must have the same dimension",
                ));
            }
            flat_data.extend(vec);
        }

        self.storage.add(num_vectors, &flat_data);

        Ok(())
    }

    fn add_buf(&mut self, buffer: &Bound<'_, PyAny>) -> PyResult<()> {
        use pyo3::buffer::PyBuffer;

        let buffer = PyBuffer::<f32>::get(buffer)?;
        let dimensions = buffer.dimensions();

        if dimensions != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input must be a 2D array",
            ));
        }

        let shape = buffer.shape();
        let num_vectors = shape[0];
        let dim = shape[1];

        if num_vectors == 0 {
            return Ok(());
        }

        if self.storage.dimension() == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dimension not set, use new(dimension) instead",
            ));
        } else if self.storage.dimension() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dimension mismatch",
            ));
        }

        validate_buffer_contiguous(&buffer)?;

        let buffer_ptr = buffer.buf_ptr() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(buffer_ptr, num_vectors * dim) };
        self.storage.add(num_vectors, slice);

        Ok(())
    }

    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<(Vec<f32>, Vec<i64>)> {
        if self.storage.size() == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        if query.len() != self.storage.dimension() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Query vector dimension mismatch",
            ));
        }

        self.search_single(&query, k)
    }

    fn search_buf(&self, buffer: &Bound<'_, PyAny>, k: usize) -> PyResult<(Vec<f32>, Vec<i64>)> {
        use pyo3::buffer::PyBuffer;

        if self.storage.size() == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let buffer = PyBuffer::<f32>::get(buffer)?;
        let dimensions = buffer.dimensions();

        if dimensions != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input must be a 2D array",
            ));
        }

        let shape = buffer.shape();
        let rows = shape[0];
        let cols = shape[1];

        if rows != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input must be a 2D array with shape (1, dimension)",
            ));
        }

        let dim = cols;

        if dim != self.storage.dimension() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Query vector dimension mismatch",
            ));
        }

        validate_buffer_contiguous(&buffer)?;

        let buffer_ptr = buffer.buf_ptr() as *const f32;
        let query_vec = unsafe { std::slice::from_raw_parts(buffer_ptr, dim) }.to_vec();

        self.search_single(&query_vec, k)
    }

    fn search_batch_buf(&self, buffer: &Bound<'_, PyAny>, k: usize) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<i64>>)> {
        use pyo3::buffer::PyBuffer;

        if self.storage.size() == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let buffer = PyBuffer::<f32>::get(buffer)?;
        let dimensions = buffer.dimensions();

        if dimensions != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input must be a 2D array",
            ));
        }

        let shape = buffer.shape();
        let num_queries = shape[0];
        let dim = shape[1];

        if dim != self.storage.dimension() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Query vector dimension mismatch",
            ));
        }

        validate_buffer_contiguous(&buffer)?;

        let buffer_ptr = buffer.buf_ptr() as *const f32;
        let buffer_slice = unsafe { std::slice::from_raw_parts(buffer_ptr, num_queries * dim) };

        let mut all_distances = Vec::with_capacity(num_queries);
        let mut all_labels = Vec::with_capacity(num_queries);

        let num_threads = self.calculate_optimal_search_threads();

        if num_threads > 1 && num_queries > 10 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to create thread pool: {}", e)
                ))?;
            
            let results: Vec<PyResult<(Vec<f32>, Vec<i64>)>> = pool.install(|| {
                (0..num_queries)
                    .into_par_iter()
                    .map(|i| {
                        let start = i * dim;
                        let end = start + dim;
                        let query = &buffer_slice[start..end];
                        self.search_single(query, k)
                    })
                    .collect()
            });

            for result in results {
                let (distances, labels) = result?;
                all_distances.push(distances);
                all_labels.push(labels);
            }
        } else {
            for i in 0..num_queries {
                let start = i * dim;
                let end = start + dim;
                let query = &buffer_slice[start..end];
                let (distances, labels) = self.search_single(query, k)?;
                all_distances.push(distances);
                all_labels.push(labels);
            }
        }

        Ok((all_distances, all_labels))
    }

    #[getter]
    fn ntotal(&self) -> usize {
        self.storage.size()
    }

    fn dimension(&self) -> usize {
        self.storage.dimension()
    }

    fn save(&self, path: &str) -> PyResult<()> {
        use std::fs::File;
        use std::io::Write;
        
        let bytes = self.storage.save_to_bytes();
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        
        let storage = VectorStorage::load_from_bytes(&bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(Self { storage })
    }
}

impl FlatIndex {
    #[inline]
    fn insert_top_k(
        &self,
        distances: &mut [f32],
        labels: &mut [i64],
        k: usize,
        dist: f32,
        idx: i64,
    ) {
        if dist >= distances[k - 1] {
            return;
        }

        let mut left = 0;
        let mut right = k;
        while left < right {
            let mid = (left + right) / 2;
            if dist < distances[mid] {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        if left < k {
            if k - left > 1 {
                distances[left..k].rotate_right(1);
                labels[left..k].rotate_right(1);
            }
            distances[left] = dist;
            labels[left] = idx;
        }
    }

    #[inline]
    fn calculate_optimal_search_threads(&self) -> usize {
        let num_threads = num_cpus::get();

        let num_threads = if self.storage.size() > 500000 {
            std::cmp::min(num_threads, 10)
        } else if self.storage.size() > 100000 {
            std::cmp::min(num_threads, 6)
        } else if self.storage.size() > 10000 {
            std::cmp::min(num_threads, 4)
        } else {
            1
        };

        if self.storage.dimension() > 256 {
            std::cmp::max(1, num_threads / 2)
        } else {
            num_threads
        }
    }

    fn search_single(&self, query: &[f32], k: usize) -> PyResult<(Vec<f32>, Vec<i64>)> {
        let actual_k = std::cmp::min(k, self.storage.size());

        if actual_k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut top_distances = vec![f32::MAX; actual_k];
        let mut top_labels = vec![0i64; actual_k];

        let mut vec_ptr = self.storage.data().as_ptr();
        let dim = self.storage.dimension();

        let mut i = 0;
        while i + 7 < self.storage.size() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    vec_ptr.add(8 * dim) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
                std::arch::x86_64::_mm_prefetch(
                    vec_ptr.add(16 * dim) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }

            let dist0 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr, dim) },
            );
            let dist1 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr.add(dim), dim) },
            );
            let dist2 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr.add(2 * dim), dim) },
            );
            let dist3 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr.add(3 * dim), dim) },
            );
            let dist4 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr.add(4 * dim), dim) },
            );
            let dist5 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr.add(5 * dim), dim) },
            );
            let dist6 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr.add(6 * dim), dim) },
            );
            let dist7 = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr.add(7 * dim), dim) },
            );

            self.insert_top_k(&mut top_distances, &mut top_labels, k, dist0, i as i64);
            self.insert_top_k(
                &mut top_distances,
                &mut top_labels,
                k,
                dist1,
                (i + 1) as i64,
            );
            self.insert_top_k(
                &mut top_distances,
                &mut top_labels,
                k,
                dist2,
                (i + 2) as i64,
            );
            self.insert_top_k(
                &mut top_distances,
                &mut top_labels,
                k,
                dist3,
                (i + 3) as i64,
            );
            self.insert_top_k(
                &mut top_distances,
                &mut top_labels,
                k,
                dist4,
                (i + 4) as i64,
            );
            self.insert_top_k(
                &mut top_distances,
                &mut top_labels,
                k,
                dist5,
                (i + 5) as i64,
            );
            self.insert_top_k(
                &mut top_distances,
                &mut top_labels,
                k,
                dist6,
                (i + 6) as i64,
            );
            self.insert_top_k(
                &mut top_distances,
                &mut top_labels,
                k,
                dist7,
                (i + 7) as i64,
            );

            vec_ptr = unsafe { vec_ptr.add(8 * dim) };
            i += 8;
        }

        while i < self.storage.size() {
            let dist = compute_l2_distance(
                query,
                unsafe { std::slice::from_raw_parts(vec_ptr, dim) },
            );

            self.insert_top_k(&mut top_distances, &mut top_labels, k, dist, i as i64);
            vec_ptr = unsafe { vec_ptr.add(dim) };
        }

        Ok((top_distances, top_labels))
    }
}

#[pymodule]
fn _flat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlatIndex>()?;
    m.add("__version__", "1.0.0")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(num_vectors: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..num_vectors)
            .map(|_| (0..dim).map(|_| rng.gen()).collect())
            .collect()
    }

    #[test]
    fn test_index_creation() {
        let index = FlatIndex::new(128);
        assert_eq!(index.dimension(), 128);
        assert_eq!(index.size(), 0);
    }

    #[test]
    fn test_add_vectors() {
        let mut index = FlatIndex::new(64);
        let vectors = generate_random_vectors(100, 64);
        
        index.add(vectors).unwrap();
        assert_eq!(index.size(), 100);
    }

    #[test]
    fn test_search_vectors() {
        let mut index = FlatIndex::new(32);
        let vectors = generate_random_vectors(100, 32);
        
        index.add(vectors.clone()).unwrap();
        
        // Search for the first vector
        let (labels, distances) = index.search(vectors[0].clone(), 5).unwrap();
        
        // The first result should be the vector itself
        assert_eq!(labels[0], 0);
        assert!(distances[0] < 1e-6);
    }

    #[test]
    fn test_search_returns_valid_labels() {
        let mut index = FlatIndex::new(64);
        let vectors = generate_random_vectors(100, 64);
        index.add(vectors).unwrap();
        
        let query = generate_random_vectors(1, 64);
        let (labels, _) = index.search(query[0].clone(), 10).unwrap();
        
        for &label in labels.iter() {
            assert!(label >= 0);
            assert!(label < 100);
        }
    }

    #[test]
    fn test_distances_are_sorted() {
        let mut index = FlatIndex::new(64);
        let vectors = generate_random_vectors(100, 64);
        index.add(vectors).unwrap();
        
        let query = generate_random_vectors(1, 64);
        let (_, distances) = index.search(query[0].clone(), 10).unwrap();
        
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i-1]);
        }
    }
}
