use pyo3::prelude::*;
use vectordb_core::distance;
use vectordb_core::VectorStorage;
use rand::Rng;
use std::collections::HashSet;

#[pyclass]
struct IndexPQ {
    storage: VectorStorage,
    M: usize,
    nbits: usize,
    ksub: usize,
    codebooks: Vec<f32>,
    codes: Vec<u8>,
}

#[pymethods]
impl IndexPQ {
    #[new]
    fn new(dimension: usize, M: Option<usize>, nbits: Option<usize>) -> PyResult<Self> {
        let M = M.unwrap_or(8);
        let nbits = nbits.unwrap_or(8);
        if dimension % M != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dimension must be divisible by M",
            ));
        }
        let ksub = 1 << nbits;
        Ok(Self {
            storage: VectorStorage::new(dimension),
            M,
            nbits,
            ksub,
            codebooks: Vec::new(),
            codes: Vec::new(),
        })
    }

    fn train(&mut self, x: Vec<Vec<f32>>) -> PyResult<()> {
        let n = x.len();
        let dim = self.storage.dimension();
        let dim_sub = dim / self.M;

        self.codebooks.resize(self.M * self.ksub * dim_sub, 0.0);

        for m in 0..self.M {
            self.train_kmeans(m, &x, n, dim_sub);
        }

        Ok(())
    }

    fn add(&mut self, x: Vec<Vec<f32>>) -> PyResult<()> {
        let n = x.len();
        if n == 0 {
            return Ok(());
        }

        if self.codebooks.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Index not trained"));
        }

        let dim = self.storage.dimension();
        let _old_total = self.storage.size();
        let mut flat_data = Vec::with_capacity(n * dim);
        for vec in x {
            flat_data.extend(vec);
        }
        self.storage.add(n, &flat_data);

        let old_size = self.codes.len();
        self.codes.resize(old_size + n * self.M, 0);

        for i in 0..n {
            let vec = &flat_data[i * dim..(i + 1) * dim];
            let mut code = vec![0u8; self.M];
            Self::encode_vector_static(vec, &mut code, &self.codebooks, self.M, self.ksub, dim);
            self.codes[old_size + i * self.M..old_size + (i + 1) * self.M].copy_from_slice(&code);
        }

        Ok(())
    }

    fn search(&self, x: Vec<Vec<f32>>, k: usize) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<i64>>)> {
        let dim = self.storage.dimension();
        let dim_sub = dim / self.M;
        let mut all_distances = Vec::with_capacity(x.len());
        let mut all_labels = Vec::with_capacity(x.len());

        for query in x {
            let mut query_code = vec![0u8; self.M];
            Self::encode_vector_static(&query, &mut query_code, &self.codebooks, self.M, self.ksub, dim);

            let mut results = Vec::with_capacity(self.storage.size());
            for i in 0..self.storage.size() {
                let code = &self.codes[i * self.M..(i + 1) * self.M];
                let dist = self.compute_distance(&query_code, code, dim_sub);
                results.push((dist, i as i64));
            }
            results.sort_by(|a, b| a.0.total_cmp(&b.0));

            let take = std::cmp::min(k, results.len());
            let mut distances = Vec::with_capacity(k);
            let mut labels = Vec::with_capacity(k);
            for (dist, idx) in results.iter().take(take) {
                distances.push(*dist);
                labels.push(*idx);
            }
            for _ in take..k {
                distances.push(0.0);
                labels.push(0);
            }

            all_distances.push(distances);
            all_labels.push(labels);
        }

        Ok((all_distances, all_labels))
    }

    #[getter]
    fn ntotal(&self) -> usize {
        self.storage.size()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.storage.dimension()
    }

    #[getter]
    fn M(&self) -> usize {
        self.M
    }

    #[getter]
    fn nbits(&self) -> usize {
        self.nbits
    }
}

impl IndexPQ {
    fn train_kmeans(&mut self, m: usize, x: &[Vec<f32>], n: usize, dim_sub: usize) {
        let max_iter = 25;
        let mut rng = rand::thread_rng();
        let mut selected = HashSet::new();

        let mut centroids = vec![0.0f32; self.ksub * dim_sub];
        for i in 0..self.ksub {
            let mut idx;
            loop {
                idx = rng.gen_range(0..n);
                if !selected.contains(&idx) {
                    selected.insert(idx);
                    break;
                }
            }
            let start = m * dim_sub;
            centroids[i * dim_sub..(i + 1) * dim_sub].copy_from_slice(&x[idx][start..start + dim_sub]);
        }

        let mut new_centroids = vec![0.0f32; self.ksub * dim_sub];
        let mut counts = vec![0usize; self.ksub];

        for _ in 0..max_iter {
            new_centroids.iter_mut().for_each(|x| *x = 0.0);
            counts.iter_mut().for_each(|x| *x = 0);

            for vec in x {
                let vec_sub = &vec[m * dim_sub..(m + 1) * dim_sub];
                let mut best_k = 0;
                let mut best_dist = f32::MAX;

                for k in 0..self.ksub {
                    let centroid = &centroids[k * dim_sub..(k + 1) * dim_sub];
                    let dist = distance::compute_l2_distance(vec_sub, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                counts[best_k] += 1;
                for j in 0..dim_sub {
                    new_centroids[best_k * dim_sub + j] += vec_sub[j];
                }
            }

            let mut max_shift = 0.0f32;
            for k in 0..self.ksub {
                if counts[k] > 0 {
                    let inv_count = 1.0 / counts[k] as f32;
                    for j in 0..dim_sub {
                        let old_val = centroids[k * dim_sub + j];
                        let new_val = new_centroids[k * dim_sub + j] * inv_count;
                        centroids[k * dim_sub + j] = new_val;
                        let shift = (new_val - old_val).abs();
                        if shift > max_shift {
                            max_shift = shift;
                        }
                    }
                } else {
                    let mut largest_k = 0;
                    let mut largest_count = 0usize;
                    for kk in 0..self.ksub {
                        if counts[kk] > largest_count {
                            largest_count = counts[kk];
                            largest_k = kk;
                        }
                    }
                    if largest_count > 1 {
                        let mut rng = rand::thread_rng();
                        let perturbation: f32 = rng.gen_range(0.01..0.1);
                        for j in 0..dim_sub {
                            centroids[k * dim_sub + j] = centroids[largest_k * dim_sub + j] + perturbation;
                        }
                    }
                }
            }

            if max_shift < 1e-4 {
                break;
            }
        }

        let start = m * self.ksub * dim_sub;
        self.codebooks[start..start + self.ksub * dim_sub].copy_from_slice(&centroids);
    }

    fn encode_vector_static(vec: &[f32], code: &mut [u8], codebooks: &[f32], M: usize, ksub: usize, dim: usize) {
        let dim_sub = dim / M;

        for m in 0..M {
            let vec_sub = &vec[m * dim_sub..(m + 1) * dim_sub];
            let centroids = &codebooks[m * ksub * dim_sub..(m + 1) * ksub * dim_sub];

            let mut best_k = 0;
            let mut best_dist = f32::MAX;

            for k in 0..ksub {
                let centroid = &centroids[k * dim_sub..(k + 1) * dim_sub];
                let dist = distance::compute_l2_distance(vec_sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_k = k;
                }
            }

            code[m] = best_k as u8;
        }
    }

    fn compute_distance(&self, code_a: &[u8], code_b: &[u8], dim_sub: usize) -> f32 {
        let mut dist = 0.0f32;

        for m in 0..self.M {
            let k_a = code_a[m] as usize;
            let k_b = code_b[m] as usize;
            let centroids = &self.codebooks[m * self.ksub * dim_sub..(m + 1) * self.ksub * dim_sub];
            let c_a = &centroids[k_a * dim_sub..(k_a + 1) * dim_sub];
            let c_b = &centroids[k_b * dim_sub..(k_b + 1) * dim_sub];
            dist += distance::compute_l2_distance(c_a, c_b);
        }

        dist
    }
}

#[pymodule]
fn vectordb_pq(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexPQ>()?;
    Ok(())
}
