use pyo3::prelude::*;
use vectordb_core::distance;
use vectordb_core::VectorStorage;
use rand::Rng;
use std::collections::HashSet;

#[pyclass]
struct IndexIVF {
    storage: VectorStorage,
    nlist: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    inverted_lists: Vec<Vec<usize>>,
}

#[pymethods]
impl IndexIVF {
    #[new]
    fn new(dimension: usize, nlist: Option<usize>) -> Self {
        let nlist = nlist.unwrap_or(100);
        Self {
            storage: VectorStorage::new(dimension),
            nlist,
            nprobe: 10,
            centroids: Vec::new(),
            inverted_lists: vec![Vec::new(); nlist],
        }
    }

    fn train(&mut self, x: Vec<Vec<f32>>) -> PyResult<()> {
        let n = x.len();
        if n < self.nlist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Training data is too small for nlist clusters",
            ));
        }

        let dim = self.storage.dimension();
        self.centroids.resize(self.nlist * dim, 0.0);

        let mut rng = rand::thread_rng();
        let mut selected = HashSet::new();
        for i in 0..self.nlist {
            let mut idx;
            loop {
                idx = rng.gen_range(0..n);
                if !selected.contains(&idx) {
                    selected.insert(idx);
                    break;
                }
            }
            self.centroids[i * dim..(i + 1) * dim].copy_from_slice(&x[idx]);
        }

        let max_iter = 25;
        let mut new_centroids = vec![0.0f32; self.nlist * dim];
        let mut counts = vec![0usize; self.nlist];
        let mut assignments = vec![0usize; n];

        for _ in 0..max_iter {
            new_centroids.iter_mut().for_each(|x| *x = 0.0);
            counts.iter_mut().for_each(|x| *x = 0);

            for (i, vec) in x.iter().enumerate() {
                let cluster = self.assign_to_cluster(vec);
                assignments[i] = cluster;
                counts[cluster] += 1;
                for j in 0..dim {
                    new_centroids[cluster * dim + j] += vec[j];
                }
            }

            let _converged = true;
            let mut max_shift = 0.0f32;

            for i in 0..self.nlist {
                if counts[i] > 0 {
                    let inv_count = 1.0 / counts[i] as f32;
                    for j in 0..dim {
                        let old_val = self.centroids[i * dim + j];
                        let new_val = new_centroids[i * dim + j] * inv_count;
                        self.centroids[i * dim + j] = new_val;
                        let shift = (new_val - old_val).abs();
                        if shift > max_shift {
                            max_shift = shift;
                        }
                    }
                } else {
                    let mut largest_i = 0;
                    let mut largest_count = 0usize;
                    for ii in 0..self.nlist {
                        if counts[ii] > largest_count {
                            largest_count = counts[ii];
                            largest_i = ii;
                        }
                    }
                    if largest_count > 1 {
                        let mut rng = rand::thread_rng();
                        let perturbation: f32 = rng.gen_range(0.01..0.1);
                        for j in 0..dim {
                            self.centroids[i * dim + j] = self.centroids[largest_i * dim + j] + perturbation;
                        }
                    }
                }
            }

            if max_shift < 1e-4 {
                break;
            }
        }

        Ok(())
    }

    fn add(&mut self, x: Vec<Vec<f32>>) -> PyResult<()> {
        let n = x.len();
        if n == 0 {
            return Ok(());
        }

        let dim = self.storage.dimension();
        let old_total = self.storage.size();
        let mut flat_data = Vec::with_capacity(n * dim);
        for vec in x {
            flat_data.extend(vec);
        }
        self.storage.add(n, &flat_data);

        for i in 0..n {
            let idx = old_total + i;
            let vec = &flat_data[i * dim..(i + 1) * dim];
            let cluster = self.assign_to_cluster(vec);
            self.inverted_lists[cluster].push(idx);
        }

        Ok(())
    }

    #[setter]
    fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe;
    }

    fn search(&self, x: Vec<Vec<f32>>, k: usize) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<i64>>)> {
        if self.centroids.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Index not trained"));
        }

        let dim = self.storage.dimension();
        let mut all_distances = Vec::with_capacity(x.len());
        let mut all_labels = Vec::with_capacity(x.len());

        for query in x {
            let mut cluster_distances = Vec::with_capacity(self.nlist);
            for i in 0..self.nlist {
                let centroid = &self.centroids[i * dim..(i + 1) * dim];
                let dist = distance::compute_l2_distance(&query, centroid);
                cluster_distances.push((dist, i));
            }
            cluster_distances.sort_by(|a, b| a.0.total_cmp(&b.0));

            let nprobe = std::cmp::min(self.nprobe, self.nlist);
            let mut candidates = HashSet::new();
            for (_, cluster) in cluster_distances.iter().take(nprobe) {
                for &idx in &self.inverted_lists[*cluster] {
                    candidates.insert(idx);
                }
            }

            let mut results = Vec::with_capacity(candidates.len());
            for idx in candidates {
                let vec = self.storage.get_vector(idx);
                let dist = distance::compute_l2_distance(&query, vec);
                results.push((dist, idx as i64));
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
    fn nlist(&self) -> usize {
        self.nlist
    }

    #[getter]
    fn nprobe(&self) -> usize {
        self.nprobe
    }
}

impl IndexIVF {
    fn assign_to_cluster(&self, vec: &[f32]) -> usize {
        let dim = self.storage.dimension();
        let mut best_cluster = 0;
        let mut best_dist = f32::MAX;

        for i in 0..self.nlist {
            let centroid = &self.centroids[i * dim..(i + 1) * dim];
            let dist = distance::compute_l2_distance(vec, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_cluster = i;
            }
        }

        best_cluster
    }
}

#[pymodule]
fn vectordb_ivf(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexIVF>()?;
    Ok(())
}
