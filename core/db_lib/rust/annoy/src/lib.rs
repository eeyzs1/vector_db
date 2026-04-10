use pyo3::prelude::*;
use vectordb_core::distance;
use vectordb_core::VectorStorage;
use rand::Rng;
use std::collections::HashSet;

struct AnnoyNode {
    index: usize,
    hyperplane_normal: Vec<f32>,
    hyperplane_offset: f32,
    left: Option<Box<AnnoyNode>>,
    right: Option<Box<AnnoyNode>>,
}

impl AnnoyNode {
    fn new() -> Self {
        Self {
            index: 0,
            hyperplane_normal: Vec::new(),
            hyperplane_offset: 0.0,
            left: None,
            right: None,
        }
    }
}

#[pyclass]
struct IndexAnnoy {
    storage: VectorStorage,
    n_trees: usize,
    trees: Vec<Option<Box<AnnoyNode>>>,
    built: bool,
}

#[pymethods]
impl IndexAnnoy {
    #[new]
    fn new(dimension: usize, n_trees: Option<usize>) -> Self {
        let n_trees = n_trees.unwrap_or(10);
        Self {
            storage: VectorStorage::new(dimension),
            n_trees,
            trees: Vec::new(),
            built: false,
        }
    }

    fn add(&mut self, x: Vec<Vec<f32>>) -> PyResult<()> {
        let n = x.len();
        if n == 0 {
            return Ok(());
        }

        let dim = self.storage.dimension();
        let mut flat_data = Vec::with_capacity(n * dim);
        for vec in x {
            flat_data.extend(vec);
        }
        self.storage.add(n, &flat_data);
        self.built = false;

        Ok(())
    }

    fn build(&mut self) -> PyResult<()> {
        if self.storage.size() == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No vectors added yet",
            ));
        }

        let dim = self.storage.dimension();
        let indices: Vec<usize> = (0..self.storage.size()).collect();
        self.trees.clear();
        self.trees.reserve(self.n_trees);
        for _ in 0..self.n_trees {
            self.trees.push(Self::build_tree_static(&indices, dim, &self.storage));
        }
        self.built = true;

        Ok(())
    }

    fn search(&self, x: Vec<Vec<f32>>, k: usize) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<i64>>)> {
        if !self.built {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Index not built. Call build() after adding vectors.",
            ));
        }

        let dim = self.storage.dimension();
        let mut all_distances = Vec::with_capacity(x.len());
        let mut all_labels = Vec::with_capacity(x.len());

        for query in x {
            let mut candidates = HashSet::new();
            for tree in &self.trees {
                if let Some(root) = tree {
                    self.get_candidates(&query, root, &mut candidates);
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
    fn n_trees(&self) -> usize {
        self.n_trees
    }
}

impl IndexAnnoy {
    fn build_tree_static(indices: &[usize], dim: usize, storage: &VectorStorage) -> Option<Box<AnnoyNode>> {
        let mut node = Box::new(AnnoyNode::new());

        if indices.len() <= 1 {
            if indices.len() == 1 {
                node.index = indices[0];
            }
            return Some(node);
        }

        let mut rng = rand::thread_rng();
        let i1 = rng.gen_range(0..indices.len());
        let mut i2 = rng.gen_range(0..indices.len());
        while i1 == i2 {
            i2 = rng.gen_range(0..indices.len());
        }

        let v1 = storage.get_vector(indices[i1]);
        let v2 = storage.get_vector(indices[i2]);

        node.hyperplane_normal = vec![0.0f32; dim];
        for j in 0..dim {
            node.hyperplane_normal[j] = v2[j] - v1[j];
        }

        let mut norm = 0.0;
        for j in 0..dim {
            norm += node.hyperplane_normal[j] * node.hyperplane_normal[j];
        }
        norm = norm.sqrt();

        if norm > 1e-6 {
            for j in 0..dim {
                node.hyperplane_normal[j] /= norm;
            }
        }

        node.hyperplane_offset = 0.0;
        for j in 0..dim {
            node.hyperplane_offset += node.hyperplane_normal[j] * (v1[j] + v2[j]) / 2.0;
        }

        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &idx in indices {
            let v = storage.get_vector(idx);
            let mut dot = 0.0;
            for j in 0..dim {
                dot += node.hyperplane_normal[j] * v[j];
            }
            if dot < node.hyperplane_offset {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        if left_indices.is_empty() {
            std::mem::swap(&mut left_indices, &mut right_indices);
        }
        if right_indices.is_empty() {
            if !left_indices.is_empty() {
                node.index = left_indices[0];
            }
            return Some(node);
        }

        node.left = Self::build_tree_static(&left_indices, dim, storage);
        node.right = Self::build_tree_static(&right_indices, dim, storage);

        Some(node)
    }

    fn get_candidates(&self, query: &[f32], node: &AnnoyNode, candidates: &mut HashSet<usize>) {
        let dim = self.storage.dimension();

        if node.left.is_none() && node.right.is_none() {
            candidates.insert(node.index);
            return;
        }

        let mut dot = 0.0;
        for j in 0..dim {
            dot += node.hyperplane_normal[j] * query[j];
        }

        let (near_child, far_child) = if dot < node.hyperplane_offset {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(child) = near_child {
            self.get_candidates(query, child, candidates);
        }
        if let Some(child) = far_child {
            self.get_candidates(query, child, candidates);
        }
    }
}

#[pymodule]
fn vectordb_annoy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexAnnoy>()?;
    Ok(())
}
