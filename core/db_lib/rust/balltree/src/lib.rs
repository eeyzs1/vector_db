use pyo3::prelude::*;
use vectordb_core::distance;
use vectordb_core::VectorStorage;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

#[pyclass]
struct BallNode {
    points: Vec<usize>,
    center: Vec<f32>,
    radius: f32,
    left: Option<Box<BallNode>>,
    right: Option<Box<BallNode>>,
}

impl BallNode {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            center: Vec::new(),
            radius: 0.0,
            left: None,
            right: None,
        }
    }
}

#[pyclass]
struct IndexBallTree {
    storage: VectorStorage,
    leaf_size: usize,
    root: Option<Box<BallNode>>,
    built: bool,
}

#[pymethods]
impl IndexBallTree {
    #[new]
    fn new(dimension: usize, leaf_size: Option<usize>) -> Self {
        let leaf_size = leaf_size.unwrap_or(40);
        Self {
            storage: VectorStorage::new(dimension),
            leaf_size,
            root: None,
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

        let indices: Vec<usize> = (0..self.storage.size()).collect();
        self.root = self.build_tree(&indices);
        self.built = true;

        Ok(())
    }

    fn search(&self, x: Vec<Vec<f32>>, k: usize) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<i64>>)> {
        if !self.built {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Index not built. Call build() after adding vectors.",
            ));
        }
        let _dim = self.storage.dimension();
        let mut all_distances = Vec::with_capacity(x.len());
        let mut all_labels = Vec::with_capacity(x.len());

        for query in x {
            let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
            if let Some(root) = &self.root {
                self.search_k_nearest(&query, root, k, &mut heap);
            }

            let mut results = Vec::with_capacity(heap.len());
            while let Some((dist, idx)) = heap.pop() {
                results.push((dist.0, idx as i64));
            }
            results.reverse();

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
    fn leaf_size(&self) -> usize {
        self.leaf_size
    }
}

impl IndexBallTree {
    fn build_tree(&self, indices: &[usize]) -> Option<Box<BallNode>> {
        let dim = self.storage.dimension();
        let mut node = Box::new(BallNode::new());
        node.points = indices.to_vec();

        if indices.len() <= self.leaf_size {
            node.center = vec![0.0f32; dim];
            for &idx in indices {
                let vec = self.storage.get_vector(idx);
                for j in 0..dim {
                    node.center[j] += vec[j];
                }
            }
            let inv_size = 1.0 / indices.len() as f32;
            for j in 0..dim {
                node.center[j] *= inv_size;
            }

            node.radius = 0.0;
            for &idx in indices {
                let vec = self.storage.get_vector(idx);
                let dist = distance::compute_l2_distance(vec, &node.center);
                node.radius = node.radius.max(dist);
            }

            return Some(node);
        }

        let mut mean = vec![0.0f32; dim];
        for &idx in indices {
            let vec = self.storage.get_vector(idx);
            for j in 0..dim {
                mean[j] += vec[j];
            }
        }
        let inv_n = 1.0 / indices.len() as f32;
        for j in 0..dim {
            mean[j] *= inv_n;
        }

        let mut split_dim = 0;
        let mut max_var = 0.0;
        for j in 0..dim {
            let mut var = 0.0;
            for &idx in indices {
                let vec = self.storage.get_vector(idx);
                let diff = vec[j] - mean[j];
                var += diff * diff;
            }
            if var > max_var {
                max_var = var;
                split_dim = j;
            }
        }

        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            let va = self.storage.get_vector(a);
            let vb = self.storage.get_vector(b);
            va[split_dim].total_cmp(&vb[split_dim])
        });

        let median = sorted_indices.len() / 2;
        let left_indices = &sorted_indices[..median];
        let right_indices = &sorted_indices[median..];

        node.left = self.build_tree(left_indices);
        node.right = self.build_tree(right_indices);

        if let (Some(left), Some(right)) = (&node.left, &node.right) {
            node.center = vec![0.0f32; dim];
            for j in 0..dim {
                node.center[j] = (left.center[j] + right.center[j]) / 2.0;
            }
            let d1 = distance::compute_l2_distance(&left.center, &node.center) + left.radius;
            let d2 = distance::compute_l2_distance(&right.center, &node.center) + right.radius;
            node.radius = d1.max(d2);
        }

        Some(node)
    }

    fn search_k_nearest(&self, query: &[f32], node: &BallNode, k: usize, heap: &mut BinaryHeap<(OrderedFloat<f32>, usize)>) {
        let dim = self.storage.dimension();

        if node.points.len() <= self.leaf_size {
            for &idx in &node.points {
                let vec = self.storage.get_vector(idx);
                let dist = distance::compute_l2_distance(query, vec);
                if heap.len() < k {
                    heap.push((OrderedFloat(dist), idx));
                } else if OrderedFloat(dist) < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), idx));
                }
            }
            return;
        }

        let (near_child, far_child, dist_to_far) = if let (Some(left), Some(right)) = (&node.left, &node.right) {
            let dist_left = distance::compute_l2_distance(query, &left.center);
            let dist_right = distance::compute_l2_distance(query, &right.center);
            if dist_left < dist_right {
                (left, right, dist_right)
            } else {
                (right, left, dist_left)
            }
        } else {
            return;
        };

        self.search_k_nearest(query, near_child, k, heap);

        let far_child_radius = far_child.radius;
        if heap.len() < k || OrderedFloat(dist_to_far - far_child_radius) < heap.peek().unwrap().0 {
            self.search_k_nearest(query, far_child, k, heap);
        }
    }
}

#[pymodule]
fn vectordb_balltree(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexBallTree>()?;
    Ok(())
}
