use pyo3::prelude::*;
use vectordb_core::distance;
use vectordb_core::VectorStorage;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

#[pyclass]
struct KDNode {
    index: usize,
    split_axis: usize,
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
}

impl KDNode {
    fn new(index: usize, split_axis: usize) -> Self {
        Self {
            index,
            split_axis,
            left: None,
            right: None,
        }
    }
}

#[pyclass]
struct IndexKDTree {
    storage: VectorStorage,
    leaf_size: usize,
    root: Option<Box<KDNode>>,
    built: bool,
}

#[pymethods]
impl IndexKDTree {
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

        let mut indices: Vec<usize> = (0..self.storage.size()).collect();
        self.root = self.build_tree(&mut indices, 0);
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

impl IndexKDTree {
    fn build_tree(&self, indices: &mut [usize], depth: usize) -> Option<Box<KDNode>> {
        if indices.is_empty() {
            return None;
        }

        let dim = self.storage.dimension();
        let axis = depth % dim;

        indices.sort_by(|&a, &b| {
            let va = self.storage.get_vector(a);
            let vb = self.storage.get_vector(b);
            va[axis].total_cmp(&vb[axis])
        });

        let median = indices.len() / 2;
        let mut node = Box::new(KDNode::new(indices[median], axis));

        let (left_part, right_part) = indices.split_at_mut(median);
        let right_part = &mut right_part[1..];

        node.left = self.build_tree(left_part, depth + 1);
        node.right = self.build_tree(right_part, depth + 1);

        Some(node)
    }

    fn search_k_nearest(&self, query: &[f32], node: &KDNode, k: usize, heap: &mut BinaryHeap<(OrderedFloat<f32>, usize)>) {
        let dim = self.storage.dimension();
        let vec = self.storage.get_vector(node.index);
        let dist = distance::compute_l2_distance(query, vec);

        if heap.len() < k {
            heap.push((OrderedFloat(dist), node.index));
        } else if OrderedFloat(dist) < heap.peek().unwrap().0 {
            heap.pop();
            heap.push((OrderedFloat(dist), node.index));
        }

        let diff = query[node.split_axis] - vec[node.split_axis];
        let near_child = if diff < 0.0 { &node.left } else { &node.right };
        let far_child = if diff < 0.0 { &node.right } else { &node.left };

        if let Some(child) = near_child {
            self.search_k_nearest(query, child, k, heap);
        }

        if heap.len() < k || OrderedFloat(diff * diff) < heap.peek().unwrap().0 {
            if let Some(child) = far_child {
                self.search_k_nearest(query, child, k, heap);
            }
        }
    }
}

#[pymodule]
fn vectordb_kdtree(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexKDTree>()?;
    Ok(())
}
