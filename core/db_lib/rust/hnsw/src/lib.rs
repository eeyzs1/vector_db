use pyo3::prelude::*;
use vectordb_core::distance;
use vectordb_core::VectorStorage;
use rand::Rng;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use ordered_float::OrderedFloat;

#[pyclass]
struct HNSWNode {
    neighbors: Vec<Vec<usize>>,
}

impl HNSWNode {
    fn new(max_level: usize) -> Self {
        Self {
            neighbors: vec![Vec::new(); max_level + 1],
        }
    }
}

#[pyclass]
struct IndexHNSW {
    storage: VectorStorage,
    M: usize,
    ef_construction: usize,
    ef_search: usize,
    max_level: usize,
    enter_point: usize,
    nodes: Vec<HNSWNode>,
}

#[pymethods]
impl IndexHNSW {
    #[new]
    fn new(dimension: usize, M: Option<usize>, ef_construction: Option<usize>) -> Self {
        let M = M.unwrap_or(16);
        let ef_construction = ef_construction.unwrap_or(200);
        Self {
            storage: VectorStorage::new(dimension),
            M,
            ef_construction,
            ef_search: ef_construction,
            max_level: 0,
            enter_point: 0,
            nodes: Vec::new(),
        }
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
            self.insert_node(idx, vec);
        }

        Ok(())
    }

    #[setter]
    fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    fn search(&self, x: Vec<Vec<f32>>, k: usize) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<i64>>)> {
        let dim = self.storage.dimension();
        let mut all_distances = Vec::with_capacity(x.len());
        let mut all_labels = Vec::with_capacity(x.len());

        for query in x {
            let mut candidates = Vec::new();
            if self.storage.size() > 0 {
                candidates.push(self.enter_point);

                for l in (1..=self.max_level).rev() {
                    self.search_layer(&query, &mut candidates, l, 1);
                }
                self.search_layer(&query, &mut candidates, 0, std::cmp::max(self.ef_search, k));
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
    fn M(&self) -> usize {
        self.M
    }

    #[getter]
    fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    #[getter]
    fn ef_search(&self) -> usize {
        self.ef_search
    }
}

impl IndexHNSW {
    fn random_level() -> usize {
        let mut rng = rand::thread_rng();
        let level = (-rng.gen::<f32>().ln() * 16.0f32) as usize;
        level
    }

    fn search_layer(&self, query: &[f32], candidates: &mut Vec<usize>, level: usize, ef: usize) {
        let dim = self.storage.dimension();
        let mut candidates_set = BinaryHeap::new();
        let mut visited = HashSet::new();

        for &idx in candidates.iter() {
            let vec = self.storage.get_vector(idx);
            let dist = distance::compute_l2_distance(query, vec);
            candidates_set.push(Reverse((OrderedFloat(dist), idx)));
            visited.insert(idx);
        }

        candidates.clear();
        while let Some(Reverse((dist_c, c))) = candidates_set.pop() {
            let (dist_f, _f) = if candidates_set.is_empty() {
                (OrderedFloat(f32::MAX), 0)
            } else {
                let &Reverse((d, f_idx)) = candidates_set.peek().unwrap();
                (d, f_idx)
            };

            if dist_c > dist_f && candidates.len() >= ef {
                break;
            }

            for &neighbor in self.nodes[c].neighbors[level].iter() {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let vec = self.storage.get_vector(neighbor);
                    let dist = distance::compute_l2_distance(query, vec);

                    if candidates_set.len() < ef || dist < *dist_f {
                        candidates_set.push(Reverse((OrderedFloat(dist), neighbor)));
                    }
                }
            }
        }

        while let Some(Reverse((_, idx))) = candidates_set.pop() {
            candidates.push(idx);
        }
    }

    fn connect(&mut self, a: usize, b: usize, level: usize) {
        self.nodes[a].neighbors[level].push(b);
        self.nodes[b].neighbors[level].push(a);
    }

    fn select_neighbors(&self, query: &[f32], candidates: &[usize], level: usize) -> Vec<usize> {
        let dim = self.storage.dimension();
        let mut dists = Vec::with_capacity(candidates.len());
        for &idx in candidates {
            let vec = self.storage.get_vector(idx);
            let dist = distance::compute_l2_distance(query, vec);
            dists.push((dist, idx));
        }
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));

        let M_max = if level == 0 { 2 * self.M } else { self.M };
        let take = std::cmp::min(M_max, dists.len());
        dists.iter().take(take).map(|&(_, idx)| idx).collect()
    }

    fn insert_node(&mut self, idx: usize, vec: &[f32]) {
        let new_level = Self::random_level();

        let max_level = std::cmp::max(new_level, self.max_level);
        self.nodes.push(HNSWNode::new(max_level));

        let mut entry_points = Vec::new();
        if self.storage.size() > 1 {
            entry_points.push(self.enter_point);
            for l in (new_level + 1..=self.max_level).rev() {
                self.search_layer(vec, &mut entry_points, l, 1);
            }
        }

        for l in (0..=std::cmp::min(new_level, self.max_level)).rev() {
            self.search_layer(vec, &mut entry_points, l, self.ef_construction);
            let neighbors = self.select_neighbors(vec, &entry_points, l);
            for neighbor in neighbors {
                self.connect(idx, neighbor, l);
            }
        }

        if new_level > self.max_level || self.storage.size() == 1 {
            self.enter_point = idx;
            self.max_level = std::cmp::max(self.max_level, new_level);
        }
    }
}

#[pymodule]
fn vectordb_hnsw(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexHNSW>()?;
    Ok(())
}
