use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flat_ip::FlatIPIndex;
use rand::Rng;

fn generate_random_vectors(num_vectors: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen()).collect())
        .collect()
}

fn bench_add(c: &mut Criterion) {
    let dim = 128;
    let mut index = FlatIPIndex::new(dim);
    let vectors = generate_random_vectors(1000, dim);

    c.bench_function("add_1000_vectors_128d", |b| {
        b.iter(|| {
            let mut idx = FlatIPIndex::new(dim);
            idx.add(black_box(vectors.clone())).unwrap();
        })
    });
}

fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let mut index = FlatIPIndex::new(dim);
    let vectors = generate_random_vectors(10000, dim);
    index.add(vectors.clone()).unwrap();
    
    let query = vectors[0].clone();
    
    c.bench_function("search_10k_vectors_128d_k10", |b| {
        b.iter(|| {
            black_box(index.search(black_box(query.clone()), black_box(10))).unwrap()
        })
    });
}

fn bench_distance_compute(c: &mut Criterion) {
    let dim = 128;
    let mut rng = rand::thread_rng();
    let vec1: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
    let vec2: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
    
    c.bench_function("compute_ip_distance_128d", |b| {
        b.iter(|| {
            black_box(vectordb_core::distance::compute_ip_distance(
                black_box(&vec1), 
                black_box(&vec2)
            ))
        })
    });
}

criterion_group!(benches, bench_add, bench_search, bench_distance_compute);
criterion_main!(benches);
