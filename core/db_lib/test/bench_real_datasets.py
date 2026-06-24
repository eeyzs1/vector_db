#!/usr/bin/env python3
"""
Benchmark C++ vs FAISS on standard ANN benchmark datasets.
Datasets: SIFT1M (128d), Fashion-MNIST (784d), MNIST (784d)
Source: ann-benchmarks.com (HDF5 format with ground truth)

Usage:
    python test/bench_real_datasets.py [--isolated] [--dataset sift|fashion-mnist|mnist|all]
"""

import time
import numpy as np
import sys
import os
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import faiss
from vectordb.cpp import _flat, _flat_ip, _hnsw, _ivf, _pq, _lsh

N_WARMUP = 3
N_RUNS = 5
N_QUERIES = 1000  # Use 1000 queries for statistical significance
K = 10

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

DATASETS = {
    'sift': {
        'file': 'sift-128-euclidean.hdf5',
        'dim': 128,
        'description': 'SIFT1M: 1M SIFT descriptors, 128-dim (gold standard)',
    },
    'fashion-mnist': {
        'file': 'fashion-mnist-784-euclidean.hdf5',
        'dim': 784,
        'description': 'Fashion-MNIST: 60K images, 784-dim',
    },
    'mnist': {
        'file': 'mnist-784-euclidean.hdf5',
        'dim': 784,
        'description': 'MNIST: 60K handwritten digits, 784-dim',
    },
}


def format_time(t):
    if t < 1e-3:
        return f"{t*1e6:.1f}us"
    elif t < 1.0:
        return f"{t*1e3:.2f}ms"
    else:
        return f"{t:.3f}s"


def load_dataset(name):
    info = DATASETS[name]
    path = os.path.join(DATASETS_DIR, info['file'])
    if not os.path.exists(path):
        print(f"  ERROR: Dataset file not found: {path}")
        return None

    f = h5py.File(path, 'r')
    train = np.array(f['train'], dtype=np.float32)
    test = np.array(f['test'][:N_QUERIES], dtype=np.float32)
    neighbors = np.array(f['neighbors'][:N_QUERIES], dtype=np.int64)
    f.close()

    print(f"  Loaded {name}: train={train.shape}, queries={test.shape}, dim={train.shape[1]}")
    return train, test, neighbors


def compute_recall_our(labels, ground_truth, k):
    """Recall@k: fraction of true top-k found."""
    nq = labels.shape[0]
    total = 0.0
    for i in range(nq):
        gt_set = set(ground_truth[i][:k].tolist())
        found = set(labels[i][:k].tolist())
        total += len(found & gt_set) / len(gt_set)
    return total / nq


def compute_recall_faiss(labels, ground_truth, k):
    return compute_recall_our(labels, ground_truth, k)


def bench_flat_l2(train, queries, gt, k):
    dim = train.shape[1]
    n = train.shape[0]

    # C++
    idx = _flat.IndexFlatL2(dim)
    t0 = time.perf_counter(); idx.add(train); add_cpp = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_cpp = np.mean(times)
    recall_cpp = compute_recall_our(np.array(l, dtype=np.int64), gt, k)

    # FAISS
    idx_f = faiss.IndexFlatL2(dim)
    t0 = time.perf_counter(); idx_f.add(train); add_f = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx_f.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx_f.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_f = np.mean(times)
    recall_f = compute_recall_faiss(np.array(l, dtype=np.int64), gt, k)

    return add_cpp, add_f, search_cpp, search_f, recall_cpp, recall_f


def bench_hnsw(train, queries, gt, k):
    dim = train.shape[1]
    M = 16
    ef_c = 200
    ef_s = 128

    # C++
    idx = _hnsw.IndexHNSW(dim, M, ef_c)
    t0 = time.perf_counter(); idx.add(train); add_cpp = time.perf_counter() - t0
    idx.set_ef_search(ef_s)
    for _ in range(N_WARMUP): idx.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_cpp = np.mean(times)
    recall_cpp = compute_recall_our(np.array(l, dtype=np.int64), gt, k)

    # FAISS
    idx_f = faiss.IndexHNSWFlat(dim, M)
    idx_f.hnsw.efConstruction = ef_c
    idx_f.hnsw.efSearch = ef_s
    t0 = time.perf_counter(); idx_f.add(train); add_f = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx_f.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx_f.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_f = np.mean(times)
    recall_f = compute_recall_faiss(np.array(l, dtype=np.int64), gt, k)

    return add_cpp, add_f, search_cpp, search_f, recall_cpp, recall_f


def bench_ivf(train, queries, gt, k):
    dim = train.shape[1]
    n = train.shape[0]
    nlist = min(100, int(np.sqrt(n)))
    nprobe = 10

    # C++
    idx = _ivf.IndexIVF(dim, nlist)
    t0 = time.perf_counter(); idx.train(train); train_cpp = time.perf_counter() - t0
    t0 = time.perf_counter(); idx.add(train); add_cpp = time.perf_counter() - t0
    idx.set_nprobe(nprobe)
    for _ in range(N_WARMUP): idx.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_cpp = np.mean(times)
    recall_cpp = compute_recall_our(np.array(l, dtype=np.int64), gt, k)

    # FAISS
    quantizer = faiss.IndexFlatL2(dim)
    idx_f = faiss.IndexIVFFlat(quantizer, dim, nlist)
    t0 = time.perf_counter(); idx_f.train(train); train_f = time.perf_counter() - t0
    t0 = time.perf_counter(); idx_f.add(train); add_f = time.perf_counter() - t0
    idx_f.nprobe = nprobe
    for _ in range(N_WARMUP): idx_f.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx_f.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_f = np.mean(times)
    recall_f = compute_recall_faiss(np.array(l, dtype=np.int64), gt, k)

    return train_cpp, train_f, search_cpp, search_f, recall_cpp, recall_f


def bench_pq(train, queries, gt, k):
    dim = train.shape[1]
    # M must divide dim. For 128: M=8. For 784: M=16 (784/16=49).
    if dim == 128:
        M = 8
    elif dim == 784:
        M = 16  # 784 / 16 = 49
    else:
        M = 8
        if dim % M != 0:
            M = 4
    nbits = 8

    # C++
    idx = _pq.IndexPQ(dim, M, nbits)
    t0 = time.perf_counter(); idx.train(train); train_cpp = time.perf_counter() - t0
    t0 = time.perf_counter(); idx.add(train); add_cpp = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_cpp = np.mean(times)
    recall_cpp = compute_recall_our(np.array(l, dtype=np.int64), gt, k)

    # FAISS
    idx_f = faiss.IndexPQ(dim, M, nbits)
    t0 = time.perf_counter(); idx_f.train(train); train_f = time.perf_counter() - t0
    t0 = time.perf_counter(); idx_f.add(train); add_f = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx_f.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx_f.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_f = np.mean(times)
    recall_f = compute_recall_faiss(np.array(l, dtype=np.int64), gt, k)

    return train_cpp, train_f, search_cpp, search_f, recall_cpp, recall_f


def bench_lsh(train, queries, gt, k):
    dim = train.shape[1]
    num_tables = 8
    num_funcs = 4

    # C++
    idx = _lsh.IndexLSH(dim, num_tables, num_funcs)
    t0 = time.perf_counter(); idx.add(train); add_cpp = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_cpp = np.mean(times)
    recall_cpp = compute_recall_our(np.array(l, dtype=np.int64), gt, k)

    # FAISS
    nbits = num_tables * num_funcs
    idx_f = faiss.IndexLSH(dim, nbits)
    t0 = time.perf_counter(); idx_f.add(train); add_f = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx_f.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d, l = idx_f.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_f = np.mean(times)
    recall_f = compute_recall_faiss(np.array(l, dtype=np.int64), gt, k)

    return add_cpp, add_f, search_cpp, search_f, recall_cpp, recall_f


def run_dataset_benchmark(dataset_name):
    print('\n' + '=' * 100)
    print(f'  Dataset: {dataset_name} — {DATASETS[dataset_name]["description"]}')
    print(f'  Queries: {N_QUERIES}, K: {K}, Warmup: {N_WARMUP}, Runs: {N_RUNS}')
    print('=' * 100)

    data = load_dataset(dataset_name)
    if data is None:
        return
    train, queries, gt = data

    # --- FlatL2 ---
    print(f"\n  [FlatL2] Ground truth computation + speed test")
    add_c, add_f, s_c, s_f, r_c, r_f = bench_flat_l2(train, queries, gt, K)
    print(f"  {'Add C++':>12} {'Add FAISS':>12} {'Ratio':>8} | {'Search C++':>12} {'Search FAISS':>14} {'Ratio':>8} | {'R@C++':>7} {'R@FAISS':>7}")
    print(f"  {format_time(add_c):>12} {format_time(add_f):>12} {add_c/add_f:>7.2f}x | "
          f"{format_time(s_c):>12} {format_time(s_f):>14} {s_c/s_f:>7.2f}x | {r_c:>6.1%} {r_f:>6.1%}")

    # --- HNSW ---
    print(f"\n  [HNSW] M=16, efC=200, efS=128")
    add_c, add_f, s_c, s_f, r_c, r_f = bench_hnsw(train, queries, gt, K)
    print(f"  {'Add C++':>12} {'Add FAISS':>12} {'Ratio':>8} | {'Search C++':>12} {'Search FAISS':>14} {'Ratio':>8} | {'R@C++':>7} {'R@FAISS':>7}")
    print(f"  {format_time(add_c):>12} {format_time(add_f):>12} {add_c/add_f:>7.2f}x | "
          f"{format_time(s_c):>12} {format_time(s_f):>14} {s_c/s_f:>7.2f}x | {r_c:>6.1%} {r_f:>6.1%}")

    # --- IVF ---
    nlist = min(100, int(np.sqrt(train.shape[0])))
    print(f"\n  [IVF] nlist={nlist}, nprobe=10")
    tr_c, tr_f, s_c, s_f, r_c, r_f = bench_ivf(train, queries, gt, K)
    print(f"  {'Train C++':>12} {'Train FAISS':>12} {'Ratio':>8} | {'Search C++':>12} {'Search FAISS':>14} {'Ratio':>8} | {'R@C++':>7} {'R@FAISS':>7}")
    print(f"  {format_time(tr_c):>12} {format_time(tr_f):>12} {tr_c/tr_f:>7.2f}x | "
          f"{format_time(s_c):>12} {format_time(s_f):>14} {s_c/s_f:>7.2f}x | {r_c:>6.1%} {r_f:>6.1%}")

    # --- PQ ---
    dim = train.shape[1]
    M = 8 if dim == 128 else 16
    print(f"\n  [PQ] M={M}, nbits=8 (compression {dim//M}:1)")
    tr_c, tr_f, s_c, s_f, r_c, r_f = bench_pq(train, queries, gt, K)
    print(f"  {'Train C++':>12} {'Train FAISS':>12} {'Ratio':>8} | {'Search C++':>12} {'Search FAISS':>14} {'Ratio':>8} | {'R@C++':>7} {'R@FAISS':>7}")
    print(f"  {format_time(tr_c):>12} {format_time(tr_f):>12} {tr_c/tr_f:>7.2f}x | "
          f"{format_time(s_c):>12} {format_time(s_f):>14} {s_c/s_f:>7.2f}x | {r_c:>6.1%} {r_f:>6.1%}")

    # --- LSH ---
    print(f"\n  [LSH] 8 tables x 4 funcs = 32 bits")
    add_c, add_f, s_c, s_f, r_c, r_f = bench_lsh(train, queries, gt, K)
    print(f"  {'Add C++':>12} {'Add FAISS':>12} {'Ratio':>8} | {'Search C++':>12} {'Search FAISS':>14} {'Ratio':>8} | {'R@C++':>7} {'R@FAISS':>7}")
    print(f"  {format_time(add_c):>12} {format_time(add_f):>12} {add_c/add_f:>7.2f}x | "
          f"{format_time(s_c):>12} {format_time(s_f):>14} {s_c/s_f:>7.2f}x | {r_c:>6.1%} {r_f:>6.1%}")


def run_isolated(datasets_to_run):
    print("=" * 100)
    print("  REAL DATASET BENCHMARK: C++ vs FAISS (ISOLATED MODE)")
    print("  Each algorithm runs in a separate process with cooldown")
    print(f"  FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    print(f"  Queries: {N_QUERIES}, K: {K}, Warmup: {N_WARMUP}, Runs: {N_RUNS}")
    print("=" * 100)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for ds in datasets_to_run:
        print(f"\n--- Running {ds} benchmark in isolated process ---")
        cmd = [
            sys.executable, '-c',
            f"import sys,os; sys.path.insert(0, os.path.abspath({repr(os.path.join(script_dir, '..', 'python'))})); "
            f"sys.path.insert(0, {repr(script_dir)}); "
            f"import bench_real_datasets as b; b.run_dataset_benchmark({repr(ds)})"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        print(result.stdout)
        if result.stderr:
            # Filter out common non-error messages
            for line in result.stderr.split('\n'):
                if line and 'warning' not in line.lower() and 'UserWarning' not in line:
                    print(line, file=sys.stderr)
        time.sleep(3)

    print('\n' + "=" * 100)
    print("  BENCHMARK COMPLETE (ISOLATED MODE)")
    print("=" * 100)


import subprocess

def main():
    args = sys.argv[1:]
    isolated = '--isolated' in args

    # Determine which datasets to run
    datasets_to_run = list(DATASETS.keys())
    for i, arg in enumerate(args):
        if arg == '--dataset' and i + 1 < len(args):
            ds = args[i + 1]
            if ds in DATASETS:
                datasets_to_run = [ds]
            elif ds == 'all':
                datasets_to_run = list(DATASETS.keys())
            else:
                print(f"Unknown dataset: {ds}. Available: {list(DATASETS.keys())}")
                return

    # Check which datasets exist
    available = []
    for ds in datasets_to_run:
        path = os.path.join(DATASETS_DIR, DATASETS[ds]['file'])
        if os.path.exists(path):
            available.append(ds)
        else:
            print(f"  Skipping {ds}: file not found ({DATASETS[ds]['file']})")

    if not available:
        print("No datasets available. Download from https://ann-benchmarks.com/")
        return

    if isolated:
        run_isolated(available)
    else:
        print("=" * 100)
        print("  REAL DATASET BENCHMARK: C++ vs FAISS")
        print(f"  FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
        print(f"  Queries: {N_QUERIES}, K: {K}, Warmup: {N_WARMUP}, Runs: {N_RUNS}")
        print("=" * 100)
        for ds in available:
            run_dataset_benchmark(ds)
        print('\n' + "=" * 100)
        print("  BENCHMARK COMPLETE")
        print("=" * 100)


if __name__ == '__main__':
    main()
