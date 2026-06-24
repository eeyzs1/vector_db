#!/usr/bin/env python3
"""Test LSH with different OMP thread counts."""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import faiss
from vectordb.cpp import _lsh

N_WARMUP = 3
N_RUNS = 5


def format_time(t):
    if t < 1e-3:
        return f"{t*1e6:.1f}us"
    elif t < 1.0:
        return f"{t*1e3:.2f}ms"
    return f"{t:.3f}s"


def compute_recall(our_labels, faiss_labels, flat_labels):
    nq = flat_labels.shape[0]
    recall_ours = 0.0
    recall_faiss = 0.0
    for i in range(nq):
        gt = set(flat_labels[i].tolist())
        if len(gt) > 0:
            recall_ours += len(set(our_labels[i].tolist()) & gt) / len(gt)
            recall_faiss += len(set(faiss_labels[i].tolist()) & gt) / len(gt)
    return recall_ours / nq, recall_faiss / nq


def test_lsh(dim, n, num_tables, num_funcs, threads):
    os.environ['OMP_NUM_THREADS'] = str(threads)
    # Need to reload to change thread count
    import importlib
    # Can't easily reload faiss, so just test our code
    
    k = 10
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(100, dim).astype(np.float32)

    idx_cpp = _lsh.IndexLSH(dim, num_tables, num_funcs)
    t0 = time.perf_counter(); idx_cpp.add(vectors); add_cpp = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx_cpp.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_cpp = np.mean(times)

    # FAISS
    nbits = num_tables * num_funcs
    idx_faiss = faiss.IndexLSH(dim, nbits)
    t0 = time.perf_counter(); idx_faiss.add(vectors); add_faiss = time.perf_counter() - t0
    for _ in range(N_WARMUP): idx_faiss.search(queries, k)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k)
        times.append(time.perf_counter() - t0)
    search_faiss = np.mean(times)

    # Ground truth
    idx_flat = faiss.IndexFlatL2(dim)
    idx_flat.add(vectors)
    _, l_flat = idx_flat.search(queries, k)

    l_cpp_arr = np.array(l_cpp, dtype=np.int64)
    l_faiss_arr = np.array(l_faiss, dtype=np.int64)
    l_flat_arr = np.array(l_flat, dtype=np.int64)
    recall_cpp, recall_faiss = compute_recall(l_cpp_arr, l_faiss_arr, l_flat_arr)

    add_spd = add_cpp / add_faiss if add_faiss > 0 else float('inf')
    search_spd = search_cpp / search_faiss if search_faiss > 0 else float('inf')

    print(f"  {dim:>4}d {n:>6} threads={threads:>2} | "
          f"Add: {format_time(add_cpp):>8} / {format_time(add_faiss):>8} ({add_spd:.2f}x) | "
          f"Search: {format_time(search_cpp):>8} / {format_time(search_faiss):>8} ({search_spd:.2f}x) | "
          f"Recall: {recall_cpp:.1%} / {recall_faiss:.1%}")


if __name__ == '__main__':
    print("Testing with different thread counts:")
    for threads in [1, 4, 8, 16]:
        test_lsh(64, 10000, 8, 4, threads)
    print()
    for threads in [1, 4, 8, 16]:
        test_lsh(128, 50000, 8, 4, threads)
