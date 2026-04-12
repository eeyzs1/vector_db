#!/usr/bin/env python3
"""
Comprehensive benchmark: C++ implementation vs FAISS
- Warmup + multi-run averaging for accurate timing
- Recall@K measurement against ground truth (FlatL2)
- Algorithms with FAISS equivalents: FlatL2, FlatIP, HNSW, IVF, PQ
- LSH: different algorithm approach, comparison for reference only
- KDTree, BallTree, Annoy: no FAISS equivalent, not compared
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import faiss
from vectordb.cpp import _flat, _flat_ip, _hnsw, _ivf, _pq, _lsh

N_WARMUP = 3
N_RUNS = 5


def format_time(t):
    if t < 1e-3:
        return f"{t*1e6:.1f}us"
    elif t < 1.0:
        return f"{t*1e3:.2f}ms"
    else:
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


def bench_flat_l2():
    print('\n' + '=' * 90)
    print('  Algorithm 1: FlatL2 - C++ vs FAISS')
    print('=' * 90)
    print(f"  {'Dim':>6} {'N':>8} {'K':>4} | {'Add C++':>10} {'Add FAISS':>10} {'Ratio':>7} | "
          f"{'Search C++':>10} {'Search FAISS':>12} {'Ratio':>7} | {'Match':>6}")
    print('-' * 90)

    configs = [
        (64, 10000, 10),
        (128, 100000, 10),
        (256, 100000, 10),
    ]

    for dim, n, k in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(100, dim).astype(np.float32)

        idx_cpp = _flat.IndexFlatL2(dim)
        t0 = time.perf_counter(); idx_cpp.add(vectors); add_cpp = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_cpp.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_cpp = np.mean(times)

        idx_faiss = faiss.IndexFlatL2(dim)
        t0 = time.perf_counter(); idx_faiss.add(vectors); add_faiss = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_faiss.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_faiss = np.mean(times)

        match = np.all(np.array(l_cpp) == l_faiss)
        add_spd = faiss_add_ratio = add_cpp / add_faiss if add_faiss > 0 else float('inf')
        search_spd = search_cpp / search_faiss if search_faiss > 0 else float('inf')

        print(f"  {dim:>6} {n:>8} {k:>4} | {format_time(add_cpp):>10} {format_time(add_faiss):>10} {add_spd:>6.2f}x | "
              f"{format_time(search_cpp):>10} {format_time(search_faiss):>12} {search_spd:>6.2f}x | {'Y' if match else 'N':>6}")


def bench_flat_ip():
    print('\n' + '=' * 90)
    print('  Algorithm 2: FlatIP - C++ vs FAISS')
    print('=' * 90)
    print(f"  {'Dim':>6} {'N':>8} {'K':>4} | {'Add C++':>10} {'Add FAISS':>10} {'Ratio':>7} | "
          f"{'Search C++':>10} {'Search FAISS':>12} {'Ratio':>7} | {'Match':>6}")
    print('-' * 90)

    configs = [
        (64, 10000, 10),
        (128, 100000, 10),
        (256, 100000, 10),
    ]

    for dim, n, k in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(100, dim).astype(np.float32)

        idx_cpp = _flat_ip.IndexFlatIP(dim)
        t0 = time.perf_counter(); idx_cpp.add(vectors); add_cpp = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_cpp.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_cpp = np.mean(times)

        idx_faiss = faiss.IndexFlatIP(dim)
        t0 = time.perf_counter(); idx_faiss.add(vectors); add_faiss = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_faiss.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_faiss = np.mean(times)

        match = np.all(np.array(l_cpp) == l_faiss)
        add_spd = add_cpp / add_faiss if add_faiss > 0 else float('inf')
        search_spd = search_cpp / search_faiss if search_faiss > 0 else float('inf')

        print(f"  {dim:>6} {n:>8} {k:>4} | {format_time(add_cpp):>10} {format_time(add_faiss):>10} {add_spd:>6.2f}x | "
              f"{format_time(search_cpp):>10} {format_time(search_faiss):>12} {search_spd:>6.2f}x | {'Y' if match else 'N':>6}")


def bench_hnsw():
    print('\n' + '=' * 90)
    print('  Algorithm 3: HNSW - C++ vs FAISS')
    print('=' * 90)
    print(f"  {'Dim':>6} {'N':>8} {'M':>4} {'efC':>5} {'efS':>5} | {'Add C++':>10} {'Add FAISS':>10} {'Ratio':>7} | "
          f"{'Search C++':>10} {'Search FAISS':>12} {'Ratio':>7} | {'R@C++':>6} {'R@FAISS':>7}")
    print('-' * 90)

    configs = [
        (64, 10000, 16, 200, 128),
        (128, 50000, 16, 200, 128),
    ]

    for dim, n, M, ef_c, ef_s in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(100, dim).astype(np.float32)
        k = 10

        idx_cpp = _hnsw.IndexHNSW(dim, M, ef_c)
        t0 = time.perf_counter(); idx_cpp.add(vectors); add_cpp = time.perf_counter() - t0
        idx_cpp.set_ef_search(ef_s)
        for _ in range(N_WARMUP): idx_cpp.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_cpp = np.mean(times)

        idx_faiss = faiss.IndexHNSWFlat(dim, M)
        idx_faiss.hnsw.efConstruction = ef_c
        idx_faiss.hnsw.efSearch = ef_s
        t0 = time.perf_counter(); idx_faiss.add(vectors); add_faiss = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_faiss.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_faiss = np.mean(times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)
        recall_cpp, recall_faiss = compute_recall(l_cpp_arr, l_faiss_arr, l_flat_arr)

        add_spd = add_cpp / add_faiss if add_faiss > 0 else float('inf')
        search_spd = search_cpp / search_faiss if search_faiss > 0 else float('inf')

        print(f"  {dim:>6} {n:>8} {M:>4} {ef_c:>5} {ef_s:>5} | {format_time(add_cpp):>10} {format_time(add_faiss):>10} {add_spd:>6.2f}x | "
              f"{format_time(search_cpp):>10} {format_time(search_faiss):>12} {search_spd:>6.2f}x | {recall_cpp:>5.1%} {recall_faiss:>6.1%}")


def bench_ivf():
    print('\n' + '=' * 90)
    print('  Algorithm 4: IVF - C++ vs FAISS')
    print('=' * 90)
    print(f"  {'Dim':>6} {'N':>8} {'nlist':>6} {'nprobe':>6} | {'Train C++':>10} {'Train FAISS':>11} {'Ratio':>7} | "
          f"{'Search C++':>10} {'Search FAISS':>12} {'Ratio':>7} | {'R@C++':>6} {'R@FAISS':>7}")
    print('-' * 90)

    configs = [
        (64, 10000, 100, 10),
        (128, 50000, 100, 10),
    ]

    for dim, n, nlist, nprobe in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(100, dim).astype(np.float32)
        k = 10

        idx_cpp = _ivf.IndexIVF(dim, nlist)
        t0 = time.perf_counter(); idx_cpp.train(vectors); train_cpp = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_cpp.add(vectors); add_cpp = time.perf_counter() - t0
        idx_cpp.set_nprobe(nprobe)
        for _ in range(N_WARMUP): idx_cpp.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_cpp = np.mean(times)

        quantizer = faiss.IndexFlatL2(dim)
        idx_faiss = faiss.IndexIVFFlat(quantizer, dim, nlist)
        t0 = time.perf_counter(); idx_faiss.train(vectors); train_faiss = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_faiss.add(vectors); add_faiss = time.perf_counter() - t0
        idx_faiss.nprobe = nprobe
        for _ in range(N_WARMUP): idx_faiss.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_faiss = np.mean(times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)
        recall_cpp, recall_faiss = compute_recall(l_cpp_arr, l_faiss_arr, l_flat_arr)

        train_spd = train_cpp / train_faiss if train_faiss > 0 else float('inf')
        search_spd = search_cpp / search_faiss if search_faiss > 0 else float('inf')

        print(f"  {dim:>6} {n:>8} {nlist:>6} {nprobe:>6} | {format_time(train_cpp):>10} {format_time(train_faiss):>11} {train_spd:>6.2f}x | "
              f"{format_time(search_cpp):>10} {format_time(search_faiss):>12} {search_spd:>6.2f}x | {recall_cpp:>5.1%} {recall_faiss:>6.1%}")


def bench_pq():
    print('\n' + '=' * 90)
    print('  Algorithm 5: PQ - C++ vs FAISS')
    print('=' * 90)
    print(f"  {'Dim':>6} {'N':>8} {'M':>4} {'nbits':>5} | {'Train C++':>10} {'Train FAISS':>11} {'Ratio':>7} | "
          f"{'Add C++':>10} {'Add FAISS':>10} {'Ratio':>7} | {'Search C++':>10} {'Search FAISS':>12} {'Ratio':>7} | {'R@C++':>6} {'R@FAISS':>7}")
    print('-' * 90)

    configs = [
        (64, 10000, 8, 8),
        (128, 50000, 8, 8),
    ]

    for dim, n, M, nbits in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(100, dim).astype(np.float32)
        k = 10

        idx_cpp = _pq.IndexPQ(dim, M, nbits)
        t0 = time.perf_counter(); idx_cpp.train(vectors); train_cpp = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_cpp.add(vectors); add_cpp = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_cpp.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_cpp = np.mean(times)

        idx_faiss = faiss.IndexPQ(dim, M, nbits)
        t0 = time.perf_counter(); idx_faiss.train(vectors); train_faiss = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_faiss.add(vectors); add_faiss = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_faiss.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_faiss = np.mean(times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)
        recall_cpp, recall_faiss = compute_recall(l_cpp_arr, l_faiss_arr, l_flat_arr)

        train_spd = train_cpp / train_faiss if train_faiss > 0 else float('inf')
        add_spd = add_cpp / add_faiss if add_faiss > 0 else float('inf')
        search_spd = search_cpp / search_faiss if search_faiss > 0 else float('inf')

        print(f"  {dim:>6} {n:>8} {M:>4} {nbits:>5} | {format_time(train_cpp):>10} {format_time(train_faiss):>11} {train_spd:>6.2f}x | "
              f"{format_time(add_cpp):>10} {format_time(add_faiss):>10} {add_spd:>6.2f}x | "
              f"{format_time(search_cpp):>10} {format_time(search_faiss):>12} {search_spd:>6.2f}x | {recall_cpp:>5.1%} {recall_faiss:>6.1%}")


def bench_lsh():
    print('\n' + '=' * 90)
    print('  Algorithm 6: LSH - C++ vs FAISS (NOTE: different algorithm approaches)')
    print('  C++ uses multi-table multi-probe LSH; FAISS uses single-table binary LSH')
    print('=' * 90)
    print(f"  {'Dim':>6} {'N':>8} {'Tables':>6} {'Funcs':>6} | {'Add C++':>10} {'Add FAISS':>10} {'Ratio':>7} | "
          f"{'Search C++':>10} {'Search FAISS':>12} {'Ratio':>7} | {'R@C++':>6} {'R@FAISS':>7}")
    print('-' * 90)

    configs = [
        (64, 10000, 8, 4),
        (128, 50000, 8, 4),
    ]

    for dim, n, num_tables, num_funcs in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(100, dim).astype(np.float32)
        k = 10

        idx_cpp = _lsh.IndexLSH(dim, num_tables, num_funcs)
        t0 = time.perf_counter(); idx_cpp.add(vectors); add_cpp = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_cpp.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_cpp = np.mean(times)

        nbits = num_tables * num_funcs
        idx_faiss = faiss.IndexLSH(dim, nbits)
        t0 = time.perf_counter(); idx_faiss.add(vectors); add_faiss = time.perf_counter() - t0
        for _ in range(N_WARMUP): idx_faiss.search(queries, k)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k)
            times.append(time.perf_counter() - t0)
        search_faiss = np.mean(times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)
        recall_cpp, recall_faiss = compute_recall(l_cpp_arr, l_faiss_arr, l_flat_arr)

        add_spd = add_cpp / add_faiss if add_faiss > 0 else float('inf')
        search_spd = search_cpp / search_faiss if search_faiss > 0 else float('inf')

        print(f"  {dim:>6} {n:>8} {num_tables:>6} {num_funcs:>6} | {format_time(add_cpp):>10} {format_time(add_faiss):>10} {add_spd:>6.2f}x | "
              f"{format_time(search_cpp):>10} {format_time(search_faiss):>12} {search_spd:>6.2f}x | {recall_cpp:>5.1%} {recall_faiss:>6.1%}")


def main():
    print("=" * 90)
    print("  COMPREHENSIVE BENCHMARK: C++ vs FAISS")
    print(f"  FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    print(f"  Warmup: {N_WARMUP} runs, Average over: {N_RUNS} runs")
    print("=" * 90)

    bench_flat_l2()
    bench_flat_ip()
    bench_hnsw()
    bench_ivf()
    bench_pq()
    bench_lsh()

    print('\n' + "=" * 90)
    print("  BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
