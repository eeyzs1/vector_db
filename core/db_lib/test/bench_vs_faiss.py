import sys
import time
import importlib.util
import numpy as np

BASE_DIR = "/home/eeyzs1/AI_Generated_Projects/7vec_db/vector_db/core/db_lib"

def load_cpp_module(file_name, module_name):
    so_path = f"{BASE_DIR}/python/vectordb/cpp/{file_name}.cpython-314-x86_64-linux-gnu.so"
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_flat_ip = load_cpp_module("_flat_ip", "_flat_ip")
vectordb_ivf = load_cpp_module("_ivf", "vectordb_ivf")

import faiss

def benchmark_flat_ip(dim, n_vectors, n_queries, k):
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / np.maximum(vectors_norm, 1e-10)
    queries_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    queries_norm = queries / np.maximum(queries_norm, 1e-10)

    idx_ours = _flat_ip.IndexFlatIP(dim)
    idx_ours.add(vectors_norm)

    idx_faiss = faiss.IndexFlatIP(dim)
    idx_faiss.add(vectors_norm)

    n_warmup = 3
    n_runs = 5

    for _ in range(n_warmup):
        idx_ours.search(queries_norm, k)
        idx_faiss.search(queries_norm, k)

    ours_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        d_ours, l_ours = idx_ours.search(queries_norm, k)
        ours_times.append(time.perf_counter() - t0)

    faiss_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        d_faiss, l_faiss = idx_faiss.search(queries_norm, k)
        faiss_times.append(time.perf_counter() - t0)

    ours_avg = np.mean(ours_times)
    faiss_avg = np.mean(faiss_times)

    l_ours_arr = np.array(l_ours, dtype=np.int64)
    l_faiss_arr = np.array(l_faiss, dtype=np.int64)
    match_rate = np.mean(l_ours_arr == l_faiss_arr)

    return ours_avg, faiss_avg, match_rate


def benchmark_ivf(dim, n_vectors, n_queries, k, nlist, nprobe):
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    idx_ours = vectordb_ivf.IndexIVF(dim, nlist)
    idx_ours.train(vectors)
    idx_ours.add(vectors)
    idx_ours.set_nprobe(nprobe)

    quantizer = faiss.IndexFlatL2(dim)
    idx_faiss = faiss.IndexIVFFlat(quantizer, dim, nlist)
    idx_faiss.train(vectors)
    idx_faiss.add(vectors)
    idx_faiss.nprobe = nprobe

    idx_flat = faiss.IndexFlatL2(dim)
    idx_flat.add(vectors)

    n_warmup = 3
    n_runs = 5

    for _ in range(n_warmup):
        idx_ours.search(queries, k)
        idx_faiss.search(queries, k)

    ours_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        d_ours, l_ours = idx_ours.search(queries, k)
        ours_times.append(time.perf_counter() - t0)

    faiss_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        d_faiss, l_faiss = idx_faiss.search(queries, k)
        faiss_times.append(time.perf_counter() - t0)

    ours_avg = np.mean(ours_times)
    faiss_avg = np.mean(faiss_times)

    _, l_flat = idx_flat.search(queries, k)
    l_ours_arr = np.array(l_ours, dtype=np.int64)
    l_faiss_arr = np.array(l_faiss, dtype=np.int64)
    l_flat_arr = np.array(l_flat, dtype=np.int64)

    recall_ours = 0.0
    recall_faiss = 0.0
    for i in range(n_queries):
        gt_set = set(l_flat_arr[i].tolist())
        ours_set = set(l_ours_arr[i].tolist())
        faiss_set = set(l_faiss_arr[i].tolist())
        if len(gt_set) > 0:
            recall_ours += len(ours_set & gt_set) / len(gt_set)
            recall_faiss += len(faiss_set & gt_set) / len(gt_set)
    recall_ours /= n_queries
    recall_faiss /= n_queries

    return ours_avg, faiss_avg, recall_ours, recall_faiss


def format_time(t):
    if t < 1e-3:
        return f"{t*1e6:.1f} us"
    elif t < 1.0:
        return f"{t*1e3:.2f} ms"
    else:
        return f"{t:.3f} s"


def main():
    print("=" * 90)
    print("  VectorDB vs FAISS Performance Benchmark")
    print("=" * 90)
    print()

    print("-" * 90)
    print("  FlatIP (Inner Product) Benchmark")
    print("-" * 90)
    print(f"  {'Dim':>6} {'N':>8} {'Queries':>8} {'K':>4} | {'Ours':>12} {'FAISS':>12} {'Speedup':>8} | {'Match':>6}")
    print("-" * 90)

    flat_configs = [
        (64, 10000, 100, 10),
        (128, 10000, 100, 10),
        (128, 100000, 100, 10),
        (256, 100000, 100, 10),
        (128, 1000000, 100, 10),
    ]

    for dim, n, nq, k in flat_configs:
        ours_t, faiss_t, match = benchmark_flat_ip(dim, n, nq, k)
        speedup = faiss_t / ours_t if ours_t > 0 else float('inf')
        print(f"  {dim:>6} {n:>8} {nq:>8} {k:>4} | {format_time(ours_t):>12} {format_time(faiss_t):>12} {speedup:>7.2f}x | {match:>5.1%}")

    print()
    print("-" * 90)
    print("  IVF (Inverted File) Benchmark — Recall vs Ground Truth (FlatL2)")
    print("-" * 90)
    print(f"  {'Dim':>6} {'N':>8} {'nlist':>6} {'nprobe':>6} | {'Ours':>12} {'FAISS':>12} {'Speedup':>8} | {'R@Ours':>7} {'R@FAISS':>8}")
    print("-" * 90)

    ivf_configs = [
        (64, 10000, 100, 10, 10),
        (64, 10000, 100, 50, 10),
        (128, 100000, 1000, 10, 10),
        (128, 100000, 1000, 100, 10),
        (128, 100000, 1000, 500, 10),
        (256, 100000, 1000, 100, 10),
        (256, 100000, 1000, 500, 10),
    ]

    for dim, n, nlist, nprobe, k in ivf_configs:
        nq = 100
        ours_t, faiss_t, recall_ours, recall_faiss = benchmark_ivf(dim, n, nq, k, nlist, nprobe)
        speedup = faiss_t / ours_t if ours_t > 0 else float('inf')
        print(f"  {dim:>6} {n:>8} {nlist:>6} {nprobe:>6} | {format_time(ours_t):>12} {format_time(faiss_t):>12} {speedup:>7.2f}x | {recall_ours:>6.1%} {recall_faiss:>7.1%}")

    print()
    print("=" * 90)
    print("  Benchmark Complete")
    print("=" * 90)


if __name__ == "__main__":
    main()
