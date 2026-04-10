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

_lsh = load_cpp_module("_lsh", "_lsh")

import faiss


def benchmark_lsh(dim, n_vectors, n_queries, k, num_hash_tables=8, num_hash_functions=4, num_probes=1):
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    idx_ours = _lsh.IndexLSH(dim, num_hash_tables, num_hash_functions)
    idx_ours.set_num_probes(num_probes)
    t0 = time.perf_counter()
    idx_ours.add(vectors)
    add_time_ours = time.perf_counter() - t0

    n_bits = num_hash_tables * num_hash_functions
    idx_faiss_lsh = faiss.IndexLSH(dim, n_bits)
    t0 = time.perf_counter()
    idx_faiss_lsh.add(vectors)
    add_time_faiss = time.perf_counter() - t0

    idx_flat = faiss.IndexFlatL2(dim)
    idx_flat.add(vectors)

    n_warmup = 3
    n_runs = 5

    for _ in range(n_warmup):
        idx_ours.search(queries, k)
        idx_faiss_lsh.search(queries, k)

    ours_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        d_ours, l_ours = idx_ours.search(queries, k)
        ours_times.append(time.perf_counter() - t0)

    faiss_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        d_faiss, l_faiss = idx_faiss_lsh.search(queries, k)
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

    return ours_avg, faiss_avg, add_time_ours, add_time_faiss, recall_ours, recall_faiss


def format_time(t):
    if t < 1e-3:
        return f"{t*1e6:.1f} us"
    elif t < 1.0:
        return f"{t*1e3:.2f} ms"
    else:
        return f"{t:.3f} s"


def main():
    print("=" * 120)
    print("  LSH Performance Benchmark: Our C++ LSH (Optimized) vs FAISS IndexLSH")
    print("  Ground Truth: FAISS IndexFlatL2 (exact search)")
    print("=" * 120)
    print()

    print("-" * 120)
    print("  Search Performance (single-probe, with Recall@K vs Ground Truth)")
    print("-" * 120)
    header = f"  {'Dim':>6} {'N':>8} {'Q':>6} {'K':>4} {'Tables':>6} {'Funcs':>6} | {'Ours':>12} {'FAISS':>12} {'Ratio':>8} | {'R@Ours':>7} {'R@FAISS':>8}"
    print(header)
    print("-" * 120)

    configs = [
        (64,  10000, 100, 10, 8, 4),
        (128, 10000, 100, 10, 8, 4),
        (128, 50000, 100, 10, 8, 4),
        (128, 100000, 100, 10, 8, 4),
        (256, 50000, 100, 10, 8, 4),
        (256, 100000, 100, 10, 8, 4),
    ]

    for dim, n, nq, k, tables, funcs in configs:
        ours_t, faiss_t, add_ours, add_faiss, recall_ours, recall_faiss = benchmark_lsh(
            dim, n, nq, k, tables, funcs
        )
        ratio = faiss_t / ours_t if ours_t > 0 else float('inf')
        print(f"  {dim:>6} {n:>8} {nq:>6} {k:>4} {tables:>6} {funcs:>6} | "
              f"{format_time(ours_t):>12} {format_time(faiss_t):>12} {ratio:>7.2f}x | "
              f"{recall_ours:>6.1%} {recall_faiss:>7.1%}")

    print()
    print("-" * 120)
    print("  Multi-Probe LSH: Varying num_probes (dim=128, N=50000, Q=100, K=10, Tables=4, Funcs=4)")
    print("  Compare: 4-tables multi-probe vs 16-tables single-probe (same total probes)")
    print("-" * 120)
    header2 = f"  {'Tables':>6} {'Funcs':>6} {'Probes':>6} {'TotPrb':>6} | {'Ours':>12} {'FAISS':>12} {'Ratio':>8} | {'R@Ours':>7} {'R@FAISS':>8}"
    print(header2)
    print("-" * 120)

    probe_configs = [
        (4, 4, 1),
        (4, 4, 2),
        (4, 4, 4),
        (4, 4, 8),
        (8, 4, 1),
        (8, 4, 2),
        (8, 4, 4),
        (16, 4, 1),
    ]

    for tables, funcs, probes in probe_configs:
        ours_t, faiss_t, _, _, recall_ours, recall_faiss = benchmark_lsh(
            128, 50000, 100, 10, tables, funcs, probes
        )
        ratio = faiss_t / ours_t if ours_t > 0 else float('inf')
        total_probes = tables * probes
        print(f"  {tables:>6} {funcs:>6} {probes:>6} {total_probes:>6} | "
              f"{format_time(ours_t):>12} {format_time(faiss_t):>12} {ratio:>7.2f}x | "
              f"{recall_ours:>6.1%} {recall_faiss:>7.1%}")

    print()
    print("-" * 120)
    print("  Varying Hash Parameters (dim=128, N=50000, Q=100, K=10, single-probe)")
    print("-" * 120)
    header3 = f"  {'Tables':>6} {'Funcs':>6} {'Bits':>6} | {'Ours':>12} {'FAISS':>12} {'Ratio':>8} | {'R@Ours':>7} {'R@FAISS':>8}"
    print(header3)
    print("-" * 120)

    param_configs = [
        (4, 4),
        (8, 4),
        (16, 4),
        (8, 8),
        (16, 8),
        (32, 4),
    ]

    for tables, funcs in param_configs:
        ours_t, faiss_t, _, _, recall_ours, recall_faiss = benchmark_lsh(
            128, 50000, 100, 10, tables, funcs
        )
        ratio = faiss_t / ours_t if ours_t > 0 else float('inf')
        bits = tables * funcs
        print(f"  {tables:>6} {funcs:>6} {bits:>6} | "
              f"{format_time(ours_t):>12} {format_time(faiss_t):>12} {ratio:>7.2f}x | "
              f"{recall_ours:>6.1%} {recall_faiss:>7.1%}")

    print()
    print("=" * 120)
    print("  Benchmark Complete")
    print("=" * 120)


if __name__ == "__main__":
    main()
