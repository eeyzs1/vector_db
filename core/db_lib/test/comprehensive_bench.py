import sys
import time
sys.path.insert(0, 'python')

import numpy as np
import faiss
from vectordb.cpp import _flat, _flat_ip, _ivf, _hnsw, _pq, _lsh

def format_time(t):
    if t < 1e-3: return f'{t*1e6:.1f}us'
    elif t < 1.0: return f'{t*1e3:.2f}ms'
    else: return f'{t:.3f}s'

def bench_flat_l2():
    print('\n' + '='*80)
    print('  Algorithm 1: FlatL2 - C++ vs FAISS')
    print('='*80)
    results = []
    configs = [
        (64, 10000, 100, 10),
        (128, 100000, 100, 10),
        (256, 100000, 100, 10),
    ]
    for dim, n, nq, k in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(nq, dim).astype(np.float32)

        idx_cpp = _flat.IndexFlatL2(dim)
        t0 = time.perf_counter(); idx_cpp.add(vectors); cpp_add = time.perf_counter() - t0
        for _ in range(3): idx_cpp.search(queries, k)
        cpp_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k); cpp_times.append(time.perf_counter() - t0)
        cpp_search = np.mean(cpp_times)

        idx_faiss = faiss.IndexFlatL2(dim)
        t0 = time.perf_counter(); idx_faiss.add(vectors); faiss_add = time.perf_counter() - t0
        for _ in range(3): idx_faiss.search(queries, k)
        faiss_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k); faiss_times.append(time.perf_counter() - t0)
        faiss_search = np.mean(faiss_times)

        match = np.all(np.array(l_cpp) == l_faiss)
        add_spd = faiss_add / cpp_add
        search_spd = faiss_search / cpp_search
        results.append((dim, n, add_spd, search_spd, match))
        print(f'  Dim={dim}, N={n}: Add={add_spd:.2f}x, Search={search_spd:.2f}x, Match={match}')

    return results

def bench_flat_ip():
    print('\n' + '='*80)
    print('  Algorithm 2: FlatIP - C++ vs FAISS')
    print('='*80)
    results = []
    configs = [
        (64, 10000, 100, 10),
        (128, 100000, 100, 10),
        (256, 100000, 100, 10),
    ]
    for dim, n, nq, k in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(nq, dim).astype(np.float32)

        idx_cpp = _flat_ip.IndexFlatIP(dim)
        t0 = time.perf_counter(); idx_cpp.add(vectors); cpp_add = time.perf_counter() - t0
        for _ in range(3): idx_cpp.search(queries, k)
        cpp_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k); cpp_times.append(time.perf_counter() - t0)
        cpp_search = np.mean(cpp_times)

        idx_faiss = faiss.IndexFlatIP(dim)
        t0 = time.perf_counter(); idx_faiss.add(vectors); faiss_add = time.perf_counter() - t0
        for _ in range(3): idx_faiss.search(queries, k)
        faiss_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k); faiss_times.append(time.perf_counter() - t0)
        faiss_search = np.mean(faiss_times)

        match = np.all(np.array(l_cpp) == l_faiss)
        add_spd = faiss_add / cpp_add
        search_spd = faiss_search / cpp_search
        results.append((dim, n, add_spd, search_spd, match))
        print(f'  Dim={dim}, N={n}: Add={add_spd:.2f}x, Search={search_spd:.2f}x, Match={match}')

    return results

def bench_ivf():
    print('\n' + '='*80)
    print('  Algorithm 3: IVF - C++ vs FAISS')
    print('='*80)
    results = []
    configs = [
        (128, 100000, 100, 10, 100, 10),
        (128, 100000, 100, 10, 100, 50),
    ]
    for dim, n, nq, k, nlist, nprobe in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(nq, dim).astype(np.float32)

        idx_cpp = _ivf.IndexIVF(dim, nlist)
        t0 = time.perf_counter(); idx_cpp.train(vectors); cpp_train = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_cpp.add(vectors); cpp_add = time.perf_counter() - t0
        idx_cpp.set_nprobe(nprobe)
        for _ in range(3): idx_cpp.search(queries, k)
        cpp_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k); cpp_times.append(time.perf_counter() - t0)
        cpp_search = np.mean(cpp_times)

        quantizer = faiss.IndexFlatL2(dim)
        idx_faiss = faiss.IndexIVFFlat(quantizer, dim, nlist)
        t0 = time.perf_counter(); idx_faiss.train(vectors); faiss_train = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_faiss.add(vectors); faiss_add = time.perf_counter() - t0
        idx_faiss.nprobe = nprobe
        for _ in range(3): idx_faiss.search(queries, k)
        faiss_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k); faiss_times.append(time.perf_counter() - t0)
        faiss_search = np.mean(faiss_times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)

        recall_cpp = recall_faiss = 0.0
        for i in range(nq):
            gt = set(l_flat_arr[i].tolist())
            recall_cpp += len(set(l_cpp_arr[i].tolist()) & gt) / len(gt) if gt else 0
            recall_faiss += len(set(l_faiss_arr[i].tolist()) & gt) / len(gt) if gt else 0
        recall_cpp /= nq
        recall_faiss /= nq

        train_spd = faiss_train / cpp_train
        search_spd = faiss_search / cpp_search
        results.append((dim, n, nprobe, train_spd, search_spd, recall_cpp, recall_faiss))
        print(f'  Dim={dim}, N={n}, nprobe={nprobe}: Train={train_spd:.2f}x, Search={search_spd:.2f}x, Recall C++={recall_cpp:.1%} FAISS={recall_faiss:.1%}')

    return results

def bench_hnsw():
    print('\n' + '='*80)
    print('  Algorithm 4: HNSW - C++ vs FAISS')
    print('='*80)
    results = []
    configs = [
        (128, 50000, 100, 10, 16, 200, 128),
    ]
    for dim, n, nq, k, M, ef_c, ef_s in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(nq, dim).astype(np.float32)

        idx_cpp = _hnsw.IndexHNSW(dim, M, ef_c)
        t0 = time.perf_counter(); idx_cpp.add(vectors); cpp_add = time.perf_counter() - t0
        idx_cpp.set_ef_search(ef_s)
        for _ in range(3): idx_cpp.search(queries, k)
        cpp_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k); cpp_times.append(time.perf_counter() - t0)
        cpp_search = np.mean(cpp_times)

        idx_faiss = faiss.IndexHNSWFlat(dim, M)
        idx_faiss.hnsw.efConstruction = ef_c
        t0 = time.perf_counter(); idx_faiss.add(vectors); faiss_add = time.perf_counter() - t0
        idx_faiss.hnsw.efSearch = ef_s
        for _ in range(3): idx_faiss.search(queries, k)
        faiss_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k); faiss_times.append(time.perf_counter() - t0)
        faiss_search = np.mean(faiss_times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)

        recall_cpp = recall_faiss = 0.0
        for i in range(nq):
            gt = set(l_flat_arr[i].tolist())
            recall_cpp += len(set(l_cpp_arr[i].tolist()) & gt) / len(gt) if gt else 0
            recall_faiss += len(set(l_faiss_arr[i].tolist()) & gt) / len(gt) if gt else 0
        recall_cpp /= nq
        recall_faiss /= nq

        add_spd = faiss_add / cpp_add
        search_spd = faiss_search / cpp_search
        results.append((dim, n, add_spd, search_spd, recall_cpp, recall_faiss))
        print(f'  Dim={dim}, N={n}: Add={add_spd:.2f}x, Search={search_spd:.2f}x, Recall C++={recall_cpp:.1%} FAISS={recall_faiss:.1%}')

    return results

def bench_pq():
    print('\n' + '='*80)
    print('  Algorithm 5: PQ - C++ vs FAISS')
    print('='*80)
    results = []
    configs = [
        (128, 100000, 100, 10, 8, 8),
        (128, 100000, 100, 10, 16, 8),
    ]
    for dim, n, nq, k, M, nbits in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(nq, dim).astype(np.float32)

        idx_cpp = _pq.IndexPQ(dim, M, nbits)
        t0 = time.perf_counter(); idx_cpp.train(vectors); cpp_train = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_cpp.add(vectors); cpp_add = time.perf_counter() - t0
        for _ in range(3): idx_cpp.search(queries, k)
        cpp_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k); cpp_times.append(time.perf_counter() - t0)
        cpp_search = np.mean(cpp_times)

        idx_faiss = faiss.IndexPQ(dim, M, nbits)
        t0 = time.perf_counter(); idx_faiss.train(vectors); faiss_train = time.perf_counter() - t0
        t0 = time.perf_counter(); idx_faiss.add(vectors); faiss_add = time.perf_counter() - t0
        for _ in range(3): idx_faiss.search(queries, k)
        faiss_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k); faiss_times.append(time.perf_counter() - t0)
        faiss_search = np.mean(faiss_times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)

        recall_cpp = recall_faiss = 0.0
        for i in range(nq):
            gt = set(l_flat_arr[i].tolist())
            recall_cpp += len(set(l_cpp_arr[i].tolist()) & gt) / len(gt) if gt else 0
            recall_faiss += len(set(l_faiss_arr[i].tolist()) & gt) / len(gt) if gt else 0
        recall_cpp /= nq
        recall_faiss /= nq

        train_spd = faiss_train / cpp_train
        search_spd = faiss_search / cpp_search
        results.append((dim, n, M, train_spd, search_spd, recall_cpp, recall_faiss))
        print(f'  Dim={dim}, N={n}, M={M}: Train={train_spd:.2f}x, Search={search_spd:.2f}x, Recall C++={recall_cpp:.1%} FAISS={recall_faiss:.1%}')

    return results

def bench_lsh():
    print('\n' + '='*80)
    print('  Algorithm 6: LSH - C++ vs FAISS')
    print('='*80)
    print('  NOTE: C++ uses multi-table multi-probe LSH; FAISS uses single-table LSH')
    print('  The implementations are fundamentally different, so direct comparison is not meaningful')
    results = []
    configs = [
        (128, 100000, 100, 10, 128),
    ]
    for dim, n, nq, k, nbits in configs:
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        queries = np.random.randn(nq, dim).astype(np.float32)

        idx_cpp = _lsh.IndexLSH(dim, nbits, 4)
        t0 = time.perf_counter(); idx_cpp.add(vectors); cpp_add = time.perf_counter() - t0
        for _ in range(3): idx_cpp.search(queries, k)
        cpp_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_cpp, l_cpp = idx_cpp.search(queries, k); cpp_times.append(time.perf_counter() - t0)
        cpp_search = np.mean(cpp_times)

        idx_faiss = faiss.IndexLSH(dim, nbits)
        t0 = time.perf_counter(); idx_faiss.add(vectors); faiss_add = time.perf_counter() - t0
        for _ in range(3): idx_faiss.search(queries, k)
        faiss_times = []
        for _ in range(5):
            t0 = time.perf_counter(); d_faiss, l_faiss = idx_faiss.search(queries, k); faiss_times.append(time.perf_counter() - t0)
        faiss_search = np.mean(faiss_times)

        idx_flat = faiss.IndexFlatL2(dim)
        idx_flat.add(vectors)
        _, l_flat = idx_flat.search(queries, k)

        l_cpp_arr = np.array(l_cpp, dtype=np.int64)
        l_faiss_arr = np.array(l_faiss, dtype=np.int64)
        l_flat_arr = np.array(l_flat, dtype=np.int64)

        recall_cpp = recall_faiss = 0.0
        for i in range(nq):
            gt = set(l_flat_arr[i].tolist())
            recall_cpp += len(set(l_cpp_arr[i].tolist()) & gt) / len(gt) if gt else 0
            recall_faiss += len(set(l_faiss_arr[i].tolist()) & gt) / len(gt) if gt else 0
        recall_cpp /= nq
        recall_faiss /= nq

        add_spd = faiss_add / cpp_add
        search_spd = faiss_search / cpp_search
        results.append((dim, n, add_spd, search_spd, recall_cpp, recall_faiss))
        print(f'  Dim={dim}, N={n}: Add={add_spd:.2f}x, Search={search_spd:.2f}x, Recall C++={recall_cpp:.1%} FAISS={recall_faiss:.1%}')

    return results

if __name__ == '__main__':
    print('='*80)
    print('  COMPREHENSIVE BENCHMARK: C++ vs FAISS')
    print('='*80)

    r1 = bench_flat_l2()
    r2 = bench_flat_ip()
    r3 = bench_ivf()
    r4 = bench_hnsw()
    r5 = bench_pq()
    r6 = bench_lsh()

    print('\n' + '='*80)
    print('  SUMMARY')
    print('='*80)
    print()
    print('  Algorithm  | Search Speedup vs FAISS | Status')
    print('  -----------+-------------------------+--------')
    
    flat_search = np.mean([r[3] for r in r1])
    print(f'  FlatL2     | {flat_search:.2f}x                   | {"✓ FASTER" if flat_search > 1 else "✗ SLOWER"}')
    
    flat_ip_search = np.mean([r[3] for r in r2])
    print(f'  FlatIP     | {flat_ip_search:.2f}x                   | {"✓ FASTER" if flat_ip_search > 1 else "✗ SLOWER"}')
    
    ivf_search = np.mean([r[4] for r in r3])
    print(f'  IVF        | {ivf_search:.2f}x                   | {"✓ FASTER" if ivf_search > 1 else "✗ SLOWER"}')
    
    hnsw_search = np.mean([r[4] for r in r4])
    print(f'  HNSW       | {hnsw_search:.2f}x                   | {"✓ FASTER" if hnsw_search > 1 else "✗ SLOWER"}')
    
    pq_search = np.mean([r[4] for r in r5])
    print(f'  PQ         | {pq_search:.2f}x                   | {"✓ FASTER" if pq_search > 1 else "✗ SLOWER"}')
    
    lsh_search = np.mean([r[4] for r in r6])
    print(f'  LSH        | {lsh_search:.2f}x                   | DIFFERENT APPROACH')
    print()
