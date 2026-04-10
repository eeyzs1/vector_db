#!/usr/bin/env python3
"""
IVF (Inverted File) performance report comparing C++/Rust implementations
"""

import time
import numpy as np
import psutil
import os
import sys
import platform
import cpuinfo

# Add paths to find implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db_lib_cpp')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db_lib_rust')))

# Try to import all implementations
try:
    from vectordb.cpp import _ivf
    cpp_available = True
except ImportError:
    cpp_available = False

try:
    import vector_db_cpp
    original_cpp_available = True
except ImportError:
    original_cpp_available = False

try:
    import vector_db_rust
    original_rust_available = True
except ImportError:
    original_rust_available = False

def get_memory_usage_simple():
    import gc
    gc.collect()
    time.sleep(0.1)
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_system_info():
    info = {}
    info['platform'] = platform.platform()
    info['python_version'] = platform.python_version()
    try:
        info['cpu'] = cpuinfo.get_cpu_info()['brand_raw']
    except:
        info['cpu'] = 'Unknown'
    info['cpu_count'] = os.cpu_count()
    info['total_memory'] = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    return info

def test_ivf_performance(dimensions=128, num_vectors=100000, num_queries=100, k=10, nlist=100, nprobe=10):
    print(f"IVF Performance Test")
    print(f"Dimension: {dimensions}, Vectors: {num_vectors:,}, Queries: {num_queries}, k: {k}")
    print(f"nlist: {nlist}, nprobe: {nprobe}")
    print("=" * 80)

    print("\nGenerating test data...")
    vectors = np.random.rand(num_vectors, dimensions).astype(np.float32)
    queries = np.random.rand(num_queries, dimensions).astype(np.float32)
    training_data = np.random.rand(min(num_vectors // 10, 10000), dimensions).astype(np.float32)

    if cpp_available:
        print("\nTesting C++ IVF implementation:")
        try:
            start_time = time.time()
            index = _ivf.IndexIVF(dimensions, nlist)
            create_time = time.time() - start_time
            print(f"  Index creation: {create_time:.4f}s")

            start_time = time.time()
            index.train(training_data)
            train_time = time.time() - start_time
            print(f"  Train time: {train_time:.4f}s")

            start_time = time.time()
            index.add(vectors)
            add_time = time.time() - start_time
            print(f"  Add time: {add_time:.4f}s")

            index.set_nprobe(nprobe)

            start_time = time.time()
            distances, labels = index.search(queries, k)
            search_time = time.time() - start_time
            print(f"  Search time: {search_time:.4f}s")
            print(f"  Queries per second: {num_queries / search_time:.2f} QPS")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("IVF test completed!")

def run_comprehensive_test():
    system_info = get_system_info()
    print("\nSYSTEM INFORMATION:")
    print("-" * 60)
    for key, value in system_info.items():
        print(f"{key}: {value}")

    test_configs = [
        (128, 50000, 100, 10, 100, 10),
        (128, 100000, 100, 10, 100, 10),
        (256, 50000, 100, 10, 100, 10),
    ]

    for dim, num_vec, num_q, k, nlist, nprobe in test_configs:
        test_ivf_performance(dim, num_vec, num_q, k, nlist, nprobe)

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        os.system("pip install psutil")
        import psutil

    try:
        import cpuinfo
    except ImportError:
        print("Installing cpuinfo...")
        os.system("pip install py-cpuinfo")
        import cpuinfo

    run_comprehensive_test()
