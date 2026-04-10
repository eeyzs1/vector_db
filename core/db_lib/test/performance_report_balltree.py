#!/usr/bin/env python3
"""
Ball-Tree performance report comparing C++/Rust implementations
"""

import time
import numpy as np
import psutil
import os
import sys
import platform
import cpuinfo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

try:
    from vectordb.cpp import _balltree
    cpp_available = True
except ImportError:
    cpp_available = False

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

def test_balltree_performance(dimensions=128, num_vectors=100000, num_queries=100, k=10, leaf_size=40):
    print(f"Ball-Tree Performance Test")
    print(f"Dimension: {dimensions}, Vectors: {num_vectors:,}, Queries: {num_queries}, k: {k}")
    print(f"Leaf size: {leaf_size}")
    print("=" * 80)

    print("\nGenerating test data...")
    vectors = np.random.rand(num_vectors, dimensions).astype(np.float32)
    queries = np.random.rand(num_queries, dimensions).astype(np.float32)

    if cpp_available:
        print("\nTesting C++ Ball-Tree implementation:")
        try:
            start_time = time.time()
            index = _balltree.IndexBallTree(dimensions, leaf_size)
            create_time = time.time() - start_time
            print(f"  Index creation: {create_time:.4f}s")

            start_time = time.time()
            index.add(vectors)
            add_time = time.time() - start_time
            print(f"  Add time: {add_time:.4f}s")

            start_time = time.time()
            distances, labels = index.search(queries, k)
            search_time = time.time() - start_time
            print(f"  Search time: {search_time:.4f}s")
            print(f"  Queries per second: {num_queries / search_time:.2f} QPS")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("Ball-Tree test completed!")

def run_comprehensive_test():
    system_info = get_system_info()
    print("\nSYSTEM INFORMATION:")
    print("-" * 60)
    for key, value in system_info.items():
        print(f"{key}: {value}")

    test_configs = [
        (128, 50000, 100, 10, 40),
        (128, 100000, 100, 10, 40),
        (256, 50000, 100, 10, 40),
    ]

    for dim, num_vec, num_q, k, leaf_size in test_configs:
        test_balltree_performance(dim, num_vec, num_q, k, leaf_size)

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
