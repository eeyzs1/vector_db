#!/usr/bin/env python3
"""
Combined performance report comparing C++/Rust vector DB implementations with FAISS
"""

import time
import numpy as np
import psutil
import os
import sys
import platform
import cpuinfo

# Add parent directory to path to find vector_db_cpp and vector_db_rust
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db_lib_cpp')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db_lib_rust')))

# Try to import all implementations
try:
    import vector_db_cpp
except ImportError:
    print("Warning: vector_db_cpp not available")
    vector_db_cpp = None

try:
    import vector_db_rust
except ImportError:
    print("Warning: vector_db_rust not available")
    vector_db_rust = None

try:
    import faiss
except ImportError:
    print("Warning: faiss not available")
    faiss = None

def get_memory_usage_simple():
    """Get current memory usage in MB"""
    import gc
    gc.collect()
    time.sleep(0.1)
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_memory_usage(samples=5, interval=0.05):
    """Get current memory usage in MB with multiple samples
    
    Args:
        samples: Number of samples to take
        interval: Time interval between samples in seconds
    
    Returns:
        Average memory usage in MB
    """
    import gc
    import time
    
    gc.collect()
    time.sleep(0.1)
    
    process = psutil.Process(os.getpid())
    memory_samples = []
    
    for _ in range(samples):
        mem = process.memory_info().rss / (1024 * 1024)
        memory_samples.append(mem)
        time.sleep(interval)
    
    return sum(memory_samples) / len(memory_samples)

def get_memory_peak(operation, *args, **kwargs):
    """Measure peak memory usage during an operation
    
    Args:
        operation: Function to execute
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation
    
    Returns:
        (result, peak_memory) - Tuple of operation result and peak memory usage in MB
    """
    import gc
    import time
    
    gc.collect()
    time.sleep(0.2)
    
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss / (1024 * 1024)
    peak = baseline
    
    def monitor_memory():
        nonlocal peak
        result = operation(*args, **kwargs)
        for _ in range(10):
            current = process.memory_info().rss / (1024 * 1024)
            if current > peak:
                peak = current
            time.sleep(0.01)
        return result
    
    result = monitor_memory()
    
    actual_usage = peak - baseline
    return result, actual_usage

def get_system_info():
    """Get system information"""
    info = {}
    info['platform'] = platform.platform()
    info['python_version'] = platform.python_version()
    info['cpu'] = cpuinfo.get_cpu_info()['brand_raw']
    info['cpu_count'] = os.cpu_count()
    info['total_memory'] = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    return info

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    return vectors / norms

def test_performance(dimensions=128, num_vectors_list=[10000, 100000], num_queries=100, k=10):
    """Test performance of vector DB implementations"""
    print(f"Performance test with dimension: {dimensions}")
    print(f"Number of queries: {num_queries}, k: {k}")
    print("=" * 80)
    
    for num_vectors in num_vectors_list:
        print(f"\nTesting with {num_vectors:,} vectors:")
        print("-" * 60)
        
        print("Generating test data...")
        vectors = np.random.rand(num_vectors, dimensions).astype(np.float64)
        vectors = normalize_vectors(vectors)
        queries = np.random.rand(num_queries, dimensions).astype(np.float64)
        queries = normalize_vectors(queries)
        
        if vector_db_cpp:
            print("\nTesting C++ implementation:")
            start_mem = get_memory_usage_simple()
            start_time = time.time()
            
            index_cpp = vector_db_cpp.IndexFlatL2(dimensions)
            index_cpp.add(vectors)
            
            add_time = time.time() - start_time
            add_mem = get_memory_usage_simple() - start_mem
            
            start_time = time.time()
            for i in range(num_queries):
                distances, labels = index_cpp.search(queries[i:i+1], k)
            search_time = time.time() - start_time
            
            print(f"  Add time: {add_time:.4f}s")
            print(f"  Search time: {search_time:.4f}s")
            print(f"  Memory usage: {add_mem:.2f}MB")
        
        if vector_db_rust:
            print("\nTesting Rust implementation:")
            start_mem = get_memory_usage_simple()
            start_time = time.time()
            
            index_rust = vector_db_rust.FlatIndex()
            try:
                index_rust.add(vectors)
            except Exception as e:
                index_rust.add(vectors.tolist())
            
            add_time = time.time() - start_time
            add_mem = get_memory_usage_simple() - start_mem
            
            start_time = time.time()
            for i in range(num_queries):
                try:
                    distances, labels = index_rust.search(queries[i:i+1], k)
                except Exception as e:
                    distances, labels = index_rust.search(queries[i].tolist(), k)
            search_time = time.time() - start_time
            
            print(f"  Add time: {add_time:.4f}s")
            print(f"  Search time: {search_time:.4f}s")
            print(f"  Memory usage: {add_mem:.2f}MB")
        
        if faiss:
            print("\nTesting FAISS implementation:")
            start_mem = get_memory_usage_simple()
            start_time = time.time()
            
            index_faiss = faiss.IndexFlatL2(dimensions)
            index_faiss.add(vectors)
            
            add_time = time.time() - start_time
            add_mem = get_memory_usage_simple() - start_mem
            
            start_time = time.time()
            distances, labels = index_faiss.search(queries, k)
            search_time = time.time() - start_time
            
            print(f"  Add time: {add_time:.4f}s")
            print(f"  Search time: {search_time:.4f}s")
            print(f"  Memory usage: {add_mem:.2f}MB")
    
    print("\n" + "=" * 80)
    print("Performance test completed!")

def test_scalability(dimensions=128, max_vectors=1000000, step=100000, num_queries=100, k=10):
    """Test scalability of implementations"""
    print(f"\nScalability test with dimension: {dimensions}")
    print(f"Max vectors: {max_vectors:,}, Step: {step:,}")
    print("=" * 80)
    
    print("Generating test data...")
    all_vectors = np.random.rand(max_vectors, dimensions).astype(np.float64)
    all_vectors = normalize_vectors(all_vectors)
    queries = np.random.rand(num_queries, dimensions).astype(np.float64)
    queries = normalize_vectors(queries)
    
    for num_vectors in range(step, max_vectors + 1, step):
        print(f"\nTesting with {num_vectors:,} vectors:")
        print("-" * 60)
        
        vectors = all_vectors[:num_vectors]
        
        if vector_db_cpp:
            start_time = time.time()
            index_cpp = vector_db_cpp.IndexFlatL2(dimensions)
            index_cpp.add(vectors)
            add_time = time.time() - start_time
            
            start_time = time.time()
            for i in range(num_queries):
                distances, labels = index_cpp.search(queries[i:i+1], k)
            search_time = time.time() - start_time
            
            print(f"  C++ - Add: {add_time:.4f}s, Search: {search_time:.4f}s")
        
        if vector_db_rust:
            start_time = time.time()
            index_rust = vector_db_rust.FlatIndex()
            index_rust.add(vectors.tolist())
            add_time = time.time() - start_time
            
            start_time = time.time()
            for i in range(num_queries):
                distances, labels = index_rust.search(queries[i].tolist(), k)
            search_time = time.time() - start_time
            
            print(f"  Rust - Add: {add_time:.4f}s, Search: {search_time:.4f}s")
        
        if faiss:
            start_time = time.time()
            index_faiss = faiss.IndexFlatL2(dimensions)
            index_faiss.add(vectors)
            add_time = time.time() - start_time
            
            start_time = time.time()
            distances, labels = index_faiss.search(queries, k)
            search_time = time.time() - start_time
            
            print(f"  FAISS - Add: {add_time:.4f}s, Search: {search_time:.4f}s")
    
    print("\n" + "=" * 80)
    print("Scalability test completed!")

def run_benchmark(dimensions=128, num_vectors=100000, num_queries=100, k=10):
    """Run benchmark for all implementations"""
    results = {}
    
    vectors = np.random.rand(num_vectors, dimensions).astype(np.float32)
    vectors = normalize_vectors(vectors)
    queries = np.random.rand(num_queries, dimensions).astype(np.float32)
    queries = normalize_vectors(queries)
    
    print(f"\nWarming up memory for {dimensions}D vectors...")
    warmup_mem = get_memory_usage()
    print(f"Warmup memory: {warmup_mem:.2f}MB")
    
    if vector_db_cpp:
        print(f"\nTesting C++ implementation...")
        
        start_time = time.time()
        index_cpp = vector_db_cpp.IndexFlatL2(dimensions)
        index_create_time = time.time() - start_time
        print(f"Index creation time: {index_create_time:.4f}s")
        
        def add_vectors_cpp():
            index_cpp.add(vectors)
        
        start_time = time.time()
        _, memory_usage = get_memory_peak(add_vectors_cpp)
        add_time = time.time() - start_time
        print(f"Add time: {add_time:.4f}s")
        print(f"Memory usage during add: {memory_usage:.2f}MB")
        
        final_mem = get_memory_usage(samples=10)
        print(f"Final memory after add: {final_mem:.2f}MB")
        
        start_time = time.time()
        distances, labels = index_cpp.search(queries, k)
        search_time = time.time() - start_time
        if search_time <= 0:
            search_time = 1e-6
        print(f"Search time: {search_time:.4f}s")
        
        if memory_usage <= 0:
            theoretical_mem = (num_vectors * dimensions * 8) / (1024 * 1024)
            print(f"Memory measurement issue, using theoretical value: {theoretical_mem:.2f}MB")
            memory_usage = theoretical_mem
        
        results['cpp'] = {
            'add_time': add_time,
            'search_time': search_time,
            'memory': memory_usage,
            'vectors_per_second': num_vectors / add_time if add_time > 0 else 0,
            'queries_per_second': num_queries / search_time if search_time > 0 else 0
        }
    
    if vector_db_rust:
        print(f"\nTesting Rust implementation...")
        
        start_time = time.time()
        index_rust = vector_db_rust.FlatIndex()
        index_create_time = time.time() - start_time
        print(f"Index creation time: {index_create_time:.4f}s")
        
        def add_vectors_rust():
            index_rust.add_buf(vectors)
        
        start_time = time.time()
        _, memory_usage = get_memory_peak(add_vectors_rust)
        add_time = time.time() - start_time
        print(f"Add time: {add_time:.4f}s")
        print(f"Memory usage during add: {memory_usage:.2f}MB")
        
        final_mem = get_memory_usage(samples=10)
        print(f"Final memory after add: {final_mem:.2f}MB")
        
        start_time = time.time()
        distances_batch, labels_batch = index_rust.search_batch_buf(queries, k)
        search_time = time.time() - start_time
        if search_time <= 0:
            search_time = 1e-6
        print(f"Search time: {search_time:.4f}s")
        
        if memory_usage <= 0:
            theoretical_mem = (num_vectors * dimensions * 8) / (1024 * 1024)
            print(f"Memory measurement issue, using theoretical value: {theoretical_mem:.2f}MB")
            memory_usage = theoretical_mem
        
        results['rust'] = {
            'add_time': add_time,
            'search_time': search_time,
            'memory': memory_usage,
            'vectors_per_second': num_vectors / add_time if add_time > 0 else 0,
            'queries_per_second': num_queries / search_time if search_time > 0 else 0
        }
    
    if faiss:
        print(f"\nTesting FAISS implementation...")
        
        start_time = time.time()
        index_faiss = faiss.IndexFlatL2(dimensions)
        index_create_time = time.time() - start_time
        print(f"Index creation time: {index_create_time:.4f}s")
        
        def add_vectors_faiss():
            index_faiss.add(vectors)
        
        start_time = time.time()
        _, memory_usage = get_memory_peak(add_vectors_faiss)
        add_time = time.time() - start_time
        print(f"Add time: {add_time:.4f}s")
        print(f"Memory usage during add: {memory_usage:.2f}MB")
        
        final_mem = get_memory_usage(samples=10)
        print(f"Final memory after add: {final_mem:.2f}MB")
        
        start_time = time.time()
        distances, labels = index_faiss.search(queries, k)
        search_time = time.time() - start_time
        if search_time <= 0:
            search_time = 1e-6
        print(f"Search time: {search_time:.4f}s")
        
        if memory_usage <= 0:
            theoretical_mem = (num_vectors * dimensions * 8) / (1024 * 1024)
            print(f"Memory measurement issue, using theoretical value: {theoretical_mem:.2f}MB")
            memory_usage = theoretical_mem
        
        results['faiss'] = {
            'add_time': add_time,
            'search_time': search_time,
            'memory': memory_usage,
            'vectors_per_second': num_vectors / add_time if add_time > 0 else 0,
            'queries_per_second': num_queries / search_time if search_time > 0 else 0
        }
    
    return results

def generate_report():
    """Generate detailed performance report"""
    print("=" * 80)
    print("VECTOR DATABASE PERFORMANCE COMPARISON")
    print("=" * 80)
    
    system_info = get_system_info()
    print("\nSYSTEM INFORMATION:")
    print("-" * 60)
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    test_configs = [
        (128, 100000),
        (128, 500000),
        (256, 100000),
        (256, 500000),
    ]
    
    for dim, num_vec in test_configs:
        print(f"\n\nTEST CONFIGURATION: {num_vec:,} vectors, {dim} dimensions")
        print("-" * 80)
        
        results = run_benchmark(dimensions=dim, num_vectors=num_vec)
        
        print("\nPERFORMANCE RESULTS:")
        print("-" * 80)
        print(f"{'Implementation':<15} {'Add Time (s)':<15} {'Search Time (s)':<15} {'Memory (MB)':<15} {'Vectors/s':<15} {'Queries/s':<15}")
        print("-" * 80)
        
        for impl, data in results.items():
            print(f"{impl:<15} {data['add_time']:<15.4f} {data['search_time']:<15.4f} {data['memory']:<15.2f} {data['vectors_per_second']:<15.0f} {data['queries_per_second']:<15.0f}")
        
        if 'faiss' in results:
            print("\nPERFORMANCE COMPARISON (relative to FAISS):")
            print("-" * 80)
            faiss_data = results['faiss']
            
            for impl, data in results.items():
                if impl != 'faiss':
                    add_ratio = data['add_time'] / faiss_data['add_time'] if faiss_data['add_time'] > 0 else 0
                    search_ratio = data['search_time'] / faiss_data['search_time'] if faiss_data['search_time'] > 0 else 0
                    memory_ratio = data['memory'] / faiss_data['memory'] if faiss_data['memory'] > 0 else 0
                    
                    print(f"{impl:<15} Add: {add_ratio:<10.2f}x Search: {search_ratio:<10.2f}x Memory: {memory_ratio:<10.2f}x")

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
        os.system("pip3 install py-cpuinfo")
        import cpuinfo
    
    if faiss is None:
        print("Installing faiss...")
        os.system("pip install faiss-cpu")
        import faiss
    
    test_performance()
    
    try:
        test_scalability(max_vectors=500000, step=100000)
    except MemoryError:
        print("\nMemory error: Reducing test size...")
        test_scalability(max_vectors=200000, step=50000)
    
    generate_report()
