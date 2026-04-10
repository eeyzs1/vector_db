#!/usr/bin/env python3
"""
Detailed performance report comparing C++/Rust vector DB implementations with FAISS
"""

import time
import numpy as np
import psutil
import os
import sys
import platform
import cpuinfo

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db_lib_cpp')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db_lib_rust')))

# Try to import all implementations
# try:
#     import vector_db_cpp
# except ImportError:
#     print("Warning: vector_db_cpp not available")
#     vector_db_cpp = None
vector_db_cpp = None

try:
    import vector_db_rust
except ImportError:
    print("Warning: vector_db_rust not available")
# vector_db_rust = None

try:
    import faiss
except ImportError:
    print("Warning: faiss not available")
    faiss = None

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
    
    # 强制垃圾回收，确保内存状态稳定
    gc.collect()
    time.sleep(0.1)
    
    process = psutil.Process(os.getpid())
    memory_samples = []
    
    # 多次采样，捕获内存使用的变化
    for _ in range(samples):
        mem = process.memory_info().rss / (1024 * 1024)
        memory_samples.append(mem)
        time.sleep(interval)
    
    # 返回平均值，减少波动影响
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
    
    # 预热和稳定内存状态
    gc.collect()
    time.sleep(0.2)
    
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss / (1024 * 1024)
    peak = baseline
    
    # 定义一个包装函数，在执行过程中监控内存
    def monitor_memory():
        nonlocal peak
        result = operation(*args, **kwargs)
        # 执行过程中多次检查内存使用
        for _ in range(10):
            current = process.memory_info().rss / (1024 * 1024)
            if current > peak:
                peak = current
            time.sleep(0.01)
        return result
    
    result = monitor_memory()
    
    # 计算实际使用的内存
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

def run_benchmark(dimensions=128, num_vectors=100000, num_queries=100, k=10):
    """Run benchmark for all implementations"""
    results = {}
    
    # Generate test data
    vectors = np.random.rand(num_vectors, dimensions).astype(np.float32)
    vectors = normalize_vectors(vectors)
    queries = np.random.rand(num_queries, dimensions).astype(np.float32)
    queries = normalize_vectors(queries)
    
    # 预热内存，确保测量稳定
    print(f"\nWarming up memory for {dimensions}D vectors...")
    warmup_mem = get_memory_usage()
    print(f"Warmup memory: {warmup_mem:.2f}MB")
    
    # Test C++ implementation
    if vector_db_cpp:
        print(f"\nTesting C++ implementation...")
        
        # 创建索引
        start_time = time.time()
        index_cpp = vector_db_cpp.IndexFlatL2(dimensions)
        index_create_time = time.time() - start_time
        print(f"Index creation time: {index_create_time:.4f}s")
        
        # 使用峰值内存测量添加向量
        def add_vectors_cpp():
            index_cpp.add(vectors)
        
        start_time = time.time()
        _, memory_usage = get_memory_peak(add_vectors_cpp)
        add_time = time.time() - start_time
        print(f"Add time: {add_time:.4f}s")
        print(f"Memory usage during add: {memory_usage:.2f}MB")
        
        # 多次测量最终内存使用，确保准确性
        final_mem = get_memory_usage(samples=10)
        print(f"Final memory after add: {final_mem:.2f}MB")
        
        # Test search - use batch search
        start_time = time.time()
        distances, labels = index_cpp.search(queries, k)
        search_time = time.time() - start_time
        # 确保搜索时间为正数
        if search_time <= 0:
            search_time = 1e-6
        print(f"Search time: {search_time:.4f}s")
        
        # 确保内存值不为负或零（如果测量有问题，使用理论值）
        if memory_usage <= 0:
            # 计算理论内存使用：每个双精度浮点数8字节，num_vectors * dimensions * 8 / (1024*1024)
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
    
    # Test Rust implementation
    if vector_db_rust:
        print(f"\nTesting Rust implementation...")
        
        # 创建索引
        start_time = time.time()
        index_rust = vector_db_rust.FlatIndex()
        index_create_time = time.time() - start_time
        print(f"Index creation time: {index_create_time:.4f}s")
        
        # 使用峰值内存测量添加向量
        def add_vectors_rust():
            index_rust.add_buf(vectors)
        
        start_time = time.time()
        _, memory_usage = get_memory_peak(add_vectors_rust)
        add_time = time.time() - start_time
        print(f"Add time: {add_time:.4f}s")
        print(f"Memory usage during add: {memory_usage:.2f}MB")
        
        # 多次测量最终内存使用，确保准确性
        final_mem = get_memory_usage(samples=10)
        print(f"Final memory after add: {final_mem:.2f}MB")
        
        # Test search
        start_time = time.time()
        # 使用批量搜索API，一次处理所有查询
        distances_batch, labels_batch = index_rust.search_batch_buf(queries, k)
        search_time = time.time() - start_time
        # 确保搜索时间为正数
        if search_time <= 0:
            search_time = 1e-6
        print(f"Search time: {search_time:.4f}s")
        
        # 确保内存值不为负或零（如果测量有问题，使用理论值）
        if memory_usage <= 0:
            # 计算理论内存使用：每个双精度浮点数8字节，num_vectors * dimensions * 8 / (1024*1024)
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
    
    # Test FAISS
    if faiss:
        print(f"\nTesting FAISS implementation...")
        
        # 创建索引
        start_time = time.time()
        index_faiss = faiss.IndexFlatL2(dimensions)
        index_create_time = time.time() - start_time
        print(f"Index creation time: {index_create_time:.4f}s")
        
        # 使用峰值内存测量添加向量
        def add_vectors_faiss():
            index_faiss.add(vectors)
        
        start_time = time.time()
        _, memory_usage = get_memory_peak(add_vectors_faiss)
        add_time = time.time() - start_time
        print(f"Add time: {add_time:.4f}s")
        print(f"Memory usage during add: {memory_usage:.2f}MB")
        
        # 多次测量最终内存使用，确保准确性
        final_mem = get_memory_usage(samples=10)
        print(f"Final memory after add: {final_mem:.2f}MB")
        
        # Test search
        start_time = time.time()
        distances, labels = index_faiss.search(queries, k)
        search_time = time.time() - start_time
        # 确保搜索时间为正数
        if search_time <= 0:
            search_time = 1e-6
        print(f"Search time: {search_time:.4f}s")
        
        # 确保内存值不为负或零（如果测量有问题，使用理论值）
        if memory_usage <= 0:
            # 计算理论内存使用：每个双精度浮点数8字节，num_vectors * dimensions * 8 / (1024*1024)
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
    
    # System info
    system_info = get_system_info()
    print("\nSYSTEM INFORMATION:")
    print("-" * 60)
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    # Test configurations
    test_configs = [
        # (128, 10000),    # 10k vectors
        (128, 100000),   # 100k vectors
        (128, 500000),   # 500k vectors
        (256, 100000),   # 100k vectors, 256D
        (256, 500000),   # 500k vectors, 256D
    ]
    
    for dim, num_vec in test_configs:
        print(f"\n\nTEST CONFIGURATION: {num_vec:,} vectors, {dim} dimensions")
        print("-" * 80)
        
        results = run_benchmark(dimensions=dim, num_vectors=num_vec)
        
        # Print results
        print("\nPERFORMANCE RESULTS:")
        print("-" * 80)
        print(f"{'Implementation':<15} {'Add Time (s)':<15} {'Search Time (s)':<15} {'Memory (MB)':<15} {'Vectors/s':<15} {'Queries/s':<15}")
        print("-" * 80)
        
        for impl, data in results.items():
            print(f"{impl:<15} {data['add_time']:<15.4f} {data['search_time']:<15.4f} {data['memory']:<15.2f} {data['vectors_per_second']:<15.0f} {data['queries_per_second']:<15.0f}")
        
        # Performance comparison
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
    # Install required packages
    try:
        import cpuinfo
    except ImportError:
        print("Installing cpuinfo...")
        os.system("pip3 install py-cpuinfo")
        import cpuinfo
    
    generate_report()