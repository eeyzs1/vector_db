#!/usr/bin/env python3
"""
Performance test comparing C++/Rust vector DB implementations with FAISS
"""

import time
import numpy as np
import psutil
import os
import sys

# Add parent directory to path to find vector_db_cpp
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

def get_memory_usage():
    """Get current memory usage in MB"""
    import gc
    # 强制垃圾回收，确保内存状态稳定
    gc.collect()
    # 等待一小段时间，让系统完成内存回收
    time.sleep(0.1)
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_performance(dimensions=128, num_vectors_list=[10000, 100000], num_queries=100, k=10):
    """Test performance of vector DB implementations"""
    print(f"Performance test with dimension: {dimensions}")
    print(f"Number of queries: {num_queries}, k: {k}")
    print("=" * 80)
    
    for num_vectors in num_vectors_list:
        print(f"\nTesting with {num_vectors:,} vectors:")
        print("-" * 60)
        
        # Generate test data
        print("Generating test data...")
        vectors = np.random.rand(num_vectors, dimensions).astype(np.float64)
        queries = np.random.rand(num_queries, dimensions).astype(np.float64)
        
        # Test C++ implementation
        if vector_db_cpp:
            print("\nTesting C++ implementation:")
            start_mem = get_memory_usage()
            start_time = time.time()
            
            index_cpp = vector_db_cpp.IndexFlatL2(dimensions)
            index_cpp.add(vectors)
            
            add_time = time.time() - start_time
            add_mem = get_memory_usage() - start_mem
            
            # Test search
            start_time = time.time()
            for i in range(num_queries):
                distances, labels = index_cpp.search(queries[i:i+1], k)
            search_time = time.time() - start_time
            
            print(f"  Add time: {add_time:.4f}s")
            print(f"  Search time: {search_time:.4f}s")
            print(f"  Memory usage: {add_mem:.2f}MB")
        
        # Test Rust implementation
        if vector_db_rust:
            print("\nTesting Rust implementation:")
            start_mem = get_memory_usage()
            start_time = time.time()
            
            index_rust = vector_db_rust.FlatIndex()
            # 尝试直接传递numpy数组，而不是转换为列表
            try:
                index_rust.add(vectors)
            except Exception as e:
                # 如果失败，回退到列表转换
                index_rust.add(vectors.tolist())
            
            add_time = time.time() - start_time
            add_mem = get_memory_usage() - start_mem
            
            # Test search
            start_time = time.time()
            for i in range(num_queries):
                # 尝试直接传递numpy数组，而不是转换为列表
                try:
                    distances, labels = index_rust.search(queries[i:i+1], k)
                except Exception as e:
                    # 如果失败，回退到列表转换
                    distances, labels = index_rust.search(queries[i].tolist(), k)
            search_time = time.time() - start_time
            
            print(f"  Add time: {add_time:.4f}s")
            print(f"  Search time: {search_time:.4f}s")
            print(f"  Memory usage: {add_mem:.2f}MB")
        
        # Test FAISS
        if faiss:
            print("\nTesting FAISS implementation:")
            start_mem = get_memory_usage()
            start_time = time.time()
            
            index_faiss = faiss.IndexFlatL2(dimensions)
            index_faiss.add(vectors)
            
            add_time = time.time() - start_time
            add_mem = get_memory_usage() - start_mem
            
            # Test search
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
    
    # Pre-generate all vectors
    print("Generating test data...")
    all_vectors = np.random.rand(max_vectors, dimensions).astype(np.float64)
    queries = np.random.rand(num_queries, dimensions).astype(np.float64)
    
    for num_vectors in range(step, max_vectors + 1, step):
        print(f"\nTesting with {num_vectors:,} vectors:")
        print("-" * 60)
        
        vectors = all_vectors[:num_vectors]
        
        # Test C++ implementation
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
        
        # Test Rust implementation
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
        
        # Test FAISS
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

if __name__ == "__main__":
    # Install psutil if not available
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        os.system("pip install psutil")
        import psutil
    
    # Install faiss if not available
    if faiss is None:
        print("Installing faiss...")
        os.system("pip install faiss-cpu")
        import faiss
    
    # Run performance tests
    test_performance()
    
    # Run scalability test (if memory allows)
    try:
        test_scalability(max_vectors=500000, step=100000)
    except MemoryError:
        print("\nMemory error: Reducing test size...")
        test_scalability(max_vectors=200000, step=50000)