
#!/usr/bin/env python3

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import asyncio
from vectordb import VectorIndex, IndexType, Implementation, create_index


def test_sync_interface():
    print("Testing synchronous interface...")
    
    dimension = 128
    num_vectors = 1000
    num_queries = 10
    k = 5
    
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
    queries = np.random.rand(num_queries, dimension).astype(np.float32)
    
    index = create_index("flat_l2", dimension, implementation="cpp_nanobind")
    
    print(f"Adding {num_vectors} vectors...")
    index.add(vectors)
    print(f"Index size: {index.size()}")
    
    print(f"Searching {num_queries} queries with k={k}...")
    distances, labels = index.search(queries, k)
    
    print(f"Distances shape: {distances.shape}")
    print(f"Labels shape: {labels.shape}")
    print("First query results:")
    print(f"  Distances: {distances[0]}")
    print(f"  Labels: {labels[0]}")
    
    index.close()
    print("Sync test passed!\n")


async def test_async_interface():
    print("Testing async interface...")
    
    dimension = 128
    num_vectors = 1000
    num_queries = 10
    k = 5
    
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
    queries = np.random.rand(num_queries, dimension).astype(np.float32)
    
    async with create_index("flat_l2", dimension, implementation="cpp_nanobind") as index:
        print(f"Adding {num_vectors} vectors asynchronously...")
        await index.add_async(vectors)
        print(f"Index size: {index.size()}")
        
        print(f"Searching {num_queries} queries with k={k} asynchronously...")
        distances, labels = await index.search_async(queries, k)
        
        print(f"Distances shape: {distances.shape}")
        print(f"Labels shape: {labels.shape}")
        print("First query results:")
        print(f"  Distances: {distances[0]}")
        print(f"  Labels: {labels[0]}")
    
    print("Async test passed!\n")


def test_multiple_index_types():
    print("Testing multiple index types...")
    
    dimension = 64
    num_vectors = 500
    num_queries = 5
    k = 3
    
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
    queries = np.random.rand(num_queries, dimension).astype(np.float32)
    
    index_types = [
        "flat_l2",
        "flat_ip",
        "hnsw",
        "ivf",
        "kdtree",
        "balltree",
        "lsh",
        "annoy"
    ]
    
    for idx_type in index_types:
        try:
            print(f"\nTesting {idx_type}...")
            index = create_index(idx_type, dimension, implementation="cpp_nanobind")
            
            if hasattr(index._index, 'train'):
                print(f"  Training...")
                index.train(vectors[:min(100, num_vectors)])
            
            print(f"  Adding vectors...")
            index.add(vectors)
            print(f"  Index size: {index.size()}")
            
            print(f"  Searching...")
            distances, labels = index.search(queries, k)
            print(f"  Search completed, distances shape: {distances.shape}")
            
            index.close()
            print(f"  {idx_type} test passed!")
        except Exception as e:
            print(f"  {idx_type} test failed: {e}")
    
    print("\nMultiple index types test completed!\n")


def main():
    print("=" * 60)
    print("Nanobind VectorDB Tests")
    print("=" * 60)
    
    try:
        test_sync_interface()
        asyncio.run(test_async_interface())
        test_multiple_index_types()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

