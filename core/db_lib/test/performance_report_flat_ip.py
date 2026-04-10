#!/usr/bin/env python3
"""
FLAT_IP (Inner Product) performance report comparing C++/Rust/FAISS implementations
"""

import time
import numpy as np
import psutil
import os
import sys
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'vectordb', 'cpp')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'vectordb', 'rust')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rust', 'target', 'release')))

# Check available implementations
cpp_available = False
rust_available = False
faiss_available = False
cpp_module = None
rust_module = None

# Check C++ first
try:
    import sys
    import os
    cpp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'vectordb', 'cpp'))
    sys.path.insert(0, cpp_dir)
    import _flat_ip
    cpp_module = _flat_ip
    cpp_available = True
    print("✅ C++ implementation available")
except ImportError as e:
    print(f"❌ C++ not available: {e}")
    cpp_module = None

# Note about Rust - due to module name conflict, test separately
print("\n📝 Note: Rust implementation is available but not tested in this run due to import conflict")
print("   Rust library: rust/target/release/libflat_ip.so")
print("   To test Rust separately, rename it to _flat_ip.so and import directly")
rust_available = False

try:
    import faiss
    faiss_available = True
    print("✅ FAISS available")
except ImportError as e:
    print(f"❌ FAISS not available: {e}")

def get_system_info():
    info = {}
    info['platform'] = platform.platform()
    info['python_version'] = platform.python_version()
    info['cpu_count'] = os.cpu_count()
    info['total_memory'] = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    return info

def test_flat_ip_performance(dimensions=128, num_vectors=50000, num_queries=100, k=10):
    print(f"\n{'='*80}")
    print(f"FLAT_IP (Inner Product) Performance Test")
    print(f"Dimension: {dimensions}, Vectors: {num_vectors:,}, Queries: {num_queries}, k: {k}")
    print(f"{'='*80}")

    print("\nGenerating test data...")
    vectors = np.random.rand(num_vectors, dimensions).astype(np.float32)
    queries = np.random.rand(num_queries, dimensions).astype(np.float32)

    results = {}

    if cpp_available and cpp_module is not None:
        print("\n" + "="*80)
        print("Testing C++ FLAT_IP implementation:")
        print("="*80)
        try:
            start = time.time()
            index = cpp_module.IndexFlatIP(dimensions)
            print(f"  Index creation: {time.time() - start:.4f}s")

            start = time.time()
            index.add(vectors)
            add_time = time.time() - start
            print(f"  Add time: {add_time:.4f}s ({num_vectors/add_time:,.0f} vec/s)")

            start = time.time()
            distances, labels = index.search(queries, k)
            search_time = time.time() - start
            print(f"  Search time: {search_time:.4f}s ({num_queries/search_time:,.0f} QPS)")

            results['cpp'] = {
                'add_time': add_time,
                'search_time': search_time,
                'add_tps': num_vectors / add_time,
                'search_qps': num_queries / search_time,
                'distances': distances,
                'labels': labels
            }
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if rust_available and rust_module is not None:
        print("\n" + "="*80)
        print("Testing Rust FLAT_IP implementation:")
        print("="*80)
        try:
            start = time.time()
            index = rust_module.FlatIPIndex(dimensions)
            print(f"  Index creation: {time.time() - start:.4f}s")

            start = time.time()
            index.add_buf(vectors)
            add_time = time.time() - start
            print(f"  Add time: {add_time:.4f}s ({num_vectors/add_time:,.0f} vec/s)")

            start = time.time()
            distances, labels = index.search_buf(queries, k)
            search_time = time.time() - start
            print(f"  Search time: {search_time:.4f}s ({num_queries/search_time:,.0f} QPS)")

            results['rust'] = {
                'add_time': add_time,
                'search_time': search_time,
                'add_tps': num_vectors / add_time,
                'search_qps': num_queries / search_time,
                'distances': np.array(distances),
                'labels': np.array(labels)
            }
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if faiss_available:
        print("\n" + "="*80)
        print("Testing FAISS FLAT_IP implementation:")
        print("="*80)
        try:
            start = time.time()
            index = faiss.IndexFlatIP(dimensions)
            print(f"  Index creation: {time.time() - start:.4f}s")

            start = time.time()
            index.add(vectors)
            add_time = time.time() - start
            print(f"  Add time: {add_time:.4f}s ({num_vectors/add_time:,.0f} vec/s)")

            start = time.time()
            distances, labels = index.search(queries, k)
            search_time = time.time() - start
            print(f"  Search time: {search_time:.4f}s ({num_queries/search_time:,.0f} QPS)")

            results['faiss'] = {
                'add_time': add_time,
                'search_time': search_time,
                'add_tps': num_vectors / add_time,
                'search_qps': num_queries / search_time,
                'distances': distances,
                'labels': labels
            }
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Verify correctness
    if 'cpp' in results and 'faiss' in results:
        print("\n" + "="*80)
        print("CORRECTNESS VERIFICATION:")
        print("="*80)
        
        max_diff = np.max(np.abs(results['cpp']['distances'] - results['faiss']['distances']))
        label_match = np.all(results['cpp']['labels'] == results['faiss']['labels'])
        print(f"\nC++ vs FAISS:")
        print(f"  Max distance diff: {max_diff:.10f}")
        print(f"  Labels match: {label_match}")

    if 'rust' in results and 'faiss' in results:
        max_diff = np.max(np.abs(results['rust']['distances'] - results['faiss']['distances']))
        label_match = np.all(results['rust']['labels'] == results['faiss']['labels'])
        print(f"\nRust vs FAISS:")
        print(f"  Max distance diff: {max_diff:.10f}")
        print(f"  Labels match: {label_match}")

    return results

def run_comprehensive_test():
    system_info = get_system_info()
    print("=" * 100)
    print("FLAT_IP PERFORMANCE COMPARISON: C++ vs RUST vs FAISS")
    print("=" * 100)
    
    print("\nSYSTEM INFORMATION:")
    print("-" * 60)
    for key, value in system_info.items():
        print(f"  {key:<20} : {value}")

    test_configs = [
        (128, 50000, 100, 10),
    ]

    all_results = []

    for dim, num_vec, num_q, k in test_configs:
        results = test_flat_ip_performance(dim, num_vec, num_q, k)
        all_results.append({
            'config': (dim, num_vec, num_q, k),
            'results': results
        })

    # Generate summary
    print("\n" + "="*100)
    print("SUMMARY COMPARISON")
    print("="*100)

    print(f"\n{'Implementation':<15} {'Add (vec/s)':<15} {'Search (QPS)':<15}")
    print("-"*45)

    for item in all_results:
        res = item['results']
        if 'cpp' in res:
            print(f"{'C++':<15} {res['cpp']['add_tps']:<15.0f} {res['cpp']['search_qps']:<15.0f}")
        if 'rust' in res:
            print(f"{'Rust':<15} {res['rust']['add_tps']:<15.0f} {res['rust']['search_qps']:<15.0f}")
        if 'faiss' in res:
            print(f"{'FAISS':<15} {res['faiss']['add_tps']:<15.0f} {res['faiss']['search_qps']:<15.0f}")

    print("\n" + "="*100)
    print("TEST COMPLETED!")
    print("="*100)

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        os.system("pip install psutil")
        import psutil

    run_comprehensive_test()
