import sys
import os
import time
import numpy as np
import json
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PerformanceTester:
    def __init__(self, db_client, name):
        self.db_client = db_client
        self.name = name
        self.results = {}
        self.results_dir = "tests/performance_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def test_insert(self, collection_name: str, vectors: List[List[float]], metadata: List[Dict[str, Any]], batch_size: int = 1000):
        """测试插入性能"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        inserted_ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            # 在元数据中添加原始索引
            for j, meta in enumerate(batch_metadata):
                meta['original_index'] = i + j
            batch_ids = self.db_client.insert(collection_name, batch_vectors, batch_metadata)
            inserted_ids.extend(batch_ids)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        elapsed = end_time - start_time
        throughput = len(vectors) / elapsed
        memory_used = end_memory - start_memory
        
        self.results['insert'] = {
            'time': elapsed,
            'throughput': throughput,
            'count': len(vectors),
            'memory_used': memory_used
        }
        
        print(f"[{self.name}] Insert performance: {throughput:.2f} vectors/sec, memory used: {memory_used:.2f} MB")
        return self.results['insert']
    
    def test_search(self, collection_name: str, query_vectors: List[List[float]], top_k: int = 10, ground_truth: List[List[int]] = None):
        """测试搜索性能和召回率"""
        start_time = time.time()
        results = []
        
        for query in query_vectors:
            result = self.db_client.search(collection_name, query, top_k)
            results.append(result)
        
        end_time = time.time()
        elapsed = end_time - start_time
        qps = len(query_vectors) / elapsed
        avg_latency = (elapsed / len(query_vectors)) * 1000  # ms
        
        # 计算召回率和精确率
        recall, precision, f1 = 0.0, 0.0, 0.0
        if ground_truth:
            recall, precision, f1 = self._calculate_metrics(results, ground_truth, top_k)
        
        self.results['search'] = {
            'time': elapsed,
            'qps': qps,
            'avg_latency': avg_latency,
            'count': len(query_vectors),
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        
        print(f"[{self.name}] Search performance: {qps:.2f} QPS, avg latency: {avg_latency:.2f} ms, recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}")
        return self.results['search']
    
    def _calculate_metrics(self, search_results: List[List[Dict]], ground_truth: List[List[int]], top_k: int) -> tuple:
        """计算召回率、精确率和F1分数"""
        total_recall = 0.0
        total_precision = 0.0
        
        for i, results in enumerate(search_results):
            # 假设ground_truth是向量的真实索引
            true_indices = set(ground_truth[i])
            retrieved_indices = set()
            
            # 提取检索到的向量索引
            for result in results:
                # 从元数据中获取原始索引
                if 'metadata' in result and 'original_index' in result['metadata']:
                    retrieved_indices.add(result['metadata']['original_index'])
            
            # 计算召回率和精确率
            if true_indices:
                recall = len(true_indices & retrieved_indices) / len(true_indices)
                total_recall += recall
            
            if retrieved_indices:
                precision = len(true_indices & retrieved_indices) / len(retrieved_indices)
                total_precision += precision
        
        avg_recall = total_recall / len(search_results)
        avg_precision = total_precision / len(search_results)
        
        # 计算F1分数
        if avg_recall + avg_precision > 0:
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        return avg_recall, avg_precision, f1
    
    def test_index_build(self, collection_name: str, vectors: List[List[float]]):
        """测试索引构建性能"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # 插入向量
        metadata = [{} for _ in vectors]
        self.db_client.insert(collection_name, vectors, metadata)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        elapsed = end_time - start_time
        memory_used = end_memory - start_memory
        
        self.results['index_build'] = {
            'time': elapsed,
            'memory_used': memory_used,
            'vector_count': len(vectors),
            'dimension': len(vectors[0])
        }
        
        print(f"[{self.name}] Index build performance: {elapsed:.2f} sec, memory used: {memory_used:.2f} MB")
        return self.results['index_build']
    
    def test_concurrent_search(self, collection_name: str, query_vectors: List[List[float]], top_k: int = 10, concurrency: int = 10):
        """测试并发搜索性能"""
        import concurrent.futures
        
        start_time = time.time()
        results = []
        
        def search_task(query):
            return self.db_client.search(collection_name, query, top_k)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_query = {executor.submit(search_task, query): query for query in query_vectors}
            for future in concurrent.futures.as_completed(future_to_query):
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        elapsed = end_time - start_time
        qps = len(query_vectors) / elapsed
        avg_latency = (elapsed / len(query_vectors)) * 1000  # ms
        
        self.results['concurrent_search'] = {
            'time': elapsed,
            'qps': qps,
            'avg_latency': avg_latency,
            'count': len(query_vectors),
            'concurrency': concurrency
        }
        
        print(f"[{self.name}] Concurrent search performance: {qps:.2f} QPS, avg latency: {avg_latency:.2f} ms (concurrency: {concurrency})")
        return self.results['concurrent_search']
    
    def save_results(self, test_name: str):
        """保存测试结果到文件"""
        filename = os.path.join(self.results_dir, f"{self.name}_{test_name}.json")
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def _get_memory_usage(self):
        """获取当前内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except:
            return 0

def generate_test_vectors(count: int, dimension: int):
    """生成测试向量"""
    return np.random.rand(count, dimension).tolist()

def generate_test_metadata(count: int):
    """生成测试元数据"""
    return [{"id": i, "category": f"cat_{i % 10}"} for i in range(count)]

def generate_ground_truth(query_vectors: List[List[float]], base_vectors: List[List[float]], top_k: int = 10):
    """生成真实的近邻结果作为基准"""
    ground_truth = []
    for query in query_vectors:
        # 计算与所有向量的距离
        distances = []
        for i, vector in enumerate(base_vectors):
            distance = np.linalg.norm(np.array(query) - np.array(vector))
            distances.append((distance, i))
        # 排序并取前top_k
        distances.sort()
        top_indices = [idx for _, idx in distances[:top_k]]
        ground_truth.append(top_indices)
    return ground_truth

if __name__ == "__main__":
    # 导入7VecDB客户端
    from core.vector_db.faiss_vector_db import FAISSVectorDB
    
    # 初始化数据库客户端
    db_client = FAISSVectorDB(db_path="./data/vector_db")
    
    # 测试参数
    test_sizes = [5000, 50000]  # 5K, 50K向量
    dimensions = [128, 384, 512]  # 不同维度
    
    for dimension in dimensions:
        for size in test_sizes:
            print(f"\n=== Testing {size} vectors with {dimension} dimensions ===")
            
            # 生成测试数据
            vectors = generate_test_vectors(size, dimension)
            metadata = generate_test_metadata(size)
            
            # 初始化性能测试器，使用不同的集合名称
            collection_name = f"test_collection_{dimension}"
            
            # 清理旧集合
            try:
                db_client.delete_collection(collection_name)
            except:
                pass
            
            # 创建新集合
            db_client.create_collection(collection_name, dimension)
            
            # 初始化性能测试器
            tester = PerformanceTester(db_client, f"7VecDB-FAISS-{dimension}")
            
            # 测试插入性能
            tester.test_insert(collection_name, vectors, metadata)
            
            # 生成查询向量
            num_queries = 100
            query_vectors = generate_test_vectors(num_queries, dimension)
            
            # 生成真实的近邻结果作为基准
            ground_truth = generate_ground_truth(query_vectors, vectors, top_k=50)
            
            # 测试搜索性能
            for top_k in [1, 5, 10, 50]:
                tester.test_search(collection_name, query_vectors, top_k, ground_truth)
            
            # 测试并发搜索
            for concurrency in [1, 5, 50]:
                tester.test_concurrent_search(collection_name, query_vectors[:100], 10, concurrency)
            
            # 保存结果
            tester.save_results(f"{size}_{dimension}")
            
            # 清理测试集合
            try:
                db_client.delete_collection(collection_name)
            except:
                pass