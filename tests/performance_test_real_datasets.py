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
            true_indices = set(ground_truth[i][:top_k])
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

def read_fvecs(file):
    """读取fvecs文件"""
    with open(file, 'rb') as f:
        while True:
            try:
                dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
                vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                yield vec
            except:
                break

def read_ivecs(file):
    """读取ivecs文件"""
    with open(file, 'rb') as f:
        while True:
            try:
                dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
                vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
                yield vec
            except:
                break

def load_dataset(dataset_name, max_vectors=50000):
    """加载数据集"""
    data_dir = "data/datasets"
    
    if dataset_name == "sift1m":
        base_file = os.path.join(data_dir, 'sift1m', 'sift_base.fvecs')
        query_file = os.path.join(data_dir, 'sift1m', 'sift_query.fvecs')
        groundtruth_file = os.path.join(data_dir, 'sift1m', 'sift_groundtruth.ivecs')
    elif dataset_name == "gist1m":
        base_file = os.path.join(data_dir, 'gist', 'gist_base.fvecs')
        query_file = os.path.join(data_dir, 'gist', 'gist_query.fvecs')
        groundtruth_file = os.path.join(data_dir, 'gist', 'gist_groundtruth.ivecs')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loading {dataset_name} dataset...")
    
    # 读取base向量
    base_vectors = list(read_fvecs(base_file))
    # 读取query向量
    query_vectors = list(read_fvecs(query_file))
    # 读取groundtruth
    groundtruth = list(read_ivecs(groundtruth_file))
    
    # 限制向量数量以节省时间和内存
    base_vectors = base_vectors[:max_vectors]
    
    print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
    return base_vectors, query_vectors, groundtruth

def generate_metadata(count: int):
    """生成测试元数据"""
    return [{"id": i, "category": f"cat_{i % 10}"} for i in range(count)]

if __name__ == "__main__":
    # 导入7VecDB客户端
    from core.vector_db.faiss_vector_db import FAISSVectorDB
    
    # 初始化数据库客户端
    db_client = FAISSVectorDB(db_path="./data/vector_db")
    
    # 测试参数
    datasets = ["sift1m", "gist1m"]
    max_vectors_list = [5000, 50000]  # 5K, 50K向量
    
    for dataset in datasets:
        for max_vectors in max_vectors_list:
            print(f"\n=== Testing {dataset} with {max_vectors} vectors ===")
            
            # 加载数据集
            base_vectors, query_vectors, groundtruth = load_dataset(dataset, max_vectors=max_vectors)
            
            # 转换为列表格式
            vectors = base_vectors
            query_vectors = query_vectors
            groundtruth = groundtruth
            
            # 生成元数据
            metadata = generate_metadata(len(vectors))
            
            # 初始化性能测试器，使用不同的集合名称
            collection_name = f"test_{dataset}_{max_vectors}"
            
            # 清理旧集合
            try:
                db_client.delete_collection(collection_name)
            except:
                pass
            
            # 创建新集合
            dimension = len(vectors[0])
            db_client.create_collection(collection_name, dimension)
            
            # 初始化性能测试器
            tester = PerformanceTester(db_client, f"7VecDB-FAISS-{dataset}")
            
            # 测试插入性能
            tester.test_insert(collection_name, vectors, metadata)
            
            # 测试搜索性能
            for top_k in [1, 5, 10, 50]:
                tester.test_search(collection_name, query_vectors, top_k, groundtruth)
            
            # 测试并发搜索
            for concurrency in [1, 5, 50]:
                tester.test_concurrent_search(collection_name, query_vectors[:100], 10, concurrency)
            
            # 保存结果
            tester.save_results(f"{dataset}_{max_vectors}")
            
            # 清理测试集合
            try:
                db_client.delete_collection(collection_name)
            except:
                pass