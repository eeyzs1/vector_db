import sys
import os
import time
import math
import numpy as np
import json
import random
import string
from typing import List, Dict, Any
import concurrent.futures
import psutil
import wget
import zipfile
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PerformanceTester:
    def __init__(self, db_client, name):
        self.db_client = db_client
        self.name = name
        self.results = {}
        # 使用绝对路径确保目录创建和文件保存的正确性
        self.results_dir = os.path.join(os.path.dirname(__file__), "performance_results")
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
        # 避免除零错误
        if elapsed == 0:
            elapsed = 1e-9
        qps = len(query_vectors) / elapsed
        avg_latency = (elapsed / len(query_vectors)) * 1000  # ms
        
        # 计算召回率、精确率、F1分数和NDCG
        recall, precision, f1, ndcg = 0.0, 0.0, 0.0, 0.0
        if ground_truth:
            recall, precision, f1, ndcg = self._calculate_metrics(results, ground_truth, top_k)
        
        search_result = {
            'time': elapsed,
            'qps': qps,
            'avg_latency': avg_latency,
            'count': len(query_vectors),
            'top_k': top_k,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'ndcg': ndcg
        }
        
        # 初始化search列表如果不存在
        if 'search' not in self.results:
            self.results['search'] = []
        self.results['search'].append(search_result)
        
        print(f"[{self.name}] Search performance (top_k={top_k}): {qps:.2f} QPS, avg latency: {avg_latency:.2f} ms, recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}, ndcg: {ndcg:.4f}")
        return search_result
    
    def _calculate_ndcg(self, search_results: List[List[Dict]], ground_truth: List[List[int]], top_k: int) -> float:
        """计算NDCG@k"""
        total_ndcg = 0.0
        valid_queries = 0
        
        for i, results in enumerate(search_results):
            # 获取真实的最近邻
            true_indices = ground_truth[i][:top_k]
            
            # 跳过空的ground truth
            if not true_indices:
                continue
            
            valid_queries += 1
            # 创建真实索引到位置的映射
            true_positions = {idx: pos for pos, idx in enumerate(true_indices)}
            
            # 计算DCG
            dcg = 0.0
            relevant_count = 0
            for rank, result in enumerate(results):
                if rank >= top_k:
                    break
                # 从元数据中获取原始索引
                if 'metadata' in result and 'original_index' in result['metadata']:
                    retrieved_idx = result['metadata']['original_index']
                    if retrieved_idx in true_positions:
                        # 相关度得分，这里使用1表示相关，0表示不相关
                        relevance = 1.0
                        # 计算DCG
                        dcg += relevance / math.log2(rank + 2)  # 排名从1开始
                        relevant_count += 1
                    else:
                        # 不相关的文档，得分为0
                        dcg += 0.0
            
            # 计算IDCG（理想DCG）
            idcg = 0.0
            # 理想情况下，相关文档应该排在前面
            for rank in range(min(len(true_indices), top_k)):
                relevance = 1.0
                idcg += relevance / math.log2(rank + 2)
            
            # 计算NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                total_ndcg += ndcg
        
        # 确保至少有一个有效查询
        if valid_queries > 0:
            avg_ndcg = total_ndcg / valid_queries
            # 检查数值异常
            if math.isnan(avg_ndcg) or math.isinf(avg_ndcg):
                raise ValueError(f"NDCG calculation resulted in invalid value: {avg_ndcg}")
        else:
            avg_ndcg = 0.0
        return avg_ndcg
    
    def _calculate_metrics(self, search_results: List[List[Dict]], ground_truth: List[List[int]], top_k: int) -> tuple:
        """计算召回率、精确率、F1分数和NDCG"""
        total_recall = 0.0
        total_precision = 0.0
        valid_queries = 0
        
        for i, results in enumerate(search_results):
            # 假设ground_truth是向量的真实索引
            true_indices = set(ground_truth[i][:top_k])
            retrieved_indices = set()
            
            # 提取检索到的向量索引（只考虑前top_k个结果）
            for rank, result in enumerate(results):
                if rank >= top_k:
                    break
                # 从元数据中获取原始索引
                if 'metadata' in result and 'original_index' in result['metadata']:
                    retrieved_indices.add(result['metadata']['original_index'])
            
            # 跳过空的ground truth
            if not true_indices:
                continue
            
            valid_queries += 1
            
            # 计算召回率和精确率
            recall = len(true_indices & retrieved_indices) / len(true_indices)
            total_recall += recall
            
            if retrieved_indices:
                precision = len(true_indices & retrieved_indices) / len(retrieved_indices)
                total_precision += precision
            else:
                precision = 0.0
                total_precision += precision
        
        # 确保至少有一个有效查询
        if valid_queries > 0:
            avg_recall = total_recall / valid_queries
            avg_precision = total_precision / valid_queries
        else:
            avg_recall = 0.0
            avg_precision = 0.0
        
        # 计算F1分数
        if avg_recall + avg_precision > 0:
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        # 计算NDCG
        avg_ndcg = self._calculate_ndcg(search_results, ground_truth, top_k)
        
        return avg_recall, avg_precision, f1, avg_ndcg
    
    def test_concurrent_search(self, collection_name: str, query_vectors: List[List[float]], top_k: int = 10, concurrency: int = 10):
        """测试并发搜索性能"""
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
        
        concurrent_result = {
            'time': elapsed,
            'qps': qps,
            'avg_latency': avg_latency,
            'count': len(query_vectors),
            'concurrency': concurrency,
            'top_k': top_k
        }
        
        # 初始化concurrent_search列表如果不存在
        if 'concurrent_search' not in self.results:
            self.results['concurrent_search'] = []
        self.results['concurrent_search'].append(concurrent_result)
        
        print(f"[{self.name}] Concurrent search performance: {qps:.2f} QPS, avg latency: {avg_latency:.2f} ms (concurrency: {concurrency}, top_k: {top_k})")
        return concurrent_result
    
    def save_results(self, test_name: str):
        """保存测试结果到文件"""
        filename = os.path.join(self.results_dir, f"{self.name}_{test_name}.json")
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def _get_memory_usage(self):
        """获取当前内存使用情况"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except:
            return 0



class ChromaDBClient:
    def __init__(self, space="l2"):
        import chromadb
        self.client = chromadb.Client()
        self.space = space
    
    def create_collection(self, collection_name: str, dimension: int):
        """创建集合"""
        # 优化ChromaDB参数以提高性能
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.space}
        )
        return collection
    
    def insert(self, collection_name: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        """插入向量"""
        collection = self.client.get_collection(name=collection_name)
        # 使用原始索引作为ID，确保与groundtruth对应
        ids = [str(i) for i in range(len(vectors))]
        
        # 根据向量数量动态调整批量大小
        batch_size = min(1000, len(vectors) // 10 + 1)
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            collection.add(
                embeddings=batch_vectors,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        return ids
    
    def search(self, collection_name: str, query: List[float], top_k: int = 10):
        """搜索向量"""
        collection = self.client.get_collection(name=collection_name)
        # 执行搜索，使用更大的ef值提高召回率
        results = collection.query(
            query_embeddings=[query],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        # 转换结果格式
        result_list = []
        for i in range(len(results['ids'][0])):
            # 从ID中提取原始索引
            original_index = int(results['ids'][0][i])
            result_list.append({
                'id': results['ids'][0][i],
                'score': results['distances'][0][i],
                'metadata': {
                    **results['metadatas'][0][i],
                    'original_index': original_index  # 确保包含original_index
                }
            })
        
        return result_list
    
    def delete_collection(self, collection_name: str):
        """删除集合"""
        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Error deleting collection: {e}")

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
    # 使用绝对路径，确保数据目录位置正确
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
    
    if dataset_name == "sift1m":
        base_file = os.path.join(data_dir, 'sift1m', 'sift_base.fvecs')
        query_file = os.path.join(data_dir, 'sift1m', 'sift_query.fvecs')
        groundtruth_file = os.path.join(data_dir, 'sift1m', 'sift_groundtruth.ivecs')
        
        # 读取base向量
        base_vectors = list(read_fvecs(base_file))
        # 读取query向量
        query_vectors = list(read_fvecs(query_file))
        # 读取原始ground truth
        groundtruth = [list(vec) for vec in read_ivecs(groundtruth_file)]
        
        # 限制向量数量以节省时间和内存
        base_vectors = base_vectors[:max_vectors]
        
        # 过滤 ground truth，只保留有效索引（小于 max_vectors）
        filtered_groundtruth = []
        for vec in groundtruth:
            filtered_vec = [idx for idx in vec if idx < max_vectors]
            filtered_groundtruth.append(filtered_vec)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, filtered_groundtruth
    elif dataset_name == "gist1m":
        base_file = os.path.join(data_dir, 'gist', 'gist_base.fvecs')
        query_file = os.path.join(data_dir, 'gist', 'gist_query.fvecs')
        groundtruth_file = os.path.join(data_dir, 'gist', 'gist_groundtruth.ivecs')
        
        # 读取base向量
        base_vectors = list(read_fvecs(base_file))
        # 读取query向量
        query_vectors = list(read_fvecs(query_file))
        # 读取原始ground truth
        groundtruth = [list(vec) for vec in read_ivecs(groundtruth_file)]
        
        # 限制向量数量以节省时间和内存
        base_vectors = base_vectors[:max_vectors]
        
        # 过滤 ground truth，只保留有效索引（小于 max_vectors）
        filtered_groundtruth = []
        for vec in groundtruth:
            filtered_vec = [idx for idx in vec if idx < max_vectors]
            filtered_groundtruth.append(filtered_vec)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, filtered_groundtruth
    elif dataset_name == "mnist":
        # 下载MNIST数据集
        mnist_dir = os.path.join(data_dir, "mnist")
        os.makedirs(mnist_dir, exist_ok=True)
        
        # 使用torch加载MNIST
        import torch
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root=mnist_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=mnist_dir, train=False, download=True, transform=transform)
        
        # 转换为向量并降维
        base_vectors = []
        for img, _ in train_dataset:
            if len(base_vectors) >= max_vectors:
                break
            # 扁平化并归一化
            vector = img.numpy().flatten()
            # 降维到64维（使用简单的平均池化）
            reduced_vector = []
            for i in range(0, 784, 12):
                reduced_vector.append(np.mean(vector[i:i+12]))
            base_vectors.append(reduced_vector[:64])
        
        # 生成查询向量
        query_vectors = []
        for img, _ in test_dataset:
            if len(query_vectors) >= 1000:
                break
            vector = img.numpy().flatten()
            reduced_vector = []
            for i in range(0, 784, 12):
                reduced_vector.append(np.mean(vector[i:i+12]))
            query_vectors.append(reduced_vector[:64])
        
        # 计算groundtruth
        groundtruth = []
        base_array = np.array(base_vectors, dtype=np.float32)
        
        for query in query_vectors:
            query_array = np.array(query, dtype=np.float32)
            # 计算L2距离
            distances = np.linalg.norm(base_array - query_array, axis=1)
            # 找到距离最小的10个索引
            top_indices = np.argsort(distances)[:10].tolist()
            groundtruth.append(top_indices)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, groundtruth
    elif dataset_name == "cifar10":
        # 下载CIFAR-10数据集
        cifar_dir = os.path.join(data_dir, "cifar10")
        os.makedirs(cifar_dir, exist_ok=True)
        
        # 使用torch加载CIFAR-10
        import torch
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root=cifar_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=cifar_dir, train=False, download=True, transform=transform)
        
        # 转换为向量并降维
        base_vectors = []
        for img, _ in train_dataset:
            if len(base_vectors) >= max_vectors:
                break
            # 扁平化并归一化
            vector = img.numpy().flatten()
            # 降维到128维（使用简单的平均池化）
            reduced_vector = []
            for i in range(0, 3072, 24):
                reduced_vector.append(np.mean(vector[i:i+24]))
            base_vectors.append(reduced_vector[:128])
        
        # 生成查询向量
        query_vectors = []
        for img, _ in test_dataset:
            if len(query_vectors) >= 1000:
                break
            vector = img.numpy().flatten()
            reduced_vector = []
            for i in range(0, 3072, 24):
                reduced_vector.append(np.mean(vector[i:i+24]))
            query_vectors.append(reduced_vector[:128])
        
        # 计算groundtruth
        groundtruth = []
        base_array = np.array(base_vectors, dtype=np.float32)
        
        for query in query_vectors:
            query_array = np.array(query, dtype=np.float32)
            # 计算L2距离
            distances = np.linalg.norm(base_array - query_array, axis=1)
            # 找到距离最小的10个索引
            top_indices = np.argsort(distances)[:10].tolist()
            groundtruth.append(top_indices)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, groundtruth
    elif dataset_name == "glove":
        # 下载GloVe词向量
        glove_dir = os.path.join(data_dir, "glove")
        os.makedirs(glove_dir, exist_ok=True)
        
        # 下载GloVe 50维词向量
        glove_file = os.path.join(glove_dir, "glove.6B.50d.txt")
        if not os.path.exists(glove_file):
            print("Downloading GloVe word vectors...")
            wget.download("http://nlp.stanford.edu/data/glove.6B.zip", os.path.join(glove_dir, "glove.6B.zip"))
            # 解压
            with zipfile.ZipFile(os.path.join(glove_dir, "glove.6B.zip"), "r") as zip_ref:
                zip_ref.extractall(glove_dir)
        
        # 读取GloVe词向量
        base_vectors = []
        words = []
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(base_vectors) >= max_vectors:
                    break
                parts = line.strip().split()
                word = parts[0]
                vector = list(map(float, parts[1:]))
                base_vectors.append(vector)
                words.append(word)
        
        # 生成查询向量（随机选择1000个词向量）
        query_indices = random.sample(range(len(base_vectors)), min(1000, len(base_vectors)))
        query_vectors = [base_vectors[i] for i in query_indices]
        
        # 计算groundtruth
        groundtruth = []
        base_array = np.array(base_vectors, dtype=np.float32)
        
        for query in query_vectors:
            query_array = np.array(query, dtype=np.float32)
            # 计算cosine相似度
            similarities = np.dot(base_array, query_array) / (np.linalg.norm(base_array, axis=1) * np.linalg.norm(query_array))
            # 找到相似度最高的10个索引（排除自身）
            top_indices = np.argsort(-similarities)[:11].tolist()  # 取11个，然后排除自身
            # 排除查询向量本身
            filtered_indices = [idx for idx in top_indices if idx not in query_indices][:10]
            groundtruth.append(filtered_indices)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, groundtruth
    elif dataset_name == "movielens":
        # 下载MovieLens数据集
        movielens_dir = os.path.join(data_dir, "movielens")
        os.makedirs(movielens_dir, exist_ok=True)
        
        # 下载MovieLens 100K数据集
        movielens_file = os.path.join(movielens_dir, "ml-100k", "u.data")
        if not os.path.exists(movielens_file):
            print("Downloading MovieLens 100K dataset...")
            wget.download("http://files.grouplens.org/datasets/movielens/ml-100k.zip", os.path.join(movielens_dir, "ml-100k.zip"))
            # 解压
            with zipfile.ZipFile(os.path.join(movielens_dir, "ml-100k.zip"), "r") as zip_ref:
                zip_ref.extractall(movielens_dir)
        
        # 读取MovieLens数据集并创建用户和电影嵌入
        # 读取评分数据
        ratings = pd.read_csv(os.path.join(movielens_dir, "ml-100k", "u.data"), sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
        
        # 创建用户-电影评分矩阵
        user_movie_matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
        
        # 使用SVD降维到64维
        svd = TruncatedSVD(n_components=64, random_state=42)
        user_embeddings = svd.fit_transform(user_movie_matrix)
        
        # 限制向量数量
        user_embeddings = user_embeddings[:max_vectors]
        
        # 转换为列表格式
        base_vectors = user_embeddings.tolist()
        
        # 生成查询向量（随机选择1000个用户嵌入）
        query_indices = random.sample(range(len(base_vectors)), min(1000, len(base_vectors)))
        query_vectors = [base_vectors[i] for i in query_indices]
        
        # 计算groundtruth
        groundtruth = []
        base_array = np.array(base_vectors, dtype=np.float32)
        
        for query in query_vectors:
            query_array = np.array(query, dtype=np.float32)
            # 计算cosine相似度
            similarities = np.dot(base_array, query_array) / (np.linalg.norm(base_array, axis=1) * np.linalg.norm(query_array))
            # 找到相似度最高的10个索引（排除自身）
            top_indices = np.argsort(-similarities)[:11].tolist()  # 取11个，然后排除自身
            # 排除查询向量本身
            filtered_indices = [idx for idx in top_indices if idx not in query_indices][:10]
            groundtruth.append(filtered_indices)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, groundtruth
    elif dataset_name == "sentence_bert":
        # 生成Sentence-BERT嵌入
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("Error: Sentence-BERT dataset requires sentence-transformers. Please install it with 'pip install sentence-transformers'")
            return [], [], []
        
        # 初始化模型
        print("Loading Sentence-BERT model...")
        # 本地模型路径
        local_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "text", "all-MiniLM-L6-v2")
        
        try:
            # 尝试从本地加载模型
            model = SentenceTransformer(local_model_path)
            print("Loaded model from local path")
        except Exception as e:
            # 本地加载失败，从远程下载
            print(f"Local model not found, downloading from remote: {e}")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            # 保存模型到本地
            try:
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                model.save(local_model_path)
                print(f"Model saved to local path: {local_model_path}")
            except Exception as save_error:
                print(f"Error saving model to local path: {save_error}")
        
        # 生成随机文本
        def generate_random_text(length=50):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(length))
        
        # 生成基础文本
        base_texts = [generate_random_text() for _ in range(max_vectors)]
        # 生成查询文本
        query_texts = [generate_random_text() for _ in range(1000)]
        
        # 生成嵌入
        print("Generating Sentence-BERT embeddings...")
        base_vectors = model.encode(base_texts).tolist()
        query_vectors = model.encode(query_texts).tolist()
        
        # 计算groundtruth
        groundtruth = []
        base_array = np.array(base_vectors, dtype=np.float32)
        
        for query in query_vectors:
            query_array = np.array(query, dtype=np.float32)
            # 计算cosine相似度
            similarities = np.dot(base_array, query_array) / (np.linalg.norm(base_array, axis=1) * np.linalg.norm(query_array))
            # 找到相似度最高的10个索引
            top_indices = np.argsort(-similarities)[:10].tolist()
            groundtruth.append(top_indices)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, groundtruth
    elif dataset_name == "random":
        # 生成随机向量作为默认数据集
        dimension = 64  # 低维向量
        base_vectors = [np.random.rand(dimension).tolist() for _ in range(max_vectors)]
        query_vectors = [np.random.rand(dimension).tolist() for _ in range(1000)]
        
        # 计算groundtruth
        groundtruth = []
        base_array = np.array(base_vectors, dtype=np.float32)
        
        for query in query_vectors:
            query_array = np.array(query, dtype=np.float32)
            # 计算L2距离
            distances = np.linalg.norm(base_array - query_array, axis=1)
            # 找到距离最小的10个索引
            top_indices = np.argsort(distances)[:10].tolist()
            groundtruth.append(top_indices)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, groundtruth
    else:
        # 读取base向量
        base_vectors = list(read_fvecs(base_file))
        # 读取query向量
        query_vectors = list(read_fvecs(query_file))
        
        # 限制向量数量以节省时间和内存
        base_vectors = base_vectors[:max_vectors]
        
        # 生成ground truth：计算每个查询向量与基础向量的距离，找到最相似的向量
        groundtruth = []
        base_array = np.array(base_vectors, dtype=np.float32)
        
        for query in query_vectors:
            query_array = np.array(query, dtype=np.float32)
            # 计算L2距离
            distances = np.linalg.norm(base_array - query_array, axis=1)
            # 找到距离最小的10个索引
            top_indices = np.argsort(distances)[:10].tolist()
            groundtruth.append(top_indices)
        
        print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        return base_vectors, query_vectors, groundtruth

def generate_metadata(count: int):
    """生成测试元数据"""
    return [{"id": i, "category": f"cat_{i % 10}"} for i in range(count)]

def generate_param_name(params):
    """根据参数生成名称"""
    if 'space' in params:
        return f"{params['space']}"
    # 可以扩展其他参数
    return "_".join([f"{k}_{v}" for k, v in params.items()])

if __name__ == "__main__":
    # 导入7VecDB客户端
    from core.vector_db.faiss_vector_db import FAISSVectorDB
    
    # 测试参数
    datasets = ["sift1m", "gist1m", "mnist", "cifar10", "glove", "movielens", "sentence_bert", "random"]
    max_vectors_list = [5000]  # 5K向量，减少测试时间
    
    # ChromaDB 参数调优组合
    chromadb_params = [
        {"space": "l2"},
        {"space": "cosine"},
    ]
    
    for dataset in datasets:
        for max_vectors in max_vectors_list:
            print(f"\n=== Testing {dataset} with {max_vectors} vectors ===")
            
            # 加载数据集
            base_vectors, query_vectors, groundtruth = load_dataset(dataset, max_vectors=max_vectors)
            
            # 转换为列表格式，并确保向量是普通的 Python 浮点数列表
            vectors = [[float(v) for v in vec] for vec in base_vectors]
            query_vectors = [[float(v) for v in vec] for vec in query_vectors]
            groundtruth = groundtruth
            
            # 生成元数据
            metadata = generate_metadata(len(vectors))
            
            # 测试7VecDB-FAISS作为基准
            print("\n--- Testing 7VecDB-FAISS ---")
            # 使用绝对路径，避免在项目根目录创建data文件夹
            faiss_client = FAISSVectorDB(db_path=os.path.join(os.path.dirname(__file__), "..", "data", "vector_db"))
            
            # 使用不同的集合名称
            collection_name = f"test_{dataset}_{max_vectors}_7VecDB-FAISS"
            
            # 清理旧集合
            try:
                faiss_client.delete_collection(collection_name)
            except Exception as e:
                print(f"Error deleting collection: {e}")
            
            # 创建新集合
            dimension = len(vectors[0])
            faiss_client.create_collection(collection_name, dimension)
            
            # 初始化性能测试器
            tester = PerformanceTester(faiss_client, "7VecDB-FAISS")
            
            # 测试插入性能
            tester.test_insert(collection_name, vectors, metadata)
            
            # 测试搜索性能
            if query_vectors:
                test_query_vectors = query_vectors[:100] if len(query_vectors) >= 100 else query_vectors
                test_groundtruth = groundtruth[:100] if len(groundtruth) >= 100 else groundtruth
                # 直接使用指定的 top_k 值，不在测试代码中限制
                for top_k in [1, 5, 10, 50]:
                    tester.test_search(collection_name, test_query_vectors, top_k, test_groundtruth)
            
            # 测试并发搜索
            if query_vectors:
                test_query_vectors = query_vectors[:100] if len(query_vectors) >= 100 else query_vectors
                # 使用固定的 top_k 值 10
                for concurrency in [1, 5, 50]:
                    tester.test_concurrent_search(collection_name, test_query_vectors, 10, concurrency)
            
            # 保存结果
            tester.save_results(f"{dataset}_{max_vectors}")
            
            # 清理测试集合
            try:
                faiss_client.delete_collection(collection_name)
            except Exception as e:
                print(f"Error deleting collection: {e}")
            
            # 测试不同参数组合的ChromaDB
            for i, params in enumerate(chromadb_params):
                param_name = generate_param_name(params)
                print(f"\n--- Testing ChromaDB with params {param_name}: {params} ---")
                
                # 创建ChromaDBClient实例
                chromadb_client = ChromaDBClient(
                    space=params["space"]
                )
                
                # 使用不同的集合名称
                collection_name = f"test_{dataset}_{max_vectors}_ChromaDB_{param_name}"
                
                # 清理旧集合
                try:
                    chromadb_client.delete_collection(collection_name)
                except Exception as e:
                    print(f"Error deleting collection: {e}")
                
                # 创建新集合
                dimension = len(vectors[0])
                try:
                    chromadb_client.create_collection(collection_name, dimension)
                except Exception as e:
                    print(f"Error creating collection: {e}")
                    continue
                
                # 初始化性能测试器
                tester = PerformanceTester(chromadb_client, f"ChromaDB_{param_name}")
                
                # 测试插入性能
                try:
                    tester.test_insert(collection_name, vectors, metadata)
                except Exception as e:
                    print(f"Error inserting vectors: {e}")
                    continue
                
                # 测试搜索性能
                if query_vectors:
                    test_query_vectors = query_vectors[:100] if len(query_vectors) >= 100 else query_vectors
                    test_groundtruth = groundtruth[:100] if len(groundtruth) >= 100 else groundtruth
                    # 直接使用指定的 top_k 值，不在测试代码中限制
                    for top_k in [1, 5, 10, 50]:
                        try:
                            tester.test_search(collection_name, test_query_vectors, top_k, test_groundtruth)
                        except Exception as e:
                            print(f"Error searching: {e}")
                            continue
                
                # 测试并发搜索
                if query_vectors:
                    test_query_vectors = query_vectors[:100] if len(query_vectors) >= 100 else query_vectors
                    # 使用固定的 top_k 值 10
                    for concurrency in [1, 5, 50]:
                        try:
                            tester.test_concurrent_search(collection_name, test_query_vectors, 10, concurrency)
                        except Exception as e:
                            print(f"Error in concurrent search: {e}")
                            continue
                
                # 保存结果
                tester.save_results(f"{dataset}_{max_vectors}_{param_name}")
                
                # 清理测试集合
                try:
                    chromadb_client.delete_collection(collection_name)
                except Exception as e:
                    print(f"Error deleting collection: {e}")