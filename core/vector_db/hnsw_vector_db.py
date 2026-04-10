import hnswlib
import os
import pickle
from typing import List, Dict, Any
from .base_vector_db import BaseVectorDB

class HNSWVectorDB(BaseVectorDB):
    def __init__(self, db_path='./data/vector_db'):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.collections = {}
        self.metadata = {}
        self._load_collections()
    
    def _load_collections(self):
        # 加载已有的集合
        for file_name in os.listdir(self.db_path):
            if file_name.endswith('.hnsw'):
                collection_id = file_name[:-5]  # 移除.hnsw后缀
                index_path = os.path.join(self.db_path, file_name)
                metadata_path = os.path.join(self.db_path, f"{collection_id}_metadata.pkl")
                
                # 加载索引
                # 首先创建一个临时索引来获取维度
                index = hnswlib.Index(space='l2', dim=128)  # 临时维度，加载后会自动调整
                index.load_index(index_path)
                self.collections[collection_id] = index
                
                # 加载元数据
                with open(metadata_path, 'rb') as f:
                    self.metadata[collection_id] = pickle.load(f)
    
    def create_collection(self, collection_id: str, dimension: int) -> bool:
        if collection_id in self.collections:
            return False
        
        # 创建HNSW索引
        index = hnswlib.Index(space='l2', dim=dimension)
        index.init_index(max_elements=100000, ef_construction=200, M=16)
        self.collections[collection_id] = index
        self.metadata[collection_id] = {}
        
        # 保存索引和元数据
        self._save_collection(collection_id)
        return True
    
    def insert(self, collection_id: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        index = self.collections[collection_id]
        vector_ids = []
        ids = []
        
        # 插入向量
        import uuid
        for i, vector in enumerate(vectors):
            vector_id = str(uuid.uuid4())
            vector_ids.append(vector_id)
            ids.append(i)
            
            # 存储元数据
            self.metadata[collection_id][vector_id] = metadata[i]
        
        # 添加向量到索引
        index.add_items(vectors, ids)
        
        # 保存索引和元数据
        self._save_collection(collection_id)
        return vector_ids
    
    def search(self, collection_id: str, query_vector: List[float], top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        index = self.collections[collection_id]
        
        # 搜索向量
        indices, distances = index.knn_query([query_vector], k=top_k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            # 查找对应的vector_id
            vector_id = list(self.metadata[collection_id].keys())[idx]
            metadata = self.metadata[collection_id][vector_id]
            results.append({
                'vector_id': vector_id,
                'distance': distances[0][i],
                'metadata': metadata
            })
        
        # 应用过滤
        if filters:
            results = [r for r in results if all(r['metadata'].get(k) == v for k, v in filters.items())]
        
        return results
    
    def modify(self, collection_id: str, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        # 查找向量索引
        idx = list(self.metadata[collection_id].keys()).index(vector_id)
        
        # 删除旧向量
        index = self.collections[collection_id]
        index.mark_deleted(idx)
        
        # 插入新向量
        new_idx = len(self.metadata[collection_id])
        index.add_items([vector], [new_idx])
        
        # 更新元数据
        self.metadata[collection_id][vector_id] = metadata
        
        # 保存索引和元数据
        self._save_collection(collection_id)
        return True
    
    def delete(self, collection_id: str, vector_ids: List[str]) -> bool:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        index = self.collections[collection_id]
        indices = []
        
        # 查找向量索引
        for vector_id in vector_ids:
            if vector_id in self.metadata[collection_id]:
                idx = list(self.metadata[collection_id].keys()).index(vector_id)
                indices.append(idx)
                del self.metadata[collection_id][vector_id]
        
        # 删除向量
        if indices:
            for idx in indices:
                try:
                    index.mark_deleted(idx)
                except RuntimeError:
                    # 忽略已经删除的向量
                    pass
            # 保存索引和元数据
            self._save_collection(collection_id)
            return True
        return False
    
    def _save_collection(self, collection_id: str):
        # 保存索引
        index_path = os.path.join(self.db_path, f"{collection_id}.hnsw")
        self.collections[collection_id].save_index(index_path)
        
        # 保存元数据
        metadata_path = os.path.join(self.db_path, f"{collection_id}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata[collection_id], f)
    
    def delete_collection(self, collection_id: str) -> bool:
        """删除集合"""
        if collection_id not in self.collections:
            return False
        
        # 删除索引文件
        index_path = os.path.join(self.db_path, f"{collection_id}.hnsw")
        if os.path.exists(index_path):
            os.remove(index_path)
        
        # 删除元数据文件
        metadata_path = os.path.join(self.db_path, f"{collection_id}_metadata.pkl")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # 从内存中删除
        del self.collections[collection_id]
        del self.metadata[collection_id]
        
        return True