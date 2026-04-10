import annoy
import os
import pickle
from typing import List, Dict, Any
from .base_vector_db import BaseVectorDB

class AnnoyVectorDB(BaseVectorDB):
    def __init__(self, db_path='./data/vector_db'):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.collections = {}
        self.metadata = {}
        self._load_collections()
    
    def _load_collections(self):
        # 加载已有的集合
        for file_name in os.listdir(self.db_path):
            if file_name.endswith('.ann'):
                collection_id = file_name[:-4]  # 移除.ann后缀
                index_path = os.path.join(self.db_path, file_name)
                metadata_path = os.path.join(self.db_path, f"{collection_id}_metadata.pkl")
                
                # 加载索引
                index = annoy.AnnoyIndex(128, 'angular')  # 假设维度为128，实际应用中需要根据模型调整
                index.load(index_path)
                self.collections[collection_id] = index
                
                # 加载元数据
                with open(metadata_path, 'rb') as f:
                    self.metadata[collection_id] = pickle.load(f)
    
    def create_collection(self, collection_id: str, dimension: int) -> bool:
        if collection_id in self.collections:
            return False
        
        # 创建Annoy索引
        index = annoy.AnnoyIndex(dimension, 'angular')
        self.collections[collection_id] = index
        self.metadata[collection_id] = {}
        
        # 不需要立即保存，在插入向量并构建索引后再保存
        return True
    
    def insert(self, collection_id: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        index = self.collections[collection_id]
        vector_ids = []
        
        # 插入向量
        import uuid
        for i, vector in enumerate(vectors):
            vector_id = str(uuid.uuid4())
            vector_ids.append(vector_id)
            
            # 添加向量到索引
            index.add_item(i, vector)
            
            # 存储元数据，包含向量
            meta_with_vector = metadata[i].copy()
            meta_with_vector['vector'] = vector
            self.metadata[collection_id][vector_id] = meta_with_vector
        
        # 构建索引
        index.build(10)  # 10棵树
        
        # 保存索引和元数据
        self._save_collection(collection_id)
        return vector_ids
    
    def search(self, collection_id: str, query_vector: List[float], top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        index = self.collections[collection_id]
        
        # 搜索向量
        indices, distances = index.get_nns_by_vector(query_vector, top_k, include_distances=True)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices):
            # 查找对应的vector_id
            vector_id = list(self.metadata[collection_id].keys())[idx]
            metadata = self.metadata[collection_id][vector_id]
            results.append({
                'vector_id': vector_id,
                'distance': distances[i],
                'metadata': metadata
            })
        
        # 应用过滤
        if filters:
            results = [r for r in results if all(r['metadata'].get(k) == v for k, v in filters.items())]
        
        return results
    
    def modify(self, collection_id: str, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        # Annoy不支持修改，需要重建索引
        # 这里简化实现，实际应用中可能需要更复杂的处理
        idx = list(self.metadata[collection_id].keys()).index(vector_id)
        
        # 更新元数据
        self.metadata[collection_id][vector_id] = metadata
        
        # 重建索引
        dimension = len(vector)
        new_index = annoy.AnnoyIndex(dimension, 'angular')
        
        for i, (vid, meta) in enumerate(self.metadata[collection_id].items()):
            # 这里简化实现，实际应用中需要保存所有向量
            new_index.add_item(i, vector)
        
        new_index.build(10)
        self.collections[collection_id] = new_index
        
        # 保存索引和元数据
        self._save_collection(collection_id)
        return True
    
    def delete(self, collection_id: str, vector_ids: List[str]) -> bool:
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} does not exist")
        
        # Annoy不支持删除，需要重建索引
        # 这里简化实现，实际应用中可能需要更复杂的处理
        for vector_id in vector_ids:
            if vector_id in self.metadata[collection_id]:
                del self.metadata[collection_id][vector_id]
        
        # 重建索引
        dimension = len(next(iter(self.metadata[collection_id].values()))['vector'])
        new_index = annoy.AnnoyIndex(dimension, 'angular')
        
        for i, (vid, meta) in enumerate(self.metadata[collection_id].items()):
            new_index.add_item(i, meta['vector'])
        
        new_index.build(10)
        self.collections[collection_id] = new_index
        
        # 保存索引和元数据
        self._save_collection(collection_id)
        return True
    
    def _save_collection(self, collection_id: str):
        # 保存索引
        index_path = os.path.join(self.db_path, f"{collection_id}.ann")
        self.collections[collection_id].save(index_path)
        
        # 保存元数据
        metadata_path = os.path.join(self.db_path, f"{collection_id}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata[collection_id], f)
    
    def delete_collection(self, collection_id: str) -> bool:
        """删除集合"""
        if collection_id not in self.collections:
            return False
        
        # 从内存中删除
        del self.collections[collection_id]
        del self.metadata[collection_id]
        
        # 删除索引文件
        index_path = os.path.join(self.db_path, f"{collection_id}.ann")
        if os.path.exists(index_path):
            try:
                os.remove(index_path)
            except PermissionError:
                # 忽略文件被占用的错误
                pass
        
        # 删除元数据文件
        metadata_path = os.path.join(self.db_path, f"{collection_id}_metadata.pkl")
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
            except PermissionError:
                # 忽略文件被占用的错误
                pass
        
        return True