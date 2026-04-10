import pytest
import numpy as np
import os
import shutil
from vector_db.core.vector_db.base_vector_db import BaseVectorDB
from vector_db.core.vector_db.faiss_vector_db import FAISSVectorDB
from vector_db.core.vector_db.hnsw_vector_db import HNSWVectorDB
from vector_db.core.vector_db.annoy_vector_db import AnnoyVectorDB

class TestVectorDB:
    def test_base_vector_db(self):
        """测试基础向量数据库接口"""
        # 测试接口定义
        assert hasattr(BaseVectorDB, 'create_collection')
        assert hasattr(BaseVectorDB, 'insert')
        assert hasattr(BaseVectorDB, 'search')
        assert hasattr(BaseVectorDB, 'modify')
        assert hasattr(BaseVectorDB, 'delete')
    
    def test_faiss_vector_db(self):
        """测试FAISS向量数据库"""
        # 清理测试目录
        if os.path.exists('./test_faiss_db'):
            shutil.rmtree('./test_faiss_db')
        # 创建FAISS向量数据库实例
        db = FAISSVectorDB('./test_faiss_db')
        
        # 测试创建集合
        collection_name = 'test_collection'
        dimension = 128
        success = db.create_collection(collection_name, dimension)
        assert success
        
        # 测试插入向量
        vectors = [np.random.rand(dimension).tolist() for _ in range(10)]
        metadata = [{'id': i, 'text': f'test {i}'} for i in range(10)]
        vector_ids = db.insert(collection_name, vectors, metadata)
        assert len(vector_ids) == 10
        
        # 测试批量插入向量
        batch_vectors = [np.random.rand(dimension).tolist() for _ in range(10)]
        batch_metadata = [{'id': i+10, 'text': f'test {i+10}'} for i in range(10)]
        batch_vector_ids = db.batch_insert(collection_name, batch_vectors, batch_metadata)
        assert len(batch_vector_ids) == 10
        
        # 测试搜索向量
        query_vector = np.random.rand(dimension).tolist()
        results = db.search(collection_name, query_vector, top_k=5)
        assert len(results) <= 5
        
        # 测试优化索引
        success = db.optimize_index(collection_name)
        assert success
        
        # 测试重建索引
        success = db.rebuild_index(collection_name)
        assert success
        
        # 测试删除向量
        success = db.delete(collection_name, vector_ids[:2])
        assert success
        
        # 测试删除集合
        success = db.delete_collection(collection_name)
        assert success
        
        # 清理测试数据
        if os.path.exists('./test_faiss_db'):
            shutil.rmtree('./test_faiss_db')
    
    def test_hnsw_vector_db(self):
        """测试HNSW向量数据库"""
        # 清理测试目录
        if os.path.exists('./test_hnsw_db'):
            shutil.rmtree('./test_hnsw_db')
        # 创建HNSW向量数据库实例
        db = HNSWVectorDB('./test_hnsw_db')
        
        # 测试创建集合
        collection_name = 'test_collection'
        dimension = 128
        success = db.create_collection(collection_name, dimension)
        assert success
        
        # 测试插入向量
        vectors = [np.random.rand(dimension).tolist() for _ in range(10)]
        metadata = [{'id': i, 'text': f'test {i}'} for i in range(10)]
        vector_ids = db.insert(collection_name, vectors, metadata)
        assert len(vector_ids) == 10
        
        # 测试搜索向量
        query_vector = np.random.rand(dimension).tolist()
        results = db.search(collection_name, query_vector, top_k=5)
        assert len(results) <= 5
        
        # 测试删除向量
        success = db.delete(collection_name, vector_ids[:2])
        assert success
        
        # 测试删除集合
        success = db.delete_collection(collection_name)
        assert success
        
        # 清理测试数据
        if os.path.exists('./test_hnsw_db'):
            shutil.rmtree('./test_hnsw_db')
    
    def test_annoy_vector_db(self):
        """测试Annoy向量数据库"""
        # 清理测试目录
        if os.path.exists('./test_annoy_db'):
            shutil.rmtree('./test_annoy_db')
        # 创建Annoy向量数据库实例
        db = AnnoyVectorDB('./test_annoy_db')
        
        # 测试创建集合
        collection_name = 'test_collection'
        dimension = 128
        success = db.create_collection(collection_name, dimension)
        assert success
        
        # 测试插入向量
        vectors = [np.random.rand(dimension).tolist() for _ in range(10)]
        metadata = [{'id': i, 'text': f'test {i}'} for i in range(10)]
        vector_ids = db.insert(collection_name, vectors, metadata)
        assert len(vector_ids) == 10
        
        # 测试搜索向量
        query_vector = np.random.rand(dimension).tolist()
        results = db.search(collection_name, query_vector, top_k=5)
        assert len(results) <= 5
        
        # 测试删除向量
        success = db.delete(collection_name, vector_ids[:2])
        assert success
        
        # 测试删除集合
        success = db.delete_collection(collection_name)
        assert success
        
        # 清理测试数据
        if os.path.exists('./test_annoy_db'):
            shutil.rmtree('./test_annoy_db')