import unittest
import os
import sys
import shutil
import pytest
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.vector_db.annoy_vector_db import ANNOYVectorDB
from core.vector_db.hnsw_vector_db import HNSWVectorDB

class TestVectorDBImplementations(unittest.TestCase):
    def setUp(self):
        # 初始化测试对象，使用临时目录
        self.test_db_path = './test_vector_db_implementations'
        self.annoy_db = ANNOYVectorDB(db_path=self.test_db_path)
        self.hnsw_db = HNSWVectorDB(db_path=self.test_db_path)
        self.collection_id = 'test_collection'
    
    def tearDown(self):
        # 清理测试数据
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
    
    @pytest.mark.timeout(2)
    def test_annoy_create_collection(self):
        """测试ANNOY创建集合，预计执行时间：2秒"""
        self.annoy_db.create_collection(self.collection_id, 384)
        self.assertIn(self.collection_id, self.annoy_db.collections)
    
    @pytest.mark.timeout(5)
    def test_annoy_insert_vectors(self):
        """测试ANNOY插入向量，预计执行时间：5秒"""
        self.annoy_db.create_collection(self.collection_id, 384)
        vectors = [[0.1] * 384, [0.2] * 384]
        metadata = [{'file_id': '1', 'file_type': 'text'}, {'file_id': '2', 'file_type': 'text'}]
        vector_ids = self.annoy_db.insert(self.collection_id, vectors, metadata)
        self.assertIsInstance(vector_ids, list)
        self.assertEqual(len(vector_ids), 2)
    
    @pytest.mark.timeout(5)
    def test_annoy_search_vectors(self):
        """测试ANNOY搜索向量，预计执行时间：5秒"""
        self.annoy_db.create_collection(self.collection_id, 384)
        # 插入测试向量
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        metadata = [{'file_id': '1', 'file_type': 'text'}, {'file_id': '2', 'file_type': 'text'}, {'file_id': '3', 'file_type': 'text'}]
        self.annoy_db.insert(self.collection_id, vectors, metadata)
        # 搜索
        query_vector = [0.1] * 384
        results = self.annoy_db.search(self.collection_id, query_vector, top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
    
    @pytest.mark.timeout(2)
    def test_annoy_delete_collection(self):
        """测试ANNOY删除集合，预计执行时间：2秒"""
        self.annoy_db.create_collection(self.collection_id, 384)
        self.annoy_db.delete_collection(self.collection_id)
        self.assertNotIn(self.collection_id, self.annoy_db.collections)
    
    @pytest.mark.timeout(2)
    def test_hnsw_create_collection(self):
        """测试HNSW创建集合，预计执行时间：2秒"""
        self.hnsw_db.create_collection(self.collection_id, 384)
        self.assertIn(self.collection_id, self.hnsw_db.collections)
    
    @pytest.mark.timeout(5)
    def test_hnsw_insert_vectors(self):
        """测试HNSW插入向量，预计执行时间：5秒"""
        self.hnsw_db.create_collection(self.collection_id, 384)
        vectors = [[0.1] * 384, [0.2] * 384]
        metadata = [{'file_id': '1', 'file_type': 'text'}, {'file_id': '2', 'file_type': 'text'}]
        vector_ids = self.hnsw_db.insert(self.collection_id, vectors, metadata)
        self.assertIsInstance(vector_ids, list)
        self.assertEqual(len(vector_ids), 2)
    
    @pytest.mark.timeout(5)
    def test_hnsw_search_vectors(self):
        """测试HNSW搜索向量，预计执行时间：5秒"""
        self.hnsw_db.create_collection(self.collection_id, 384)
        # 插入测试向量
        vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        metadata = [{'file_id': '1', 'file_type': 'text'}, {'file_id': '2', 'file_type': 'text'}, {'file_id': '3', 'file_type': 'text'}]
        self.hnsw_db.insert(self.collection_id, vectors, metadata)
        # 搜索
        query_vector = [0.1] * 384
        results = self.hnsw_db.search(self.collection_id, query_vector, top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
    
    @pytest.mark.timeout(2)
    def test_hnsw_delete_collection(self):
        """测试HNSW删除集合，预计执行时间：2秒"""
        self.hnsw_db.create_collection(self.collection_id, 384)
        self.hnsw_db.delete_collection(self.collection_id)
        self.assertNotIn(self.collection_id, self.hnsw_db.collections)

if __name__ == '__main__':
    unittest.main()