import unittest
import os
import sys
import tempfile
import time
import pytest
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 模拟配置，避免模型下载
os.environ['TEXT_PROCESSING_MODEL_TYPE'] = 'local'
os.environ['TEXT_PROCESSING_MODEL_NAME'] = 'test-model'
os.environ['VECTOR_DB_TYPE'] = 'faiss'
os.environ['VECTOR_DB_PATH'] = './data/vector_db'
os.environ['METADATA_STORAGE_TYPE'] = 'memory'
os.environ['FILE_STORAGE_TYPE'] = 'local'
os.environ['LOCAL_STORAGE_PATH'] = './data/files'

from core.data_flow import DataFlowManager

class TestDataFlow(unittest.TestCase):
    def setUp(self):
        # 初始化数据流程管理器，使用测试模式
        self.data_flow = DataFlowManager(test_mode=True)
        # 创建测试文件
        self.test_file_path = './test_data_flow.txt'
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            f.write('This is a test file for data flow.\nIt contains test content for vector embedding.')
    
    def tearDown(self):
        # 清理测试文件
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
    
    @pytest.mark.timeout(10)
    def test_process_file(self):
        """测试文件处理流程，预计执行时间：10秒"""
        collection_id = 'test_collection'
        result = self.data_flow.process_file(self.test_file_path, collection_id)
        
        self.assertIsInstance(result, dict)
        self.assertIn('file_id', result)
        self.assertIn('collection_id', result)
        self.assertIn('chunks_processed', result)
        self.assertIn('vectors_stored', result)
        
        # 验证处理结果
        self.assertEqual(result['collection_id'], collection_id)
        self.assertGreater(result['chunks_processed'], 0)
        self.assertGreater(result['vectors_stored'], 0)
    
    @pytest.mark.timeout(5)
    def test_search(self):
        """测试向量检索流程，预计执行时间：5秒"""
        # 先处理文件
        collection_id = 'test_collection'
        self.data_flow.process_file(self.test_file_path, collection_id)
        
        # 测试向量检索流程
        query = 'test file'
        results = self.data_flow.search(collection_id, query, top_k=5)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # 验证检索结果
        for result in results:
            self.assertIn('metadata', result)
            self.assertIn('score', result)
            self.assertIn('full_metadata', result)
    
    @pytest.mark.timeout(5)
    def test_filter_by_metadata(self):
        """测试元数据过滤功能，预计执行时间：5秒"""
        # 先处理文件
        collection_id = 'test_collection'
        self.data_flow.process_file(self.test_file_path, collection_id)
        
        # 测试元数据过滤功能
        filters = {'file_type': 'text'}
        results = self.data_flow.filter_by_metadata(collection_id, filters, top_k=5)
        
        self.assertIsInstance(results, list)
        
        # 验证过滤结果
        for result in results:
            self.assertIn('metadata', result)
            self.assertEqual(result['metadata']['file_type'], 'text')

if __name__ == '__main__':
    unittest.main()