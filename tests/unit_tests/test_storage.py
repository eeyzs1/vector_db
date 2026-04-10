import unittest
import os
import sys
import tempfile
import time
import pytest
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.storage.local_file_storage import LocalFileSystemStorage
from core.storage.metadata_storage import MetadataStorageInterface

class TestStorage(unittest.TestCase):
    def setUp(self):
        # 创建临时存储目录
        self.temp_dir = tempfile.mkdtemp()
        # 初始化本地文件存储
        self.file_storage = LocalFileSystemStorage(storage_path=self.temp_dir)
        
        # 创建一个简单的内存元数据存储实现
        class InMemoryMetadataStorage(MetadataStorageInterface):
            def __init__(self):
                self.metadata = {}
            
            def store_metadata(self, metadata):
                metadata_id = metadata.get('file_id', 'test-id')
                self.metadata[metadata_id] = metadata
                return metadata_id
            
            def get_metadata(self, metadata_id):
                return self.metadata.get(metadata_id, {})
            
            def update_metadata(self, metadata_id, metadata):
                if metadata_id in self.metadata:
                    self.metadata[metadata_id].update(metadata)
                    return True
                return False
            
            def delete_metadata(self, metadata_id):
                if metadata_id in self.metadata:
                    del self.metadata[metadata_id]
                    return True
                return False
            
            def search_metadata(self, filters):
                results = []
                for metadata in self.metadata.values():
                    match = True
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        results.append(metadata)
                return results
        
        # 初始化元数据存储
        self.metadata_storage = InMemoryMetadataStorage()
    
    def tearDown(self):
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.timeout(0.5)
    def test_file_storage(self):
        """测试文件存储功能，预计执行时间：0.5秒"""
        test_content = b'Test file content'
        file_path = 'test_file.txt'
        
        # 存储文件
        stored_file_id = self.file_storage.store_file(file_path, test_content)
        self.assertIsInstance(stored_file_id, str)
        
        # 读取文件
        retrieved_content = self.file_storage.read_file(stored_file_id)
        self.assertEqual(retrieved_content, test_content)
        
        # 删除文件
        delete_result = self.file_storage.delete_file(stored_file_id)
        self.assertTrue(delete_result)
    
    @pytest.mark.timeout(0.5)
    def test_metadata_storage(self):
        """测试元数据存储功能，预计执行时间：0.5秒"""
        test_metadata = {
            'file_id': 'test-file-id',
            'file_type': 'text',
            'collection_id': 'test-collection',
            'timestamp': '2024-01-01T00:00:00',
            'hash': 'test-hash'
        }
        
        # 存储元数据
        metadata_id = self.metadata_storage.store_metadata(test_metadata)
        self.assertIsInstance(metadata_id, str)
        
        # 获取元数据
        retrieved_metadata = self.metadata_storage.get_metadata(metadata_id)
        self.assertEqual(retrieved_metadata['file_id'], test_metadata['file_id'])
        
        # 更新元数据
        update_data = {'file_type': 'updated-text'}
        update_result = self.metadata_storage.update_metadata(metadata_id, update_data)
        self.assertTrue(update_result)
        
        # 验证更新
        updated_metadata = self.metadata_storage.get_metadata(metadata_id)
        self.assertEqual(updated_metadata['file_type'], 'updated-text')
        
        # 搜索元数据
        search_results = self.metadata_storage.search_metadata({'file_type': 'updated-text'})
        self.assertGreater(len(search_results), 0)
        
        # 删除元数据
        delete_result = self.metadata_storage.delete_metadata(metadata_id)
        self.assertTrue(delete_result)
        
        # 验证删除
        deleted_metadata = self.metadata_storage.get_metadata(metadata_id)
        self.assertEqual(deleted_metadata, {})

if __name__ == '__main__':
    unittest.main()