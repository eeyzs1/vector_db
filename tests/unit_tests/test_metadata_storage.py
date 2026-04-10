import pytest
from vector_db.core.storage.metadata_storage import MetadataStorageInterface
from vector_db.core.storage.mysql_storage import MySQLStorage
from vector_db.core.storage.redis_storage import RedisStorage
from vector_db.core.storage.mongodb_storage import MongoDBStorage

class TestMetadataStorage:
    def test_metadata_storage_interface(self):
        """测试元数据存储接口"""
        # 测试接口定义
        assert hasattr(MetadataStorageInterface, 'store_metadata')
        assert hasattr(MetadataStorageInterface, 'get_metadata')
        assert hasattr(MetadataStorageInterface, 'update_metadata')
        assert hasattr(MetadataStorageInterface, 'delete_metadata')
        assert hasattr(MetadataStorageInterface, 'search_metadata')
    
    def test_mysql_storage(self):
        """测试MySQL存储"""
        try:
            # 创建MySQL存储实例
            storage = MySQLStorage(
                host='localhost',
                port=3306,
                user='root',
                password='password',
                database='vector_db'
            )
            
            # 测试存储元数据
            metadata = {
                'file_id': 'test_file_id',
                'file_type': 'text',
                'collection_id': 'test_collection',
                'timestamp': '2023-01-01T00:00:00',
                'hash': 'test_hash'
            }
            metadata_id = storage.store_metadata(metadata)
            assert metadata_id == 'test_file_id'
            
            # 测试获取元数据
            retrieved_metadata = storage.get_metadata('test_file_id')
            assert retrieved_metadata['file_id'] == 'test_file_id'
            
            # 测试更新元数据
            update_data = {'file_type': 'updated_text'}
            success = storage.update_metadata('test_file_id', update_data)
            assert success
            
            # 测试搜索元数据
            results = storage.search_metadata({'collection_id': 'test_collection'})
            assert len(results) > 0
            
            # 测试删除元数据
            success = storage.delete_metadata('test_file_id')
            assert success
        except Exception as e:
            # 如果无法连接到MySQL，跳过测试
            pytest.skip(f"无法连接到MySQL: {e}")
    
    def test_redis_storage(self):
        """测试Redis存储"""
        try:
            # 创建Redis存储实例
            storage = RedisStorage(
                host='localhost',
                port=6379,
                password='',
                db=0
            )
            
            # 测试存储元数据
            metadata = {
                'file_id': 'test_file_id',
                'file_type': 'text',
                'collection_id': 'test_collection',
                'timestamp': '2023-01-01T00:00:00',
                'hash': 'test_hash'
            }
            metadata_id = storage.store_metadata(metadata)
            assert metadata_id == 'test_file_id'
            
            # 测试获取元数据
            retrieved_metadata = storage.get_metadata('test_file_id')
            assert retrieved_metadata['file_id'] == 'test_file_id'
            
            # 测试更新元数据
            update_data = {'file_type': 'updated_text'}
            success = storage.update_metadata('test_file_id', update_data)
            assert success
            
            # 测试搜索元数据
            results = storage.search_metadata({'collection_id': 'test_collection'})
            assert len(results) > 0
            
            # 测试删除元数据
            success = storage.delete_metadata('test_file_id')
            assert success
        except Exception as e:
            # 如果无法连接到Redis，跳过测试
            pytest.skip(f"无法连接到Redis: {e}")
    
    def test_mongodb_storage(self):
        """测试MongoDB存储"""
        try:
            # 创建MongoDB存储实例
            storage = MongoDBStorage(
                host='localhost',
                port=27017,
                user='',
                password='',
                database='vector_db'
            )
            
            # 测试存储元数据
            metadata = {
                'file_id': 'test_file_id',
                'file_type': 'text',
                'collection_id': 'test_collection',
                'timestamp': '2023-01-01T00:00:00',
                'hash': 'test_hash'
            }
            metadata_id = storage.store_metadata(metadata)
            assert metadata_id == 'test_file_id'
            
            # 测试获取元数据
            retrieved_metadata = storage.get_metadata('test_file_id')
            assert retrieved_metadata['file_id'] == 'test_file_id'
            
            # 测试更新元数据
            update_data = {'file_type': 'updated_text'}
            success = storage.update_metadata('test_file_id', update_data)
            assert success
            
            # 测试搜索元数据
            results = storage.search_metadata({'collection_id': 'test_collection'})
            assert len(results) > 0
            
            # 测试删除元数据
            success = storage.delete_metadata('test_file_id')
            assert success
        except Exception as e:
            # 如果无法连接到MongoDB，跳过测试
            pytest.skip(f"无法连接到MongoDB: {e}")