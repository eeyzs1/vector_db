import os
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from config.config import config
from core.readers.base_reader import BaseDocumentReader
from core.readers.text_reader import TextDocumentReader
from core.readers.image_reader import ImageDocumentReader
from core.readers.video_reader import VideoDocumentReader
from core.readers.audio_reader import AudioDocumentReader
from core.storage.file_storage import FileStorageInterface
from core.storage.local_file_storage import LocalFileSystemStorage
from core.storage.object_storage import ObjectStorage
from core.storage.database_storage import DatabaseStorage
from core.storage.metadata_storage import MetadataStorageInterface
from core.storage.mysql_storage import MySQLStorage
from core.storage.redis_storage import RedisStorage
from core.storage.mongodb_storage import MongoDBStorage
from core.processors.base_processor import BaseProcessor
from core.processors.text_processor import TextProcessor
from core.processors.image_processor import ImageProcessor
from core.processors.video_processor import VideoProcessor
from core.processors.audio_processor import AudioProcessor
from core.vector_db.base_vector_db import BaseVectorDB
from core.vector_db.faiss_vector_db import FAISSVectorDB
from core.vector_db.hnsw_vector_db import HNSWVectorDB
from core.vector_db.annoy_vector_db import AnnoyVectorDB
from core.utils.parallel_processor import parallel_processor
from core.logging.log_manager import logger


class _InMemoryFileStorage(FileStorageInterface):
    def __init__(self):
        self.files = {}

    def store_file(self, file_path: str, content: bytes) -> str:
        file_id = str(uuid.uuid4())
        self.files[file_id] = (content, file_path)
        return file_id

    def get_file(self, file_id: str) -> bytes:
        return self.files.get(file_id, (b'', ''))[0]

    def read_file(self, file_id: str) -> bytes:
        return self.get_file(file_id)

    def delete_file(self, file_id: str) -> bool:
        if file_id in self.files:
            del self.files[file_id]
            return True
        return False

    def list_files(self) -> List[str]:
        return [fp for _, fp in self.files.values()]


class _InMemoryMetadataStorage(MetadataStorageInterface):
    def __init__(self):
        self.metadata = {}

    def store_metadata(self, metadata: Dict[str, Any]) -> str:
        metadata_id = metadata.get('file_id', str(uuid.uuid4()))
        self.metadata[metadata_id] = metadata
        return metadata_id

    def get_metadata(self, metadata_id: str) -> Dict[str, Any]:
        return self.metadata.get(metadata_id, {})

    def update_metadata(self, metadata_id: str, metadata: Dict[str, Any]) -> bool:
        if metadata_id in self.metadata:
            self.metadata[metadata_id].update(metadata)
            return True
        return False

    def delete_metadata(self, metadata_id: str) -> bool:
        if metadata_id in self.metadata:
            del self.metadata[metadata_id]
            return True
        return False

    def search_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for metadata in self.metadata.values():
            if all(metadata.get(k) == v for k, v in filters.items()):
                results.append(metadata)
        return results


class DataFlowManager:
    def __init__(self, test_mode=False):
        self.config = config
        self.test_mode = test_mode
        # 验证配置
        validation_errors = self.config.validate_config()
        if validation_errors:
            logger.warning("配置验证错误:")
            for error in validation_errors:
                logger.warning(f"- {error}")
            logger.warning("使用默认配置继续运行...")
        self.file_storage = self._init_file_storage()
        self.metadata_storage = self._init_metadata_storage()
        self.vector_db = self._init_vector_db()
        self.processors = self._init_processors()
        self.readers = self._init_readers()
    
    def _init_file_storage(self) -> FileStorageInterface:
        if self.test_mode:
            return _InMemoryFileStorage()

        storage_type = self.config.get('FILE_STORAGE_TYPE')
        if storage_type == 'local':
            return LocalFileSystemStorage(self.config.get('LOCAL_STORAGE_PATH'))
        elif storage_type == 's3':
            return ObjectStorage(
                bucket_name=self.config.get('S3_BUCKET_NAME'),
                access_key=self.config.get('S3_ACCESS_KEY'),
                secret_key=self.config.get('S3_SECRET_KEY'),
                region=self.config.get('S3_REGION')
            )
        elif storage_type == 'database':
            return DatabaseStorage()
        else:
            return LocalFileSystemStorage(self.config.get('LOCAL_STORAGE_PATH'))
    
    def _init_metadata_storage(self) -> MetadataStorageInterface:
        if self.test_mode:
            return _InMemoryMetadataStorage()

        storage_type = self.config.get('METADATA_STORAGE_TYPE')
        try:
            if storage_type == 'mysql':
                return MySQLStorage(
                    host=self.config.get('MYSQL_HOST'),
                    port=self.config.get('MYSQL_PORT'),
                    user=self.config.get('MYSQL_USER'),
                    password=self.config.get('MYSQL_PASSWORD'),
                    database=self.config.get('MYSQL_DATABASE')
                )
            elif storage_type == 'redis':
                return RedisStorage(
                    host=self.config.get('REDIS_HOST'),
                    port=self.config.get('REDIS_PORT'),
                    password=self.config.get('REDIS_PASSWORD'),
                    db=self.config.get('REDIS_DB')
                )
            elif storage_type == 'mongodb':
                return MongoDBStorage(
                    host=self.config.get('MONGO_HOST'),
                    port=self.config.get('MONGO_PORT'),
                    user=self.config.get('MONGO_USER'),
                    password=self.config.get('MONGO_PASSWORD'),
                    database=self.config.get('MONGO_DATABASE')
                )
        except Exception as e:
            logger.error(f"无法连接到{storage_type}存储，使用内存存储作为备选: {e}")
        
        # 如果所有存储都失败，使用内存存储
        return _InMemoryMetadataStorage()
    
    def _init_vector_db(self) -> BaseVectorDB:
        # 检查是否处于测试模式
        test_mode = getattr(self, 'test_mode', False)
        
        if test_mode:
            # 在测试模式下使用内存向量数据库
            class InMemoryVectorDB(BaseVectorDB):
                def __init__(self):
                    self.vectors = []
                    self.metadata = []
                    self.collections = set()
                
                def create_collection(self, collection_name: str, dimension: int) -> bool:
                    self.collections.add(collection_name)
                    return True
                
                def insert(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]], collection_name: str) -> List[str]:
                    import uuid
                    vector_ids = []
                    for vector, metadata in zip(vectors, metadatas):
                        vector_id = str(uuid.uuid4())
                        self.vectors.append(vector)
                        self.metadata.append({
                            'id': vector_id,
                            'collection': collection_name,
                            **metadata
                        })
                        vector_ids.append(vector_id)
                    return vector_ids
                
                def search(self, query_vector: List[float], top_k: int = 5, filters: Dict[str, Any] = None, collection_name: str = None) -> List[Dict[str, Any]]:
                    import numpy as np
                    results = []
                    for i, vector in enumerate(self.vectors):
                        # 检查集合名称
                        if collection_name and self.metadata[i].get('collection') != collection_name:
                            continue
                        # 计算余弦相似度
                        similarity = np.dot(vector, query_vector) / (np.linalg.norm(vector) * np.linalg.norm(query_vector))
                        results.append((similarity, self.metadata[i]))
                    
                    # 按相似度排序
                    results.sort(key=lambda x: x[0], reverse=True)
                    
                    # 应用过滤
                    if filters:
                        filtered_results = []
                        for score, metadata in results:
                            match = True
                            for key, value in filters.items():
                                if metadata.get(key) != value:
                                    match = False
                                    break
                            if match:
                                filtered_results.append((score, metadata))
                        results = filtered_results
                    
                    # 返回前top_k个结果
                    return [
                        {
                            'id': metadata['id'],
                            'score': score,
                            'metadata': metadata
                        }
                        for score, metadata in results[:top_k]
                    ]
                
                def delete(self, vector_ids: List[str]) -> bool:
                    for vector_id in vector_ids:
                        for i, metadata in enumerate(self.metadata):
                            if metadata['id'] == vector_id:
                                del self.vectors[i]
                                del self.metadata[i]
                                break
                    return True
                
                def modify(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
                    for i, meta in enumerate(self.metadata):
                        if meta['id'] == vector_id:
                            self.metadata[i].update(metadata)
                            return True
                    return False
                
                def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> str:
                    import uuid
                    vector_id = str(uuid.uuid4())
                    self.vectors.append(vector)
                    self.metadata.append({
                        'id': vector_id,
                        **metadata
                    })
                    return vector_id
                
                def delete_vector(self, vector_id: str) -> bool:
                    for i, metadata in enumerate(self.metadata):
                        if metadata['id'] == vector_id:
                            del self.vectors[i]
                            del self.metadata[i]
                            return True
                    return False
                
                def get_vector(self, vector_id: str) -> Dict[str, Any]:
                    for i, metadata in enumerate(self.metadata):
                        if metadata['id'] == vector_id:
                            return {
                                'id': vector_id,
                                'vector': self.vectors[i],
                                'metadata': metadata
                            }
                    return None
            return InMemoryVectorDB()
        
        db_type = self.config.get('VECTOR_DB_TYPE')
        db_path = self.config.get('VECTOR_DB_PATH')
        if db_type == 'faiss':
            return FAISSVectorDB(db_path)
        elif db_type == 'hnsw':
            return HNSWVectorDB(db_path)
        elif db_type == 'annoy':
            return AnnoyVectorDB(db_path)
        else:
            return FAISSVectorDB(db_path)
    
    def _init_processors(self) -> Dict[str, BaseProcessor]:
        processors = {}
        
        # 检查是否处于测试模式
        test_mode = getattr(self, 'test_mode', False)
        
        # 在测试模式下使用模拟处理器
        if test_mode:
            logger.info("使用模拟处理器")
            # 创建模拟文本处理器
            class MockTextProcessor(BaseProcessor):
                def chunk(self, content: str):
                    return [content]
                def clean(self, content: str):
                    return content.strip()
                def embed(self, content: str):
                    import random
                    return [random.random() for _ in range(384)]
            processors['text'] = MockTextProcessor()
            
            # 创建模拟图像处理器
            class MockImageProcessor(BaseProcessor):
                def chunk(self, content):
                    return [content]
                def clean(self, content):
                    return content
                def embed(self, content):
                    import random
                    return [random.random() for _ in range(512)]
            processors['image'] = MockImageProcessor()
            
            # 创建模拟视频处理器
            class MockVideoProcessor(BaseProcessor):
                def chunk(self, content):
                    return [content]
                def clean(self, content):
                    return content
                def embed(self, content):
                    import random
                    return [random.random() for _ in range(512)]
            processors['video'] = MockVideoProcessor()

            # 创建模拟音频处理器
            processors['audio'] = AudioProcessor(test_mode=True)
        else:
            # 尝试初始化文本处理器
            try:
                text_config = self.config.apply_model_config('text')
                processors['text'] = TextProcessor(**text_config)
            except Exception as e:
                logger.error(f"初始化文本处理器失败，使用模拟处理器: {e}")
                # 创建一个简单的模拟文本处理器
                class MockTextProcessor(BaseProcessor):
                    def chunk(self, content: str):
                        return [content]
                    def clean(self, content: str):
                        return content.strip()
                    def embed(self, content: str):
                        import random
                        return [random.random() for _ in range(384)]
                processors['text'] = MockTextProcessor()
            
            # 尝试初始化图像处理器
            try:
                image_config = self.config.apply_model_config('image')
                processors['image'] = ImageProcessor(**image_config)
            except Exception as e:
                logger.error(f"初始化图像处理器失败，使用模拟处理器: {e}")
                # 创建一个简单的模拟图像处理器
                class MockImageProcessor(BaseProcessor):
                    def chunk(self, content):
                        return [content]
                    def clean(self, content):
                        return content
                    def embed(self, content):
                        import random
                        return [random.random() for _ in range(512)]
                processors['image'] = MockImageProcessor()
            
            # 尝试初始化视频处理器
            try:
                video_config = self.config.apply_model_config('image')  # 视频处理器使用图像模型配置
                processors['video'] = VideoProcessor(**video_config)
            except Exception as e:
                logger.error(f"初始化视频处理器失败，使用模拟处理器: {e}")
                # 创建一个简单的模拟视频处理器
                class MockVideoProcessor(BaseProcessor):
                    def chunk(self, content):
                        return [content]
                    def clean(self, content):
                        return content
                    def embed(self, content):
                        import random
                        return [random.random() for _ in range(512)]
                processors['video'] = MockVideoProcessor()

            # 初始化音频处理器
            try:
                processors['audio'] = AudioProcessor(test_mode=False)
            except Exception as e:
                logger.error(f"初始化音频处理器失败，使用模拟处理器: {e}")
                processors['audio'] = AudioProcessor(test_mode=True)
        
        return processors
    
    def _init_readers(self) -> Dict[str, type]:
        readers = {
            'text': TextDocumentReader,
            'image': ImageDocumentReader,
            'video': VideoDocumentReader,
            'audio': AudioDocumentReader
        }
        return readers
    
    def _get_reader(self, file_extension: str, file_path: str) -> Optional[BaseDocumentReader]:
        ext = file_extension.lower()
        if ext in ['.txt', '.md', '.docx', '.pdf']:
            return self.readers['text'](file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return self.readers['image'](file_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            return self.readers['video'](file_path)
        elif ext in ['.mp3', '.wav']:
            return self.readers['audio'](file_path)
        return None
    
    def _get_processor(self, file_type: str) -> Optional[BaseProcessor]:
        # 映射文件类型到处理器类型
        file_type_map = {
            'txt': 'text',
            'md': 'text',
            'docx': 'text',
            'pdf': 'text',
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'bmp': 'image',
            'mp4': 'video',
            'avi': 'video',
            'mov': 'video',
            'mp3': 'audio',
            'wav': 'audio'
        }
        
        # 获取对应的处理器类型
        processor_type = file_type_map.get(file_type, file_type)
        
        if processor_type == 'text':
            return self.processors['text']
        elif processor_type == 'image':
            return self.processors['image']
        elif processor_type == 'video':
            return self.processors['video']
        elif processor_type == 'audio':
            return self.processors.get('audio')
        return None
    
    def _get_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def process_file(self, file_path: str, collection_id: str) -> Dict[str, Any]:
        """处理单个文件的完整流程"""
        # 1. 文件输入处理
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 2. 文件读取和元数据提取
        file_extension = os.path.splitext(file_path)[1]
        reader = self._get_reader(file_extension, file_path)
        if not reader:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        reader.read()
        full_metadata = reader.extract_metadata()
        content = reader.get_content()
        
        # 生成文件ID和核心元数据
        file_id = str(uuid.uuid4())
        core_metadata = {
            'file_id': file_id,
            'file_type': full_metadata.get('file_type', 'unknown'),
            'collection_id': collection_id,
            'timestamp': full_metadata.get('timestamp'),
            'hash': self._get_file_hash(file_path)
        }
        
        # 3. 原始文件存储
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        stored_file_id = self.file_storage.store_file(file_path, file_bytes)
        full_metadata['storage_path'] = stored_file_id
        
        # 4. 处理流程
        processor = self._get_processor(core_metadata['file_type'])
        if not processor:
            raise ValueError(f"不支持的文件类型处理: {core_metadata['file_type']}")
        
        # Chunking
        chunks = processor.chunk(content)
        
        # Cleaning
        cleaned_chunks = [processor.clean(chunk) for chunk in chunks]
        
        # Embedding
        vectors = [processor.embed(chunk) for chunk in cleaned_chunks]
        
        # 5. 向量存储
        dimension = len(vectors[0]) if vectors else 0
        self.vector_db.create_collection(collection_id, dimension)
        
        # 为每个chunk创建元数据
        chunk_metadata = []
        for i, chunk in enumerate(cleaned_chunks):
            chunk_meta = core_metadata.copy()
            chunk_meta['chunk_id'] = f"{file_id}_chunk_{i}"
            chunk_meta['chunk_index'] = i
            chunk_meta['chunk_length'] = len(chunk)
            chunk_metadata.append(chunk_meta)
        
        self.vector_db.insert(collection_id, vectors, chunk_metadata)
        
        # 存储完整元数据
        full_metadata['file_id'] = file_id
        self.metadata_storage.store_metadata(full_metadata)
        
        return {
            'file_id': file_id,
            'collection_id': collection_id,
            'chunks_processed': len(chunks),
            'vectors_stored': len(vectors)
        }
    
    def process_batch_files(self, file_paths: List[str], collection_id: str) -> List[Dict[str, Any]]:
        """批量处理文件"""
        # 定义处理函数
        def process_file_wrapper(file_path):
            try:
                result = self.process_file(file_path, collection_id)
                return result
            except Exception as e:
                return {
                    'file_path': file_path,
                    'error': str(e)
                }
        
        # 使用并行处理器处理文件
        results = parallel_processor.process_batch(file_paths, process_file_wrapper)
        return results
    
    def search(self, collection_id: str, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """向量检索流程"""
        # 1. 生成查询向量
        processor = self.processors['text']
        query_vector = processor.embed(query)
        
        # 2. 在指定集合分区中检索
        # 3. 应用metadata过滤
        search_results = self.vector_db.search(query_vector, top_k, filters, collection_id)
        
        # 4. 返回核心元数据
        # 5. 通过核心元数据获取全部元数据
        # 6. 通过全部元数据获取原始文件
        enhanced_results = []
        for result in search_results:
            core_meta = result['metadata']
            file_id = core_meta['file_id']
            
            # 获取完整元数据
            full_metadata = self.metadata_storage.get_metadata(file_id)
            
            # 获取原始文件路径
            storage_path = full_metadata.get('storage_path')
            if storage_path:
                try:
                    file_content = self.file_storage.read_file(storage_path)
                    result['file_content'] = file_content
                except Exception as e:
                    result['file_content_error'] = str(e)
            
            result['full_metadata'] = full_metadata
            enhanced_results.append(result)
        
        return enhanced_results
    
    def filter_by_metadata(self, collection_id: str, filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """基于元数据过滤搜索结果"""
        # 这里简化实现，实际应该根据向量数据库的能力进行过滤
        # 目前先执行一个简单的搜索，然后在结果中过滤
        dummy_query = ""
        processor = self.processors['text']
        query_vector = processor.embed(dummy_query)
        
        results = self.vector_db.search(query_vector, top_k * 10, None, collection_id)  # 获取更多结果以便过滤
        
        # 过滤结果
        filtered_results = []
        for result in results:
            metadata = result['metadata']
            match = True
            for key, value in filters.items():
                if metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results

# 全局数据流程管理器实例，通过环境变量 TEST_MODE=true 控制测试模式
import os as _os
_test_mode = _os.environ.get('TEST_MODE', 'false').lower() == 'true'
data_flow_manager = DataFlowManager(test_mode=_test_mode)