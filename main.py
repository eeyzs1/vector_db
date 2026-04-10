from core.readers.text_reader import TextDocumentReader
from core.readers.image_reader import ImageDocumentReader
from core.readers.video_reader import VideoDocumentReader
from core.readers.audio_reader import AudioDocumentReader
from core.processors.text_processor import TextProcessor
from core.processors.image_processor import ImageProcessor
from core.processors.video_processor import VideoProcessor
from core.vector_db.faiss_vector_db import FAISSVectorDB
from core.storage.local_file_storage import LocalFileSystemStorage
from core.storage.mysql_storage import MySQLStorage
from config.config import config


def test_text_processing():
    print("Testing text processing...")
    # 创建文本读取器
    reader = TextDocumentReader("example.txt")
    # 读取文本
    content = reader.read()
    print(f"Read content: {content[:100]}...")
    # 提取元数据
    metadata = reader.extract_metadata()
    print(f"Metadata: {metadata}")
    # 创建文本处理器
    processor = TextProcessor()
    # 分块
    chunks = processor.chunk(content)
    print(f"Chunks: {len(chunks)} chunks")
    # 清洗
    cleaned_content = processor.clean(content)
    print(f"Cleaned content: {cleaned_content[:100]}...")
    # 嵌入
    embedding = processor.embed(content)
    print(f"Embedding length: {len(embedding)}")
    print("Text processing test completed successfully!")


def test_vector_db():
    print("Testing vector database...")
    # 创建向量数据库
    vector_db = FAISSVectorDB()
    # 创建集合
    vector_db.create_collection("test_collection", 384)  # MiniLM-L6-v2维度为384
    # 测试向量
    vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    metadata = [
        {"file_id": "1", "file_type": "text", "collection_id": "test_collection"},
        {"file_id": "2", "file_type": "text", "collection_id": "test_collection"},
        {"file_id": "3", "file_type": "text", "collection_id": "test_collection"}
    ]
    # 插入向量
    vector_ids = vector_db.insert("test_collection", vectors, metadata)
    print(f"Inserted vectors with IDs: {vector_ids}")
    # 搜索向量
    query_vector = [0.15] * 384
    results = vector_db.search("test_collection", query_vector, top_k=2)
    print(f"Search results: {results}")
    print("Vector database test completed successfully!")


if __name__ == "__main__":
    print("Starting vector database system...")
    # 测试文本处理
    test_text_processing()
    # 测试向量数据库
    test_vector_db()
    print("All tests completed successfully!")