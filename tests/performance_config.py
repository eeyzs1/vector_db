import os

# 测试配置
class PerformanceConfig:
    # 测试数据配置
    TEST_SIZES = [5000, 50000, 500000]  # 5K, 50K, 500K向量
    DIMENSIONS = [128, 384, 512, 960]  # 不同维度
    TOP_K_VALUES = [1, 5, 10, 50]  # 不同k值
    CONCURRENCY_LEVELS = [1, 5, 50, 250]  # 并发级别
    
    # 数据库配置
    VECTOR_DB_CONFIGS = {
        "faiss": {
            "class": "FAISSVectorDB",
            "path": "./data/vector_db/faiss"
        },
        "hnsw": {
            "class": "HNSWVectorDB",
            "path": "./data/vector_db/hnsw"
        },
        "annoy": {
            "class": "AnnoyVectorDB",
            "path": "./data/vector_db/annoy"
        }
    }
    
    # 测试结果保存目录
    RESULTS_DIR = "tests/performance_results"
    
    # 测试运行配置
    BATCH_SIZE = 1000  # 批量插入大小
    NUM_QUERY_VECTORS = 100  # 查询向量数量
    
    @classmethod
    def ensure_directories(cls):
        """确保测试目录存在"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        for db_type, config in cls.VECTOR_DB_CONFIGS.items():
            os.makedirs(config['path'], exist_ok=True)