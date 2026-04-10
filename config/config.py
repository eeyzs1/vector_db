import os
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, env_file='.env'):
        self.env_file = env_file
        self.config = self._load_config()
    
    def _load_config(self):
        load_dotenv(self.env_file, override=True)
        config = {}
        
        # 文本处理模型配置
        config['TEXT_PROCESSING_MODEL_TYPE'] = os.getenv('TEXT_PROCESSING_MODEL_TYPE', 'local')
        config['TEXT_PROCESSING_MODEL_NAME'] = os.getenv('TEXT_PROCESSING_MODEL_NAME', 'shibing624/text2vec-base-chinese')
        config['TEXT_PROCESSING_MODEL_PATH'] = os.getenv('TEXT_PROCESSING_MODEL_PATH', './models/text')
        config['TEXT_PROCESSING_API_KEY'] = os.getenv('TEXT_PROCESSING_API_KEY', '')
        config['TEXT_PROCESSING_BASE_URL'] = os.getenv('TEXT_PROCESSING_BASE_URL', 'https://api.openai.com/v1')
        
        # 图像处理模型配置
        config['IMAGE_PROCESSING_MODEL_TYPE'] = os.getenv('IMAGE_PROCESSING_MODEL_TYPE', 'local')
        config['IMAGE_PROCESSING_MODEL_NAME'] = os.getenv('IMAGE_PROCESSING_MODEL_NAME', 'OFA-Sys/chinese-clip-vit-base-patch16')
        config['IMAGE_PROCESSING_MODEL_PATH'] = os.getenv('IMAGE_PROCESSING_MODEL_PATH', './models/image')
        config['IMAGE_PROCESSING_API_KEY'] = os.getenv('IMAGE_PROCESSING_API_KEY', '')
        config['IMAGE_PROCESSING_BASE_URL'] = os.getenv('IMAGE_PROCESSING_BASE_URL', '')
        
        # 文本清洗模型配置
        config['TEXT_CLEANING_MODEL_TYPE'] = os.getenv('TEXT_CLEANING_MODEL_TYPE', 'local')
        config['TEXT_CLEANING_MODEL_NAME'] = os.getenv('TEXT_CLEANING_MODEL_NAME', 'shibing624/text2vec-base-chinese')
        config['TEXT_CLEANING_MODEL_PATH'] = os.getenv('TEXT_CLEANING_MODEL_PATH', './models/text')
        config['TEXT_CLEANING_API_KEY'] = os.getenv('TEXT_CLEANING_API_KEY', '')
        config['TEXT_CLEANING_BASE_URL'] = os.getenv('TEXT_CLEANING_BASE_URL', 'https://api.openai.com/v1')
        
        # 图像清洗模型配置
        config['IMAGE_CLEANING_MODEL_TYPE'] = os.getenv('IMAGE_CLEANING_MODEL_TYPE', 'local')
        config['IMAGE_CLEANING_MODEL_NAME'] = os.getenv('IMAGE_CLEANING_MODEL_NAME', 'OFA-Sys/chinese-clip-vit-base-patch16')
        config['IMAGE_CLEANING_MODEL_PATH'] = os.getenv('IMAGE_CLEANING_MODEL_PATH', './models/image')
        config['IMAGE_CLEANING_API_KEY'] = os.getenv('IMAGE_CLEANING_API_KEY', '')
        config['IMAGE_CLEANING_BASE_URL'] = os.getenv('IMAGE_CLEANING_BASE_URL', '')
        
        # 向量嵌入模型配置
        config['EMBEDDING_MODEL_TYPE'] = os.getenv('EMBEDDING_MODEL_TYPE', 'local')
        config['EMBEDDING_MODEL_NAME'] = os.getenv('EMBEDDING_MODEL_NAME', 'shibing624/text2vec-base-chinese')
        config['EMBEDDING_MODEL_PATH'] = os.getenv('EMBEDDING_MODEL_PATH', './models/embedding')
        config['EMBEDDING_API_KEY'] = os.getenv('EMBEDDING_API_KEY', '')
        config['EMBEDDING_BASE_URL'] = os.getenv('EMBEDDING_BASE_URL', '')
        
        # 向量数据库配置
        config['VECTOR_DB_TYPE'] = os.getenv('VECTOR_DB_TYPE', 'faiss')
        config['VECTOR_DB_PATH'] = os.getenv('VECTOR_DB_PATH', './data/vector_db')
        
        # 元数据存储配置
        config['METADATA_STORAGE_TYPE'] = os.getenv('METADATA_STORAGE_TYPE', 'mysql')
        config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
        config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', '3306'))
        config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
        config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'password')
        config['MYSQL_DATABASE'] = os.getenv('MYSQL_DATABASE', 'vector_db')
        
        # Redis配置
        config['REDIS_HOST'] = os.getenv('REDIS_HOST', 'localhost')
        config['REDIS_PORT'] = int(os.getenv('REDIS_PORT', '6379'))
        config['REDIS_PASSWORD'] = os.getenv('REDIS_PASSWORD', '')
        config['REDIS_DB'] = int(os.getenv('REDIS_DB', '0'))
        
        # MongoDB配置
        config['MONGO_HOST'] = os.getenv('MONGO_HOST', 'localhost')
        config['MONGO_PORT'] = int(os.getenv('MONGO_PORT', '27017'))
        config['MONGO_USER'] = os.getenv('MONGO_USER', '')
        config['MONGO_PASSWORD'] = os.getenv('MONGO_PASSWORD', '')
        config['MONGO_DATABASE'] = os.getenv('MONGO_DATABASE', 'vector_db')
        
        # 文件存储配置
        config['FILE_STORAGE_TYPE'] = os.getenv('FILE_STORAGE_TYPE', 'local')
        config['LOCAL_STORAGE_PATH'] = os.getenv('LOCAL_STORAGE_PATH', './data/files')
        
        # S3配置
        config['S3_BUCKET_NAME'] = os.getenv('S3_BUCKET_NAME', '')
        config['S3_ACCESS_KEY'] = os.getenv('S3_ACCESS_KEY', '')
        config['S3_SECRET_KEY'] = os.getenv('S3_SECRET_KEY', '')
        config['S3_REGION'] = os.getenv('S3_REGION', 'us-east-1')
        
        # API配置
        config['API_PORT'] = int(os.getenv('API_PORT', '8000'))
        config['API_HOST'] = os.getenv('API_HOST', '0.0.0.0')
        config['API_SECRET_KEY'] = os.getenv('API_SECRET_KEY', 'your-api-secret-key')
        config['BASE_URL'] = os.getenv('BASE_URL', f'http://{os.getenv("API_HOST", "localhost")}:{os.getenv("API_PORT", "8000")}')
        
        # 日志配置
        config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'INFO')
        config['LOG_FILE'] = os.getenv('LOG_FILE', './logs/app.log')

        # 缓存配置
        config['CACHE_BACKEND'] = os.getenv('CACHE_BACKEND', 'memory')
        
        return config
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def get_all(self):
        return self.config
    
    def validate_config(self):
        """验证配置的有效性"""
        validation_errors = []
        
        # 验证向量数据库配置
        vector_db_type = self.get('VECTOR_DB_TYPE')
        if vector_db_type not in ['faiss', 'hnsw', 'annoy']:
            validation_errors.append(f"无效的向量数据库类型: {vector_db_type}")
        
        # 验证元数据存储配置
        metadata_storage_type = self.get('METADATA_STORAGE_TYPE')
        if metadata_storage_type not in ['mysql', 'redis', 'mongodb', 'memory']:
            validation_errors.append(f"无效的元数据存储类型: {metadata_storage_type}")
        
        # 验证文件存储配置
        file_storage_type = self.get('FILE_STORAGE_TYPE')
        if file_storage_type not in ['local', 's3', 'database']:
            validation_errors.append(f"无效的文件存储类型: {file_storage_type}")
        
        # 验证模型类型配置
        embedding_model_type = self.get('EMBEDDING_MODEL_TYPE')
        if embedding_model_type not in ['local', 'remote']:
            validation_errors.append(f"无效的嵌入模型类型: {embedding_model_type}")
        
        # 验证路径配置
        vector_db_path = self.get('VECTOR_DB_PATH')
        if not vector_db_path:
            validation_errors.append("向量数据库路径未配置")
        
        local_storage_path = self.get('LOCAL_STORAGE_PATH')
        if not local_storage_path:
            validation_errors.append("本地存储路径未配置")
        
        # 验证API配置
        api_secret_key = self.get('API_SECRET_KEY')
        if api_secret_key == 'your-api-secret-key':
            validation_errors.append("API密钥未设置，请修改配置")
        
        return validation_errors
    
    def apply_model_config(self, model_type: str) -> dict:
        """根据模型类型获取应用的模型配置"""
        if model_type == 'text':
            return {
                'model_type': self.get('TEXT_PROCESSING_MODEL_TYPE'),
                'model_name': self.get('TEXT_PROCESSING_MODEL_NAME'),
                'model_path': self.get('TEXT_PROCESSING_MODEL_PATH'),
                'api_key': self.get('TEXT_PROCESSING_API_KEY'),
                'base_url': self.get('TEXT_PROCESSING_BASE_URL')
            }
        elif model_type == 'image':
            return {
                'model_type': self.get('IMAGE_PROCESSING_MODEL_TYPE'),
                'model_name': self.get('IMAGE_PROCESSING_MODEL_NAME'),
                'model_path': self.get('IMAGE_PROCESSING_MODEL_PATH'),
                'api_key': self.get('IMAGE_PROCESSING_API_KEY'),
                'base_url': self.get('IMAGE_PROCESSING_BASE_URL')
            }
        elif model_type == 'text_cleaning':
            return {
                'model_type': self.get('TEXT_CLEANING_MODEL_TYPE'),
                'model_name': self.get('TEXT_CLEANING_MODEL_NAME'),
                'model_path': self.get('TEXT_CLEANING_MODEL_PATH'),
                'api_key': self.get('TEXT_CLEANING_API_KEY'),
                'base_url': self.get('TEXT_CLEANING_BASE_URL')
            }
        elif model_type == 'image_cleaning':
            return {
                'model_type': self.get('IMAGE_CLEANING_MODEL_TYPE'),
                'model_name': self.get('IMAGE_CLEANING_MODEL_NAME'),
                'model_path': self.get('IMAGE_CLEANING_MODEL_PATH'),
                'api_key': self.get('IMAGE_CLEANING_API_KEY'),
                'base_url': self.get('IMAGE_CLEANING_BASE_URL')
            }
        elif model_type == 'embedding':
            return {
                'model_type': self.get('EMBEDDING_MODEL_TYPE'),
                'model_name': self.get('EMBEDDING_MODEL_NAME'),
                'model_path': self.get('EMBEDDING_MODEL_PATH'),
                'api_key': self.get('EMBEDDING_API_KEY'),
                'base_url': self.get('EMBEDDING_BASE_URL')
            }
        return {}

# 全局配置实例
config = ConfigManager()