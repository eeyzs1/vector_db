import unittest
import os
import sys
import tempfile
import time
import pytest
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import ConfigManager

class TestConfig(unittest.TestCase):
    def setUp(self):
        # 创建临时环境文件
        self.temp_env = tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False)
        self.temp_env.write('TEXT_PROCESSING_MODEL_NAME=test-model\n')
        self.temp_env.write('VECTOR_DB_TYPE=faiss\n')
        self.temp_env.write('METADATA_STORAGE_TYPE=mysql\n')
        self.temp_env.write('FILE_STORAGE_TYPE=local\n')
        self.temp_env.write('API_SECRET_KEY=test-secret-key\n')
        self.temp_env.close()
        
        # 初始化配置管理器
        self.config = ConfigManager(env_file=self.temp_env.name)
    
    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.temp_env.name):
            os.remove(self.temp_env.name)
    
    @pytest.mark.timeout(0.1)
    def test_config_loading(self):
        """测试配置加载功能，预计执行时间：0.1秒"""
        self.assertEqual(self.config.get('TEXT_PROCESSING_MODEL_NAME'), 'test-model')
        self.assertEqual(self.config.get('VECTOR_DB_TYPE'), 'faiss')
        self.assertEqual(self.config.get('METADATA_STORAGE_TYPE'), 'mysql')
        self.assertEqual(self.config.get('FILE_STORAGE_TYPE'), 'local')
    
    @pytest.mark.timeout(0.1)
    def test_config_default_values(self):
        """测试默认值功能，预计执行时间：0.1秒"""
        self.assertEqual(self.config.get('TEXT_PROCESSING_MODEL_TYPE'), 'local')
        self.assertEqual(self.config.get('VECTOR_DB_PATH'), './data/vector_db')
        self.assertEqual(self.config.get('LOCAL_STORAGE_PATH'), './data/files')
    
    @pytest.mark.timeout(0.1)
    def test_config_validation(self):
        """测试配置验证功能，预计执行时间：0.1秒"""
        errors = self.config.validate_config()
        self.assertIsInstance(errors, list)
        # 由于使用了有效的配置，错误列表应该为空
        self.assertEqual(len(errors), 0)
    
    @pytest.mark.timeout(0.1)
    def test_model_config_application(self):
        """测试模型配置应用功能，预计执行时间：0.1秒"""
        text_config = self.config.apply_model_config('text')
        self.assertIsInstance(text_config, dict)
        self.assertEqual(text_config['model_name'], 'test-model')
        self.assertEqual(text_config['model_type'], 'local')
        
        image_config = self.config.apply_model_config('image')
        self.assertIsInstance(image_config, dict)
        self.assertEqual(image_config['model_type'], 'local')
        
        embedding_config = self.config.apply_model_config('embedding')
        self.assertIsInstance(embedding_config, dict)
        self.assertEqual(embedding_config['model_type'], 'local')

if __name__ == '__main__':
    unittest.main()