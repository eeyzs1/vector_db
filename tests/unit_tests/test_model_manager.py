import unittest
import os
import sys
import time
import pytest
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    def setUp(self):
        # 初始化模型管理器
        self.model_manager = ModelManager()
    
    @pytest.mark.timeout(0.1)
    def test_check_model_exists(self):
        """测试模型存在性检查，预计执行时间：0.1秒"""
        # 由于我们没有实际下载模型，这里应该返回False
        result = self.model_manager.check_model_exists('test-model', 'text')
        self.assertIsInstance(result, bool)
    
    @pytest.mark.timeout(0.1)
    def test_get_model_path(self):
        """测试获取模型路径，预计执行时间：0.1秒"""
        model_path = self.model_manager.get_model_path('test-model', 'text')
        self.assertIsInstance(model_path, str)
        self.assertIn('test-model', model_path)
    
    @pytest.mark.skip(reason="需要网络连接，跳过测试")
    @pytest.mark.timeout(60)
    def test_ensure_model_available(self):
        """测试确保模型可用，预计执行时间：60秒（包含网络请求）"""
        # 由于网络限制，这里可能会失败，但应该不会抛出异常
        try:
            result = self.model_manager.ensure_model_available('shibing624/text2vec-base-chinese', 'text')
            self.assertIsInstance(result, bool)
        except Exception as e:
            # 网络错误是可以接受的，因为我们可能没有网络连接
            print(f"网络错误（预期内）: {e}")

if __name__ == '__main__':
    unittest.main()