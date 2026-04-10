import unittest
import os
import sys
import pytest
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.readers.text_reader import TextDocumentReader

class TestReaders(unittest.TestCase):
    def setUp(self):
        # 创建测试文件
        self.test_file_path = './test_file.txt'
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            f.write('This is a test file for text reader.\nIt contains multiple lines.')
        # 初始化测试对象
        self.text_reader = TextDocumentReader(self.test_file_path)
    
    def tearDown(self):
        # 清理测试文件
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
    
    @pytest.mark.timeout(1)
    def test_read_text_file(self):
        """测试读取文本文件，预计执行时间：1秒"""
        content = self.text_reader.read()
        self.assertIsInstance(content, str)
        self.assertIn('This is a test file', content)
    
    @pytest.mark.timeout(1)
    def test_extract_metadata(self):
        """测试提取元数据，预计执行时间：1秒"""
        metadata = self.text_reader.extract_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertTrue('file_path' in metadata)
        self.assertTrue('file_size' in metadata)
        self.assertTrue('file_type' in metadata)
    
    @pytest.mark.timeout(1)
    def test_get_content(self):
        """测试获取内容，预计执行时间：1秒"""
        content = self.text_reader.get_content()
        self.assertIsInstance(content, str)
        self.assertIn('This is a test file', content)

if __name__ == '__main__':
    unittest.main()