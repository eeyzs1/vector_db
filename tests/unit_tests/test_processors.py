import unittest
import os
import sys
import pytest
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.processors.text_processor import TextProcessor
from core.processors.image_processor import ImageProcessor
from PIL import Image
import io

class TestProcessors(unittest.TestCase):
    def setUp(self):
        # 初始化测试对象，使用测试模式
        self.text_processor = TextProcessor(test_mode=True)
        self.image_processor = ImageProcessor(test_mode=True)
    
    @pytest.mark.timeout(2)
    def test_text_processor_chunk(self):
        """测试文本分块功能，预计执行时间：2秒"""
        test_text = " " * 1000  # 1000个空格
        chunks = self.text_processor.chunk(test_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    @pytest.mark.timeout(1)
    def test_text_processor_clean(self):
        """测试文本清洗功能，预计执行时间：1秒"""
        test_text = "  This is a  test  text  "
        cleaned_text = self.text_processor.clean(test_text)
        self.assertEqual(cleaned_text, "This is a test text")
    
    @pytest.mark.timeout(5)
    def test_text_processor_embed(self):
        """测试文本嵌入功能，预计执行时间：5秒"""
        test_text = "This is a test text"
        embedding = self.text_processor.embed(test_text)
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 384)  # 验证向量维度
    
    @pytest.mark.timeout(2)
    def test_image_processor_chunk(self):
        """测试图像分块功能，预计执行时间：2秒"""
        # 创建一个简单的测试图像
        img = Image.new('RGB', (100, 100), color='red')
        chunks = self.image_processor.chunk(img)
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 1)
    
    @pytest.mark.timeout(2)
    def test_image_processor_clean(self):
        """测试图像清洗功能，预计执行时间：2秒"""
        # 创建一个测试图像
        img = Image.new('L', (100, 100), color=128)  # 灰度图像
        cleaned_img = self.image_processor.clean(img)
        self.assertEqual(cleaned_img.mode, 'RGB')
    
    @pytest.mark.timeout(10)
    def test_image_processor_embed(self):
        """测试图像嵌入功能，预计执行时间：10秒"""
        # 创建一个测试图像
        img = Image.new('RGB', (100, 100), color='blue')
        embedding = self.image_processor.embed(img)
        self.assertIsInstance(embedding, list)
        # 验证向量维度（真实模型返回512维，模拟返回512维）
        self.assertEqual(len(embedding), 512)

if __name__ == '__main__':
    unittest.main()