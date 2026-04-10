import unittest
import os
import sys
import pytest
from PIL import Image
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.processors.video_processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        # 初始化测试对象，使用测试模式
        self.video_processor = VideoProcessor(test_mode=True)
    
    @pytest.mark.timeout(2)
    def test_video_processor_chunk(self):
        """测试视频分块功能，预计执行时间：2秒"""
        # 创建一个测试图像作为视频帧
        test_frame = Image.new('RGB', (100, 100), color='red')
        chunks = self.video_processor.chunk(test_frame)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    @pytest.mark.timeout(2)
    def test_video_processor_clean(self):
        """测试视频清洗功能，预计执行时间：2秒"""
        # 创建一个测试图像作为视频帧
        test_frame = Image.new('L', (100, 100), color=128)  # 灰度图像
        cleaned_frame = self.video_processor.clean(test_frame)
        self.assertEqual(cleaned_frame.mode, 'RGB')
    
    @pytest.mark.timeout(10)
    def test_video_processor_embed(self):
        """测试视频嵌入功能，预计执行时间：10秒"""
        # 创建一个测试图像作为视频帧
        test_frame = Image.new('RGB', (100, 100), color='blue')
        embedding = self.video_processor.embed(test_frame)
        self.assertIsInstance(embedding, list)
        # 验证向量维度（真实模型返回512维，模拟返回512维）
        self.assertEqual(len(embedding), 512)

if __name__ == '__main__':
    unittest.main()