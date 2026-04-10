import pytest
from vector_db.core.readers.text_reader import TextDocumentReader
from vector_db.core.readers.image_reader import ImageDocumentReader
from vector_db.core.readers.video_reader import VideoDocumentReader
from vector_db.core.readers.audio_reader import AudioDocumentReader

class TestFileReaders:
    def test_text_document_reader(self):
        """测试文本文件读取器"""
        # 创建一个临时文本文件
        test_content = "这是一个测试文本文件"
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write(test_content)
        
        try:
            # 测试文本文件读取
            reader = TextDocumentReader("test.txt")
            content = reader.read()
            assert content == test_content
            
            # 测试元数据提取
            metadata = reader.extract_metadata()
            assert "file_type" in metadata
            assert "file_size" in metadata
            
            # 测试获取内容
            content = reader.get_content()
            assert content == test_content
        finally:
            # 清理临时文件
            import os
            if os.path.exists("test.txt"):
                os.remove("test.txt")
    
    def test_image_document_reader(self):
        """测试图像文件读取器"""
        # 创建一个临时图像文件
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save("test.jpg")
        
        try:
            # 测试图像文件读取
            reader = ImageDocumentReader("test.jpg")
            content = reader.read()
            assert content is not None
            
            # 测试元数据提取
            metadata = reader.extract_metadata()
            assert "file_type" in metadata
            assert "file_size" in metadata
            assert "width" in metadata
            assert "height" in metadata
            
            # 测试获取内容
            content = reader.get_content()
            assert content is not None
            # 关闭图像以释放文件
            if hasattr(reader, 'image') and reader.image:
                reader.image.close()
        finally:
            # 清理临时文件
            import os
            import time
            time.sleep(0.1)  # 等待文件释放
            if os.path.exists("test.jpg"):
                os.remove("test.jpg")
    
    def test_video_document_reader(self):
        """测试视频文件读取器"""
        # 创建一个临时视频文件
        import cv2
        import numpy as np
        
        # 创建一个简单的视频
        height, width = 100, 100
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test.avi', fourcc, 20.0, (width, height))
        
        # 生成一些帧
        for i in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f'Frame {i}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        out.release()
        
        try:
            # 测试视频文件读取
            reader = VideoDocumentReader("test.avi")
            content = reader.read()
            assert content is not None
            
            # 测试元数据提取
            metadata = reader.extract_metadata()
            assert "file_type" in metadata
            assert "file_size" in metadata
            
            # 测试获取内容
            content = reader.get_content()
            assert content is not None
        finally:
            # 清理临时文件
            import os
            if os.path.exists("test.avi"):
                os.remove("test.avi")
    
    def test_audio_document_reader(self):
        """测试音频文件读取器"""
        # 创建一个临时音频文件
        import numpy as np
        import soundfile as sf
        
        # 生成一些音频数据
        samplerate = 44100
        duration = 1  # 1秒
        t = np.linspace(0, duration, int(samplerate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz 正弦波
        
        sf.write('test.wav', audio, samplerate)
        
        try:
            # 测试音频文件读取
            reader = AudioDocumentReader("test.wav")
            content = reader.read()
            assert content is not None
            
            # 测试元数据提取
            metadata = reader.extract_metadata()
            assert "file_type" in metadata
            assert "file_size" in metadata
            
            # 测试获取内容
            content = reader.get_content()
            assert content is not None
        finally:
            # 清理临时文件
            import os
            if os.path.exists("test.wav"):
                os.remove("test.wav")