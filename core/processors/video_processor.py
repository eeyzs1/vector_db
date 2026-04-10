import cv2
from typing import List, Any
from PIL import Image
from sentence_transformers import SentenceTransformer
from .base_processor import BaseProcessor
from config.config import config
from core.model_manager import model_manager

class VideoProcessor(BaseProcessor):
    def __init__(self, model_type=None, model_name=None, model_path=None):
        # 使用配置中的参数，如果没有提供
        self.model_type = model_type or config.get('IMAGE_PROCESSING_MODEL_TYPE', 'local')
        self.model_name = model_name or config.get('IMAGE_PROCESSING_MODEL_NAME', 'OFA-Sys/chinese-clip-vit-base-patch16')
        self.model_path = model_path or config.get('IMAGE_PROCESSING_MODEL_PATH', './models/image')
        
        # 确保模型可用
        if self.model_type == 'local':
            model_manager.ensure_model_available(self.model_name, 'image')
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.use_real_model = True
        except Exception as e:
            print(f"加载模型失败，使用模拟模型: {e}")
            self.use_real_model = False
    
    def chunk(self, content: List[cv2.Mat]) -> List[List[cv2.Mat]]:
        # 视频分块（这里简单返回原始帧列表）
        return [content]
    
    def clean(self, content: List[cv2.Mat]) -> List[cv2.Mat]:
        # 简单的视频帧清洗
        cleaned_frames = []
        for frame in content:
            # 转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cleaned_frames.append(frame)
        return cleaned_frames
    
    def embed(self, content: List[cv2.Mat]) -> List[float]:
        # 对视频帧进行嵌入，然后取平均值
        if self.use_real_model:
            try:
                embeddings = []
                for frame in content[:10]:  # 只取前10帧以提高速度
                    # 转换为PIL Image
                    pil_image = Image.fromarray(frame)
                    # 生成嵌入
                    embedding = self.model.encode(pil_image)
                    embeddings.append(embedding)
                
                # 计算平均嵌入
                if embeddings:
                    avg_embedding = sum(embeddings) / len(embeddings)
                    return avg_embedding.tolist()
            except Exception as e:
                print(f"嵌入错误: {e}")
        
        # 使用模拟嵌入向量
        import random
        return [random.random() for _ in range(512)]