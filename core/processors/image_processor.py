import time
from typing import List, Any
from PIL import Image
from .base_processor import BaseProcessor
from config.config import config
from core.model_manager import model_manager
from sentence_transformers import SentenceTransformer

class ImageProcessor(BaseProcessor):
    def __init__(self, model_type=None, model_name=None, model_path=None, api_key=None, base_url=None, test_mode=False):
        # 使用配置中的参数，如果没有提供
        model_config = config.apply_model_config('image')
        self.model_type = model_type or model_config['model_type']
        self.model_name = model_name or model_config['model_name']
        self.model_path = model_path or model_config['model_path']
        self.api_key = api_key or model_config['api_key']
        self.base_url = base_url or model_config['base_url']
        self.test_mode = test_mode
        
        # 确保模型可用
        if self.model_type == 'local' and not self.test_mode:
            try:
                model_manager.ensure_model_available(self.model_name, 'image')
            except Exception as e:
                print(f"模型下载失败: {e}")
        
        try:
            if not self.test_mode:
                self.model = SentenceTransformer(self.model_name)
                self.use_real_model = True
            else:
                raise Exception("Test mode: skipping model loading")
        except Exception as e:
            print(f"加载模型失败，使用模拟模型: {e}")
            self.use_real_model = False

    
    def chunk(self, content: Image.Image) -> List[Image.Image]:
        # 图像分块（这里简单返回原始图像）
        return [content]
    
    def clean(self, content: Image.Image) -> Image.Image:
        # 简单的图像清洗
        # 转换为RGB模式
        if content.mode != 'RGB':
            content = content.convert('RGB')
        return content
    
    def embed(self, content: Image.Image) -> List[float]:
        start_time = time.time()
        # 使用CLIP生成图像嵌入向量
        if self.use_real_model:
            try:
                embedding = self.model.encode(content)
                embedding = embedding.tolist()
            except Exception as e:
                print(f"Embedding error: {e}")
                # 失败时使用模拟向量
                import random
                embedding = [random.random() for _ in range(512)]
        else:
            # 使用模拟嵌入向量
            import random
            embedding = [random.random() for _ in range(512)]
        
        # 记录模型使用情况
        processing_time = time.time() - start_time
        model_manager.record_model_usage(self.model_name, 'image', processing_time)
        
        return embedding