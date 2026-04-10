import time
from typing import List, Any
from .base_processor import BaseProcessor
from config.config import config
from core.model_manager import model_manager

class TextProcessor(BaseProcessor):
    def __init__(self, model_type=None, model_name=None, api_key=None, model_path=None, base_url=None, test_mode=False):
        model_config = config.apply_model_config('text')
        self.model_type = model_type or model_config['model_type']
        self.model_name = model_name or model_config['model_name']
        self.api_key = api_key or model_config['api_key']
        self.model_path = model_path or model_config['model_path']
        self.base_url = base_url or model_config['base_url']
        self.test_mode = test_mode
        self._model = None

        if not self.test_mode and self.model_type == 'local':
            try:
                model_manager.ensure_model_available(self.model_name, 'text')
            except Exception as e:
                print(f"模型下载失败: {e}")

    def _get_model(self):
        """懒加载 sentence-transformers 模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk(self, content: str) -> List[str]:
        words = content.split()
        if not words:
            return ['']
        chunk_size = 100
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def clean(self, content: str) -> str:
        return ' '.join(content.split())

    def embed(self, content: str) -> List[float]:
        start_time = time.time()

        if self.test_mode:
            import random
            embedding = [random.random() for _ in range(384)]
        elif self.model_type == 'local':
            model = self._get_model()
            embedding = model.encode(content).tolist()
        else:
            # 远程 API 嵌入（OpenAI 兼容接口）
            import requests
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={'Authorization': f'Bearer {self.api_key}'},
                json={'input': content, 'model': self.model_name}
            )
            response.raise_for_status()
            embedding = response.json()['data'][0]['embedding']

        processing_time = time.time() - start_time
        model_manager.record_model_usage(self.model_name, 'text', processing_time)
        return embedding