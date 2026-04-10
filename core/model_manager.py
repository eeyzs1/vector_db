import os
import subprocess
import json
import time
from datetime import datetime
from config.config import config

class ModelManager:
    def __init__(self):
        self.config = config
        self.model_paths = {
            'text': self.config.get('TEXT_PROCESSING_MODEL_PATH'),
            'image': self.config.get('IMAGE_PROCESSING_MODEL_PATH'),
            'text_cleaning': self.config.get('TEXT_CLEANING_MODEL_PATH'),
            'image_cleaning': self.config.get('IMAGE_CLEANING_MODEL_PATH'),
            'embedding': self.config.get('EMBEDDING_MODEL_PATH')
        }
        # 确保模型目录存在
        for path in self.model_paths.values():
            os.makedirs(path, exist_ok=True)
        # 国内源配置
        self.huggingface_mirrors = [
            "https://mirror.sjtu.edu.cn/huggingface",
            "https://hf-mirror.com",
            "https://modelscope.cn"
        ]
        # 模型使用统计
        self.model_stats = {}
        # 模型版本管理
        self.model_versions = {}
        # 加载模型版本信息
        self._load_model_versions()
    
    def check_model_exists(self, model_name: str, model_type: str = 'text') -> bool:
        """检查模型是否存在"""
        model_path = self.model_paths.get(model_type)
        if not model_path:
            return False
        
        # 检查模型目录是否存在且非空
        model_dir = os.path.join(model_path, model_name.replace('/', '_'))
        return os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0
    
    def download_model(self, model_name: str, model_type: str = 'text') -> bool:
        """自动下载模型，尝试国内源"""
        try:
            # 使用sentence-transformers的方式下载模型
            from sentence_transformers import SentenceTransformer
            
            model_path = self.model_paths.get(model_type)
            if not model_path:
                return False
            
            # 尝试使用国内源
            for mirror in self.huggingface_mirrors:
                try:
                    # 设置环境变量
                    os.environ['HF_ENDPOINT'] = mirror
                    print(f"尝试从国内源下载模型: {mirror}")
                    
                    # 下载模型
                    model = SentenceTransformer(model_name, cache_folder=model_path)
                    model_dir = os.path.join(model_path, model_name.replace('/', '_'))
                    model.save(model_dir)
                    
                    # 记录模型版本信息
                    self._record_model_version(model_name, model_type, model_dir)
                    return True
                except Exception as e:
                    print(f"从{mirror}下载失败: {e}")
                    continue
            
            # 如果所有国内源都失败，尝试默认源
            print("尝试从默认源下载模型...")
            if 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']
            model = SentenceTransformer(model_name, cache_folder=model_path)
            model_dir = os.path.join(model_path, model_name.replace('/', '_'))
            model.save(model_dir)
            
            # 记录模型版本信息
            self._record_model_version(model_name, model_type, model_dir)
            return True
        except Exception as e:
            print(f"下载模型失败: {e}")
            return False
    
    def ensure_model_available(self, model_name: str, model_type: str = 'text') -> bool:
        """确保模型可用，如果不存在则自动下载"""
        if self.check_model_exists(model_name, model_type):
            return True
        else:
            print(f"模型 {model_name} 不存在，开始自动下载...")
            return self.download_model(model_name, model_type)
    
    def get_model_path(self, model_name: str, model_type: str = 'text') -> str:
        """获取模型路径"""
        model_path = self.model_paths.get(model_type)
        if not model_path:
            return ''
        return os.path.join(model_path, model_name.replace('/', '_'))
    
    def _record_model_version(self, model_name: str, model_type: str, model_dir: str):
        """记录模型版本信息"""
        version_info = {
            'model_name': model_name,
            'model_type': model_type,
            'download_time': datetime.now().isoformat(),
            'model_path': model_dir,
            'version': self._generate_version()
        }
        
        if model_type not in self.model_versions:
            self.model_versions[model_type] = {}
        self.model_versions[model_type][model_name] = version_info
        self._save_model_versions()
    
    def _generate_version(self) -> str:
        """生成版本号"""
        return datetime.now().strftime('%Y%m%d%H%M%S')
    
    def _save_model_versions(self):
        """保存模型版本信息"""
        version_file = os.path.join(os.path.dirname(__file__), '../../data/model_versions.json')
        os.makedirs(os.path.dirname(version_file), exist_ok=True)
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_versions, f, ensure_ascii=False, indent=2)
    
    def _load_model_versions(self):
        """加载模型版本信息"""
        version_file = os.path.join(os.path.dirname(__file__), '../../data/model_versions.json')
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    self.model_versions = json.load(f)
            except Exception as e:
                print(f"加载模型版本信息失败: {e}")
                self.model_versions = {}
    
    def get_model_version(self, model_name: str, model_type: str = 'text') -> dict:
        """获取模型版本信息"""
        if model_type in self.model_versions and model_name in self.model_versions[model_type]:
            return self.model_versions[model_type][model_name]
        return {}
    
    def record_model_usage(self, model_name: str, model_type: str = 'text', processing_time: float = 0):
        """记录模型使用情况"""
        if model_type not in self.model_stats:
            self.model_stats[model_type] = {}
        if model_name not in self.model_stats[model_type]:
            self.model_stats[model_type][model_name] = {
                'usage_count': 0,
                'total_processing_time': 0,
                'last_used': None
            }
        
        stats = self.model_stats[model_type][model_name]
        stats['usage_count'] += 1
        stats['total_processing_time'] += processing_time
        stats['last_used'] = datetime.now().isoformat()
        
        # 保存使用统计
        self._save_model_stats()
    
    def _save_model_stats(self):
        """保存模型使用统计"""
        stats_file = os.path.join(os.path.dirname(__file__), '../../data/model_stats.json')
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_stats, f, ensure_ascii=False, indent=2)
    
    def get_model_stats(self, model_name: str = None, model_type: str = None) -> dict:
        """获取模型使用统计"""
        if model_type and model_name:
            if model_type in self.model_stats and model_name in self.model_stats[model_type]:
                return self.model_stats[model_type][model_name]
            return {}
        elif model_type:
            return self.model_stats.get(model_type, {})
        else:
            return self.model_stats
    
    def list_available_models(self) -> dict:
        """列出所有可用的模型"""
        available_models = {}
        for model_type, path in self.model_paths.items():
            if os.path.exists(path):
                model_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                available_models[model_type] = model_dirs
        return available_models
    
    def delete_model(self, model_name: str, model_type: str = 'text') -> bool:
        """删除模型"""
        model_path = self.get_model_path(model_name, model_type)
        if os.path.exists(model_path):
            try:
                import shutil
                shutil.rmtree(model_path)
                # 从版本信息中删除
                if model_type in self.model_versions and model_name in self.model_versions[model_type]:
                    del self.model_versions[model_type][model_name]
                    self._save_model_versions()
                # 从统计信息中删除
                if model_type in self.model_stats and model_name in self.model_stats[model_type]:
                    del self.model_stats[model_type][model_name]
                    self._save_model_stats()
                return True
            except Exception as e:
                print(f"删除模型失败: {e}")
                return False
        return False

# 全局模型管理器实例
model_manager = ModelManager()