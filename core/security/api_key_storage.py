"""API Key 持久化存储"""
import json
import os
from typing import Set, Optional
from abc import ABC, abstractmethod


class APIKeyStorageInterface(ABC):
    """API Key 存储接口"""

    @abstractmethod
    def add(self, api_key: str) -> None:
        """添加 API Key"""
        pass

    @abstractmethod
    def remove(self, api_key: str) -> None:
        """移除 API Key"""
        pass

    @abstractmethod
    def exists(self, api_key: str) -> bool:
        """检查 API Key 是否存在"""
        pass

    @abstractmethod
    def list_all(self) -> Set[str]:
        """列出所有 API Key"""
        pass


class RedisAPIKeyStorage(APIKeyStorageInterface):
    """Redis 存储实现"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.key_prefix = "api_key:"

    def add(self, api_key: str) -> None:
        self.redis.sadd("api_keys", api_key)

    def remove(self, api_key: str) -> None:
        self.redis.srem("api_keys", api_key)

    def exists(self, api_key: str) -> bool:
        return self.redis.sismember("api_keys", api_key)

    def list_all(self) -> Set[str]:
        return {k.decode() if isinstance(k, bytes) else k for k in self.redis.smembers("api_keys")}


class FileAPIKeyStorage(APIKeyStorageInterface):
    """文件存储实现（fallback）"""

    def __init__(self, file_path: str = "./data/api_keys.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            self._save_keys(set())

    def _load_keys(self) -> Set[str]:
        try:
            with open(self.file_path, 'r') as f:
                return set(json.load(f))
        except:
            return set()

    def _save_keys(self, keys: Set[str]) -> None:
        with open(self.file_path, 'w') as f:
            json.dump(list(keys), f)

    def add(self, api_key: str) -> None:
        keys = self._load_keys()
        keys.add(api_key)
        self._save_keys(keys)

    def remove(self, api_key: str) -> None:
        keys = self._load_keys()
        keys.discard(api_key)
        self._save_keys(keys)

    def exists(self, api_key: str) -> bool:
        return api_key in self._load_keys()

    def list_all(self) -> Set[str]:
        return self._load_keys()
