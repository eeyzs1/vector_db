import time
import pickle
from typing import Dict, Any, Optional
from collections import OrderedDict

class CacheManager:
    def __init__(self, max_size: int = 1000, expiration_time: int = 3600,
                 use_redis: bool = False, redis_host: str = 'localhost',
                 redis_port: int = 6379, redis_password: str = '',
                 redis_db: int = 0):
        self.max_size = max_size
        self.expiration_time = expiration_time
        self.use_redis = use_redis

        if use_redis:
            import redis
            self._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password or None,
                db=redis_db,
                decode_responses=False,
            )
        else:
            self._redis = None
            self.cache = OrderedDict()
            self.cache_times = {}

    def get(self, key: str) -> Optional[Any]:
        if self.use_redis:
            data = self._redis.get(key)
            if data is None:
                return None
            return pickle.loads(data)

        if key not in self.cache:
            return None
        if time.time() - self.cache_times[key] > self.expiration_time:
            self.remove(key)
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        self.cache_times[key] = time.time()
        return value

    def set(self, key: str, value: Any) -> None:
        if self.use_redis:
            self._redis.setex(key, self.expiration_time, pickle.dumps(value))
            return

        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.remove(oldest_key)
        self.cache[key] = value
        self.cache_times[key] = time.time()

    def remove(self, key: str) -> None:
        if self.use_redis:
            self._redis.delete(key)
            return
        if key in self.cache:
            del self.cache[key]
            del self.cache_times[key]

    def clear(self) -> None:
        if self.use_redis:
            self._redis.flushdb()
            return
        self.cache.clear()
        self.cache_times.clear()

    def size(self) -> int:
        if self.use_redis:
            return self._redis.dbsize()
        return len(self.cache)

    def contains(self, key: str) -> bool:
        if self.use_redis:
            return self._redis.exists(key) > 0
        return key in self.cache and time.time() - self.cache_times[key] <= self.expiration_time


# 全局缓存管理器实例，通过环境变量 CACHE_BACKEND=redis 启用 Redis
import os as _os
_cache_backend = _os.environ.get('CACHE_BACKEND', 'memory').lower()
if _cache_backend == 'redis':
    from config.config import config as _config
    cache_manager = CacheManager(
        use_redis=True,
        redis_host=_config.get('REDIS_HOST', 'localhost'),
        redis_port=int(_config.get('REDIS_PORT', 6379)),
        redis_password=_config.get('REDIS_PASSWORD', ''),
        redis_db=int(_config.get('REDIS_DB', 0)),
    )
else:
    cache_manager = CacheManager()