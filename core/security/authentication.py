import time
import jwt
import secrets
from typing import Optional, Dict, Any
from config.config import config
from core.security.api_key_storage import APIKeyStorageInterface, RedisAPIKeyStorage, FileAPIKeyStorage
from core.security.user_storage import UserStorage

class AuthenticationManager:
    def __init__(self, storage: Optional[APIKeyStorageInterface] = None):
        """初始化认证管理器"""
        self.secret_key = config.get('API_SECRET_KEY', 'your-api-secret-key')
        self.user_storage = UserStorage()
        self.redis_client = None

        # 尝试初始化 Redis 客户端
        try:
            import redis
            self.redis_client = redis.Redis(
                host=config.get('REDIS_HOST', 'localhost'),
                port=int(config.get('REDIS_PORT', 6379)),
                password=config.get('REDIS_PASSWORD') or None,
                db=int(config.get('REDIS_DB', 0)),
                decode_responses=True
            )
            self.redis_client.ping()
        except:
            pass

        # 从配置中加载初始 API 密钥
        api_key = config.get('API_KEY')
        if api_key:
            # 添加默认用户
            self.add_user('default', api_key, 10)
    
    def validate_api_key(self, api_key: str) -> bool:
        """验证API密钥（仅内部使用，不推荐直接使用）

        Args:
            api_key: API密钥

        Returns:
            是否有效
        """
        # 从数据库验证
        user = self.user_storage.get_user_by_api_key(api_key)
        return user is not None
    
    def validate_user(self, username: str, api_key: str) -> Optional[Dict[str, Any]]:
        """验证用户和API密钥

        Args:
            username: 用户名
            api_key: API密钥

        Returns:
            用户信息，如果验证失败返回None
        """
        # 优先从 Redis 验证
        if self.redis_client:
            try:
                stored_api_key = self.redis_client.get(f"user:{username}:api_key")
                if stored_api_key == api_key:
                    level = self.redis_client.get(f"user:{username}:level")
                    return {
                        'username': username,
                        'api_key': api_key,
                        'level': int(level) if level else 0
                    }
            except:
                pass
        
        # 从数据库验证
        user = self.user_storage.get_user_by_username(username)
        if user and user['api_key'] == api_key:
            # 存入 Redis 缓存
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"user:{username}:api_key",
                        3600,  # 1小时过期
                        api_key
                    )
                    self.redis_client.setex(
                        f"user:{username}:level",
                        3600,
                        user['level']
                    )
                    self.redis_client.setex(
                        f"user:api_key:{api_key}",
                        3600,
                        username
                    )
                except:
                    pass
            return user
        
        return None
    
    def generate_jwt(self, user_id: str, role: str = 'user') -> str:
        """生成JWT令牌
        
        Args:
            user_id: 用户ID
            role: 用户角色
            
        Returns:
            JWT令牌
        """
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': time.time() + 3600  # 1小时过期
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            令牌 payload，如果无效返回None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def add_user(self, username: str, api_key: str, level: int = 0) -> bool:
        """添加用户

        Args:
            username: 用户名
            api_key: API密钥
            level: 用户级别
            
        Returns:
            是否添加成功
        """
        success = self.user_storage.add_user(username, api_key, level)
        
        # 同时添加到 Redis 缓存
        if success and self.redis_client:
            try:
                self.redis_client.setex(
                    f"user:{username}:api_key",
                    3600,  # 1小时过期
                    api_key
                )
                self.redis_client.setex(
                    f"user:{username}:level",
                    3600,
                    level
                )
                self.redis_client.setex(
                    f"user:api_key:{api_key}",
                    3600,
                    username
                )
            except:
                pass
        
        return success
    
    def remove_user(self, username: str) -> bool:
        """移除用户

        Args:
            username: 用户名
            
        Returns:
            是否移除成功
        """
        # 先获取用户信息
        user = self.user_storage.get_user_by_username(username)
        success = self.user_storage.delete_user(username)
        
        # 从 Redis 缓存中删除
        if success and user and self.redis_client:
            try:
                self.redis_client.delete(f"user:{username}:api_key")
                self.redis_client.delete(f"user:{username}:level")
                self.redis_client.delete(f"user:api_key:{user['api_key']}")
            except:
                pass
        
        return success
    
    def remove_api_key(self, api_key: str) -> bool:
        """通过API Key移除用户

        Args:
            api_key: API密钥
            
        Returns:
            是否移除成功
        """
        # 先获取用户信息
        user = self.user_storage.get_user_by_api_key(api_key)
        if not user:
            return False
        
        return self.remove_user(user['username'])
    
    def list_users(self) -> list:
        """列出所有用户

        Returns:
            用户列表（脱敏显示API密钥）
        """
        users = self.user_storage.list_users()
        for user in users:
            user['api_key'] = self._mask_key(user['api_key'])
        return users

    def generate_api_key(self, username: str, level: int = 0) -> str:
        """生成新的API密钥并关联到用户

        Args:
            username: 用户名
            level: 用户级别
            
        Returns:
            新生成的API密钥
        """
        api_key = f"vdb_{secrets.token_urlsafe(32)}"
        
        # 检查用户是否存在
        existing_user = self.user_storage.get_user_by_username(username)
        if existing_user:
            # 更新API密钥
            self.user_storage.update_user(username, api_key=api_key, level=level)
        else:
            # 添加新用户
            self.user_storage.add_user(username, api_key, level)
        
        # 更新 Redis 缓存
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"user:{username}:api_key",
                    3600,  # 1小时过期
                    api_key
                )
                self.redis_client.setex(
                    f"user:{username}:level",
                    3600,
                    level
                )
                self.redis_client.setex(
                    f"user:api_key:{api_key}",
                    3600,
                    username
                )
            except:
                pass
        
        return api_key

    def get_user_level(self, username: str) -> Optional[int]:
        """获取用户级别

        Args:
            username: 用户名
            
        Returns:
            用户级别
        """
        # 优先从 Redis 获取
        if self.redis_client:
            try:
                level = self.redis_client.get(f"user:{username}:level")
                if level:
                    return int(level)
            except:
                pass
        
        # 从数据库获取
        user = self.user_storage.get_user_by_username(username)
        if user:
            # 更新 Redis 缓存
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"user:{username}:level",
                        3600,
                        user['level']
                    )
                except:
                    pass
            return user['level']
        
        return None

    def _mask_key(self, key: str) -> str:
        """脱敏显示密钥"""
        if len(key) <= 12:
            return key[:4] + "*" * (len(key) - 4)
        return key[:8] + "*" * (len(key) - 12) + key[-4:]

# 全局认证管理器实例
auth_manager = AuthenticationManager()