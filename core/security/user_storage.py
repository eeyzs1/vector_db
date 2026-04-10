"""用户和API Key存储"""
import sqlite3
import os
from typing import Optional, Dict, Any

class UserStorage:
    """用户和API Key存储"""
    
    def __init__(self, db_path: str = "./data/users.db"):
        """初始化存储
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            api_key TEXT NOT NULL,
            level INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建API Key索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_key ON users(api_key)')
        
        conn.commit()
        conn.close()
    
    def add_user(self, username: str, api_key: str, level: int = 0) -> bool:
        """添加用户
        
        Args:
            username: 用户名
            api_key: API Key
            level: 用户级别
            
        Returns:
            是否添加成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO users (username, api_key, level) VALUES (?, ?, ?)",
                (username, api_key, level)
            )
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # 用户名已存在
            return False
    
    def update_user(self, username: str, api_key: Optional[str] = None, level: Optional[int] = None) -> bool:
        """更新用户信息
        
        Args:
            username: 用户名
            api_key: API Key（可选）
            level: 用户级别（可选）
            
        Returns:
            是否更新成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if api_key is not None:
                updates.append("api_key = ?")
                params.append(api_key)
            
            if level is not None:
                updates.append("level = ?")
                params.append(level)
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                sql = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"
                params.append(username)
                
                cursor.execute(sql, params)
                conn.commit()
                
            conn.close()
            return True
        except:
            return False
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """根据用户名获取用户信息
        
        Args:
            username: 用户名
            
        Returns:
            用户信息
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT username, api_key, level, created_at, updated_at FROM users WHERE username = ?",
            (username,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_user_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """根据API Key获取用户信息
        
        Args:
            api_key: API Key
            
        Returns:
            用户信息
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT username, api_key, level, created_at, updated_at FROM users WHERE api_key = ?",
            (api_key,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def delete_user(self, username: str) -> bool:
        """删除用户
        
        Args:
            username: 用户名
            
        Returns:
            是否删除成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def list_users(self) -> list:
        """列出所有用户
        
        Returns:
            用户列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT username, api_key, level, created_at, updated_at FROM users")
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]