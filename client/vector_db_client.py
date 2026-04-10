"""向量数据库 Python 客户端 SDK"""
import requests
from typing import List, Dict, Any, Optional
from config.config import config


class VectorDBClient:
    """向量数据库客户端"""

    def __init__(self, base_url: str = None, username: str = None, api_key: Optional[str] = None, jwt_token: Optional[str] = None):
        """初始化客户端

        Args:
            base_url: 服务器地址，默认从配置文件读取
            username: 用户名（必需，用于新的认证系统）
            api_key: API Key 认证
            jwt_token: JWT Token 认证
        """
        self.base_url = (base_url or config.get('BASE_URL', 'http://localhost:8000')).rstrip('/')
        self.username = username
        self.headers = {}

        if api_key:
            if not username:
                raise ValueError("Username is required when using API Key authentication")
            self.headers['api-key'] = api_key
        elif jwt_token:
            self.headers['Authorization'] = f'Bearer {jwt_token}'

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送 HTTP 请求"""
        url = f"{self.base_url}{endpoint}"
        headers = self.headers.copy()
        
        # 确保同时提供用户名和API Key
        if self.username:
            headers['username'] = self.username
        
        kwargs.setdefault('headers', {}).update(headers)

        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    # ========== 认证管理 ==========

    def create_api_key(self, username: str, level: int = 0) -> Dict[str, Any]:
        """生成新的 API Key
        
        Args:
            username: 用户名
            level: 用户级别
            
        Returns:
            包含api_key、username和level的字典
        """
        result = self._request('POST', '/api/auth/apikey/create', json={'username': username, 'level': level})
        return result

    def list_api_keys(self) -> List[str]:
        """列出所有 API Key（脱敏）"""
        result = self._request('GET', '/api/auth/apikey/list')
        return result['api_keys']

    def delete_api_key(self, api_key: str) -> bool:
        """删除指定 API Key"""
        result = self._request('DELETE', '/api/auth/apikey/delete', json={'api_key': api_key})
        return result['status'] == 'success'

    # ========== 文本处理 ==========

    def process_text(self, text: str) -> Dict[str, Any]:
        """处理文本：分块 + 嵌入

        Args:
            text: 文本内容

        Returns:
            {'embedding': [...], 'chunks': [...]}
        """
        return self._request('POST', '/api/process/text', json={'text': text})

    # ========== 向量操作 ==========

    def insert_vectors(self, collection_id: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        """插入向量

        Args:
            collection_id: 集合 ID
            vectors: 向量列表
            metadata: 元数据列表

        Returns:
            向量 ID 列表
        """
        result = self._request('POST', '/api/vector/insert', json={
            'collection_id': collection_id,
            'vectors': vectors,
            'metadata': metadata
        })
        return result['vector_ids']

    def search_vectors(self, collection_id: str, query_vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """向量相似度搜索

        Args:
            collection_id: 集合 ID
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        result = self._request('POST', '/api/vector/search', json={
            'collection_id': collection_id,
            'query_vector': query_vector,
            'top_k': top_k,
            'filters': filters
        })
        return result['results']

    # ========== 文件操作 ==========

    def upload_file(self, file_path: str, collection_id: str) -> Dict[str, Any]:
        """上传文件并处理

        Args:
            file_path: 本地文件路径
            collection_id: 集合 ID

        Returns:
            {'file_id': ..., 'collection_id': ..., 'chunks_processed': ..., 'vectors_stored': ...}
        """
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'collection_id': collection_id}
            return self._request('POST', '/api/files/upload', files=files, data=data)

    def batch_upload_files(self, file_paths: List[str], collection_id: str) -> List[Dict[str, Any]]:
        """批量上传文件

        Args:
            file_paths: 本地文件路径列表
            collection_id: 集合 ID

        Returns:
            处理结果列表
        """
        files = [('files', open(fp, 'rb')) for fp in file_paths]
        try:
            data = {'collection_id': collection_id}
            result = self._request('POST', '/api/files/batch/upload', files=files, data=data)
            return result['results']
        finally:
            for _, f in files:
                f.close()

    # ========== 元数据操作 ==========

    def get_metadata(self, file_id: str) -> Dict[str, Any]:
        """获取文件元数据"""
        return self._request('GET', f'/api/metadata/{file_id}')

    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> str:
        """更新元数据"""
        result = self._request('POST', '/api/metadata/update', json={
            'file_id': file_id,
            'metadata': metadata
        })
        return result['metadata_id']

    def delete_metadata(self, file_id: str) -> bool:
        """删除元数据"""
        result = self._request('DELETE', f'/api/metadata/{file_id}')
        return result['status'] == 'success'

    # ========== 系统管理 ==========

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return self._request('GET', '/api/system/status')

    def switch_model(self, model_type: str, model_name: str) -> str:
        """切换模型"""
        result = self._request('POST', '/api/model/switch', json={
            'model_type': model_type,
            'model_name': model_name
        })
        return result['model_path']

    # ========== 备份管理 ==========

    def create_backup(self) -> str:
        """创建备份"""
        result = self._request('POST', '/api/backup/create')
        return result['backup_file']

    def list_backups(self) -> List[str]:
        """列出所有备份"""
        result = self._request('GET', '/api/backup/list')
        return result['backups']

    def restore_backup(self, backup_file: str) -> bool:
        """恢复备份"""
        result = self._request('POST', f'/api/backup/restore/{backup_file}')
        return result['status'] == 'success'

    def delete_backup(self, backup_file: str) -> bool:
        """删除备份"""
        result = self._request('DELETE', f'/api/backup/delete/{backup_file}')
        return result['status'] == 'success'