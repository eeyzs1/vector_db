import json
import redis
from typing import Dict, Any, List
from .metadata_storage import MetadataStorageInterface

class RedisStorage(MetadataStorageInterface):
    def __init__(self, host='localhost', port=6379, password='', db=0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
    
    def store_metadata(self, metadata: Dict[str, Any]) -> str:
        import uuid
        metadata_id = str(uuid.uuid4())
        key = f"metadata:{metadata_id}"
        self.redis_client.set(key, json.dumps(metadata))
        return metadata_id
    
    def get_metadata(self, metadata_id: str) -> Dict[str, Any]:
        key = f"metadata:{metadata_id}"
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def update_metadata(self, metadata_id: str, metadata: Dict[str, Any]) -> bool:
        key = f"metadata:{metadata_id}"
        existing_data = self.redis_client.get(key)
        if existing_data:
            existing_metadata = json.loads(existing_data)
            existing_metadata.update(metadata)
            self.redis_client.set(key, json.dumps(existing_metadata))
            return True
        return False
    
    def delete_metadata(self, metadata_id: str) -> bool:
        key = f"metadata:{metadata_id}"
        result = self.redis_client.delete(key)
        return result > 0
    
    def search_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        # 简单实现：遍历所有元数据并过滤
        for key in self.redis_client.keys("metadata:*"):
            data = self.redis_client.get(key)
            if data:
                metadata = json.loads(data)
                match = True
                for k, v in filters.items():
                    if metadata.get(k) != v:
                        match = False
                        break
                if match:
                    results.append(metadata)
        return results