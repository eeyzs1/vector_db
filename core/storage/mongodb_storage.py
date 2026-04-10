from pymongo import MongoClient
from typing import Dict, Any, List
from .metadata_storage import MetadataStorageInterface

class MongoDBStorage(MetadataStorageInterface):
    def __init__(self, host='localhost', port=27017, user='', password='', database='vector_db'):
        if user and password:
            uri = f"mongodb://{user}:{password}@{host}:{port}/{database}"
        else:
            uri = f"mongodb://{host}:{port}/{database}"
        
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.collection = self.db['metadata']
    
    def store_metadata(self, metadata: Dict[str, Any]) -> str:
        import uuid
        metadata_id = str(uuid.uuid4())
        metadata['_id'] = metadata_id
        result = self.collection.insert_one(metadata)
        return str(result.inserted_id)
    
    def get_metadata(self, metadata_id: str) -> Dict[str, Any]:
        result = self.collection.find_one({'_id': metadata_id})
        if result:
            # 转换ObjectId为字符串
            result['_id'] = str(result['_id'])
        return result
    
    def update_metadata(self, metadata_id: str, metadata: Dict[str, Any]) -> bool:
        result = self.collection.update_one(
            {'_id': metadata_id},
            {'$set': metadata}
        )
        return result.modified_count > 0
    
    def delete_metadata(self, metadata_id: str) -> bool:
        result = self.collection.delete_one({'_id': metadata_id})
        return result.deleted_count > 0
    
    def search_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for doc in self.collection.find(filters):
            doc['_id'] = str(doc['_id'])
            results.append(doc)
        return results