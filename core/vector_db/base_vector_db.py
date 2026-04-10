from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorDB(ABC):
    @abstractmethod
    def create_collection(self, collection_id: str, dimension: int) -> bool:
        pass
    
    @abstractmethod
    def insert(self, collection_id: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        pass
    
    @abstractmethod
    def search(self, collection_id: str, query_vector: List[float], top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def modify(self, collection_id: str, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def delete(self, collection_id: str, vector_ids: List[str]) -> bool:
        pass