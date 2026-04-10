from abc import ABC, abstractmethod
from typing import Dict, Any, List

class MetadataStorageInterface(ABC):
    @abstractmethod
    def store_metadata(self, metadata: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def get_metadata(self, metadata_id: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def update_metadata(self, metadata_id: str, metadata: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def delete_metadata(self, metadata_id: str) -> bool:
        pass
    
    @abstractmethod
    def search_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass