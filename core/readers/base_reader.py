from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDocumentReader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_type = self._get_file_type()
        self.metadata = {}
    
    def _get_file_type(self) -> str:
        return self.file_path.split('.')[-1].lower()
    
    @abstractmethod
    def read(self) -> Any:
        pass
    
    @abstractmethod
    def extract_metadata(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_content(self) -> Any:
        pass