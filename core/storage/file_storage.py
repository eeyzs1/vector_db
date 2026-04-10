from abc import ABC, abstractmethod
from typing import Optional

class FileStorageInterface(ABC):
    @abstractmethod
    def store_file(self, file_path: str, content: bytes) -> str:
        pass
    
    @abstractmethod
    def read_file(self, file_id: str) -> bytes:
        pass
    
    @abstractmethod
    def delete_file(self, file_id: str) -> bool:
        pass