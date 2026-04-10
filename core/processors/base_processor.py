from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BaseProcessor(ABC):
    @abstractmethod
    def chunk(self, content: Any) -> List[Any]:
        pass
    
    @abstractmethod
    def clean(self, content: Any) -> Any:
        pass
    
    @abstractmethod
    def embed(self, content: Any) -> List[float]:
        pass