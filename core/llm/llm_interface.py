from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, Any] = None) -> str:
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        pass