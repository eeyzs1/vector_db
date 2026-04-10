import requests
from typing import Dict, Any
from .llm_interface import LLMInterface


class RemoteLLM(LLMInterface):
    def __init__(self, model_name: str, api_key: str, api_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url or "https://api.openai.com/v1/chat/completions"
        self.params = {
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 0.9,
        }

    def generate(self, prompt: str, params: Dict[str, Any] = None) -> str:
        merged = {**self.params, **(params or {})}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **merged,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Remote LLM API error {response.status_code}: {response.text}"
            ) from e
        return response.json()["choices"][0]["message"]["content"]

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params.update(params)