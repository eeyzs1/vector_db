from typing import Dict, Any
from .llm_interface import LLMInterface


class LocalLLM(LLMInterface):
    def __init__(self, model_name: str, model_path: str = None):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.params = {
            'temperature': 0.7,
            'max_new_tokens': 1000,
            'top_p': 0.9,
            'do_sample': True,
        }
        self._pipeline = None

    def _load_model(self):
        """Lazy-load the transformers text-generation pipeline."""
        from transformers import pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_path,
            device_map="auto",
        )

    def generate(self, prompt: str, params: Dict[str, Any] = None) -> str:
        if self._pipeline is None:
            self._load_model()

        merged = {**self.params, **(params or {})}
        allowed = {'temperature', 'max_new_tokens', 'top_p', 'do_sample', 'repetition_penalty'}
        gen_params = {k: v for k, v in merged.items() if k in allowed}

        outputs = self._pipeline(prompt, **gen_params)
        generated = outputs[0]['generated_text']
        # Strip the original prompt prefix from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params.update(params)