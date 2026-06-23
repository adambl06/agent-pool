from typing import Any, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LLM

from agents.core.settings import AppConfig


class QwenLLM(LLM):
    """
    LangChain-compatible LLM client for Ollama/OpenAI-style endpoints.

    Instantiated by build_llm during runtime warm-up or lazy initialization.
    """

    endpoint: str
    model_name: str = "qwen3"
    provider: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 1024
    request_timeout_seconds: float = 60.0

    @property
    def _llm_type(self) -> str:
        """Return framework identifier used by LangChain internals."""
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute a single text generation call.

        Called by LangChain chains when generating final answers or schema
        routing decisions.
        """

        provider = self.provider.lower()

        if provider == "ollama":
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            url = f"{self.endpoint.rstrip('/')}/api/chat"
            response = requests.post(url, json=payload, timeout=self.request_timeout_seconds)
            response.raise_for_status()
            return response.json()["message"]["content"]

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        url = f"{self.endpoint.rstrip('/')}/v1/chat/completions"
        response = requests.post(url, json=payload, timeout=self.request_timeout_seconds)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def build_llm(config: AppConfig) -> LLM:
    """
    Factory used by runtime orchestration to construct LLM clients.

    Called by AgentRuntime when creating RAG or general-chat pipelines.
    """
    if config.llm_type != "qwen3":
        raise ValueError(f"Unsupported LLM type: {config.llm_type}")

    return QwenLLM(
        endpoint=config.llm_endpoint,
        model_name=config.llm_model_name,
        provider=config.llm_provider,
        request_timeout_seconds=config.llm_timeout_seconds,
    )
