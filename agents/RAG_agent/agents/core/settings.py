import os
from dataclasses import dataclass
from typing import Literal, Tuple


Intent = Literal["doc", "chat"]


@dataclass(frozen=True)
class AppConfig:
    """
    Immutable runtime settings loaded from environment variables.

    Created once at process start and shared by API, routing, and runtime
    orchestration components.
    """

    llm_type: str
    llm_endpoint: str
    llm_model_name: str
    llm_display_name: str
    llm_provider: str
    llm_timeout_seconds: float
    rag_top_k: int
    agent_verbose: bool
    ingest_on_startup: bool
    schema_tools_enabled: bool
    router_force_rag: bool
    router_doc_keywords: Tuple[str, ...]

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Build AppConfig from process environment.

        Called during app initialization before FastAPI starts serving traffic.
        """

        configured_keywords = os.getenv("ROUTER_DOC_KEYWORDS", "")
        if configured_keywords.strip():
            keywords = tuple(k.strip().lower() for k in configured_keywords.split(",") if k.strip())
        else:
            keywords = (
                "document",
                "docs",
                "pdf",
                "runbook",
                "procedure",
                "internal",
                "cnip",
                "deprovision",
                "command",
                "commands",
                "steps",
                "node",
                "platform",
            )

        return cls(
            llm_type=os.getenv("LLM_TYPE", "qwen3").lower(),
            llm_endpoint=os.getenv("LLM_ENDPOINT", "http://ollama.svc.cluster.local:11434"),
            llm_model_name=os.getenv("LLM_MODEL_NAME", "qwen3"),
            llm_display_name=os.getenv("LLM_DISPLAY_NAME", os.getenv("LLM_MODEL_NAME", "rag-agent")),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            agent_verbose=os.getenv("AGENT_VERBOSE", "false").lower() == "true",
            ingest_on_startup=os.getenv("INGEST_ON_STARTUP", "false").lower() == "true",
            schema_tools_enabled=os.getenv("ENABLE_SCHEMA_TOOLS", "false").lower() == "true",
            router_force_rag=os.getenv("ROUTER_FORCE_RAG", "false").lower() == "true",
            router_doc_keywords=keywords,
        )
