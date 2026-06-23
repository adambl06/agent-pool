import logging
from typing import Any, List, Optional, Tuple

from langchain_classic.chains import RetrievalQA
from langchain_core.language_models import LLM

from agents.core.settings import AppConfig
from agents.services.llm_client import build_llm
from agents.services.tool_decision import decide_tool_call_with_schema
from agents.services.tool_models import QueryDocumentsArgs
from rag_tools.rag import ingest_documents, load_vector_store


class AgentRuntime:
    """
    Runtime orchestrator for answering requests after intent routing.

    Owns lazy-loaded expensive objects (RAG chain and general LLM) and is
    called from the HTTP layer for both doc and chat execution paths.
    """

    def __init__(self, config: AppConfig, logger: logging.Logger):
        """Create runtime container. Called once during application startup."""
        self.config = config
        self.logger = logger
        self._rag_chain: Optional[RetrievalQA] = None
        self._general_llm: Optional[LLM] = None

    def warm_start(self) -> None:
        """
        Startup hook invoked by FastAPI startup event.

        Optionally ingests documents and pre-warms lazy dependencies to reduce
        first-request latency.
        """
        self.logger.info("Startup begin INGEST_ON_STARTUP=%s", self.config.ingest_on_startup)
        if self.config.ingest_on_startup:
            ingest_documents()
        _ = self.rag_chain
        _ = self.general_llm
        self.logger.info("Startup complete")

    @property
    def rag_chain(self) -> RetrievalQA:
        """
        Lazily build and cache the RetrievalQA chain.

        First called on startup warm-up or the first doc-routed request.
        """
        if self._rag_chain is None:
            self.logger.info(
                "Creating RAG chain llm_type=%s provider=%s endpoint=%s model=%s timeout=%.1fs",
                self.config.llm_type,
                self.config.llm_provider,
                self.config.llm_endpoint,
                self.config.llm_model_name,
                self.config.llm_timeout_seconds,
            )
            llm = build_llm(self.config)
            self.logger.info("Loading vector store for RAG")
            vector_db = load_vector_store()
            retriever = vector_db.as_retriever(search_kwargs={"k": self.config.rag_top_k})
            self._rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=self.config.agent_verbose,
            )
            self.logger.info("RAG chain created and ready")
        return self._rag_chain

    @property
    def general_llm(self) -> LLM:
        """
        Lazily build and cache the direct-chat LLM.

        First called on startup warm-up or the first chat-routed request.
        """
        if self._general_llm is None:
            self.logger.info(
                "Creating general LLM llm_type=%s provider=%s endpoint=%s model=%s timeout=%.1fs",
                self.config.llm_type,
                self.config.llm_provider,
                self.config.llm_endpoint,
                self.config.llm_model_name,
                self.config.llm_timeout_seconds,
            )
            self._general_llm = build_llm(self.config)
        return self._general_llm

    def answer_doc_query(self, query: str) -> Tuple[str, List[Any]]:
        """
        Run deterministic retrieve-then-generate answer path.

        Called when router returns `doc` intent.
        """
        response = self.rag_chain.invoke({"query": query})
        return response.get("result", ""), response.get("source_documents", [])

    def answer_chat_query(self, query: str) -> Tuple[str, List[Any]]:
        """
        Run non-document chat path.

        Called when router returns `chat` intent. May still call document tool
        if schema decision explicitly requests it.
        """
        if not self.config.schema_tools_enabled:
            return self.general_llm.invoke(query), []

        try:
            decision = decide_tool_call_with_schema(self.general_llm, query)
            self.logger.info(
                "Schema tool decision use_tool=%s tool_name=%s reason=%s",
                decision.use_tool,
                decision.tool_name,
                decision.reason[:200],
            )
            if decision.use_tool and decision.tool_name == "query_documents":
                args = QueryDocumentsArgs.model_validate(decision.arguments)
                return self.answer_doc_query(args.query)
            return self.general_llm.invoke(query), []
        except Exception:
            self.logger.exception("Schema tool decision failed; falling back to direct chat")
            return self.general_llm.invoke(query), []
