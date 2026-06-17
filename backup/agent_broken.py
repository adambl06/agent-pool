# agent.py
import os
import yaml
import logging
import time
import uuid
from typing import Any, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.language_models import LLM
from langchain_classic.chains import RetrievalQA
from rag_tools.rag import ingest_documents


# Set up logging (stdout by default so kubectl logs shows application events)
log_level = logging.DEBUG if os.environ.get("LOG_LEVEL") == "DEBUG" else logging.INFO
log_file = os.environ.get("LOG_FILE")
if log_file:
    logging.basicConfig(
        level=log_level,
        filename=log_file,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
else:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger = logging.getLogger(__name__)


# Shared singleton for API mode.
_AGENT: Optional[Any] = None


# OpenAI-compatible models
class Message(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen", description="Model identifier")
    messages: List[Message] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, description="Max tokens in response")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Not supported, always false")


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


# --- Custom Qwen LLM (calls your self-hosted Kubernetes endpoint) ---
class QwenLLM(LLM):
    """
    Custom LangChain LLM wrapper for a self-hosted Qwen model.
    Expects an OpenAI-compatible HTTP API (e.g. served via vLLM).
    """
    endpoint: str
    model_name: str = "qwen3"
    provider: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 1024
    request_timeout_seconds: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        provider = self.provider.lower()
        logger.info(
            "LLM call started provider=%s endpoint=%s model=%s timeout=%.1fs prompt_chars=%d",
            provider,
            self.endpoint,
            self.model_name,
            self.request_timeout_seconds,
            len(prompt),
        )

        if provider == "ollama":
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            }
            url = f"{self.endpoint.rstrip('/')}/api/chat"
            response = requests.post(url, json=payload, timeout=self.request_timeout_seconds)
            if response.status_code >= 400:
                logger.error(
                    "Ollama request failed status=%s url=%s body=%s",
                    response.status_code,
                    url,
                    response.text[:500],
                )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            logger.info("LLM call finished provider=ollama response_chars=%d", len(content))
            return content

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
        if response.status_code >= 400:
            logger.error(
                "OpenAI-compatible request failed status=%s url=%s body=%s",
                response.status_code,
                url,
                response.text[:500],
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        logger.info("LLM call finished provider=openai-compatible response_chars=%d", len(content))
        return content


# --- LLM Factory ---
def get_llm(
    llm_type: str,
    llm_endpoint: str,
    model_name: str = "qwen3",
    llm_provider: str = "ollama",
    request_timeout_seconds: float = 60.0,
) -> LLM:
    llm_type = llm_type.lower()
    if llm_type == "qwen3":
        return QwenLLM(
            endpoint=llm_endpoint,
            model_name=model_name,
            provider=llm_provider,
            request_timeout_seconds=request_timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")


# --- Initialize RAG Tool ---
# --- Create Agent (RetrievalQA Chain) ---
def create_agent() -> Any:
    llm_endpoint = os.getenv("LLM_ENDPOINT", "http://ollama.svc.cluster.local:11434")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen3")
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    logger.info(
        "Creating agent llm_type=%s provider=%s endpoint=%s model=%s timeout=%.1fs",
        os.getenv("LLM_TYPE", "qwen3"),
        llm_provider,
        llm_endpoint,
        llm_model_name,
        llm_timeout_seconds,
    )
    llm = get_llm(
        llm_type=os.getenv("LLM_TYPE", "qwen3"),
        llm_endpoint=llm_endpoint,
        model_name=llm_model_name,
        llm_provider=llm_provider,
        request_timeout_seconds=llm_timeout_seconds,
    )
    # Load vector store and create RetrievalQA chain
    from rag_tools.rag import load_vector_store
    logger.info("Loading vector store for RAG")
    vector_db = load_vector_store()
    k = int(os.getenv("RAG_TOP_K", "4"))
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    
    # RetrievalQA always retrieves before answering
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=os.getenv("AGENT_VERBOSE", "false").lower() == "true",
    )
    logger.info("RAG chain created and ready")
    return rag_chain


def get_or_create_agent() -> Any:
    global _AGENT
    if _AGENT is None:
        _AGENT = create_agent()
    return _AGENT


app = FastAPI(
    title="RAG Agent API",
    version="1.0.0",
    description="OpenAI-compatible API for RAG agent with Qwen model through Ollama",
)


@app.on_event("startup")
def startup_event() -> None:
    # Optional: set INGEST_ON_STARTUP=true to rebuild the index when the pod starts.
    logger.info("Startup begin INGEST_ON_STARTUP=%s", os.getenv("INGEST_ON_STARTUP", "false"))
    if os.getenv("INGEST_ON_STARTUP", "false").lower() == "true":
        ingest_documents()
    get_or_create_agent()
    logger.info("Startup complete")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict:
    display_name = os.getenv("LLM_DISPLAY_NAME", os.getenv("LLM_MODEL_NAME", "rag-agent"))
    return {
        "object": "list",
        "data": [
            {
                "id": display_name,
                "object": "model",
                "owned_by": "local",
                "permission": [],
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint.
    Extracts the user's message and runs the RAG agent.
    """
    # Extract the user query (last user message)
    request_id = str(uuid.uuid4())[:8]
    user_query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content.strip()
            break

    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    logger.info(
        "Chat request received request_id=%s model=%s messages=%d query=%s",
        request_id,
        request.model,
        len(request.messages),
        user_query[:200],
    )

    agent = get_or_create_agent()
    try:
        answer = agent.run(user_query)
    except Exception:
        logger.exception("Chat request failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Agent execution failed")
    logger.info("Chat request completed request_id=%s answer_chars=%d", request_id, len(answer))
        agent = get_or_create_agent()
        try:
            # RetrievalQA returns dict with 'result' and 'source_documents'
            response = agent({"query": user_query})
            answer = response.get("result", "")
            sources = response.get("source_documents", [])
            logger.info("Chat request completed request_id=%s answer_chars=%d sources=%d", request_id, len(answer), len(sources))
        except Exception:
            logger.exception("Chat request failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail="Agent execution failed")
        logger.info(
            "Chat request received request_id=%s model=%s messages=%d query=%s",
            request_id,
            request.model,
            len(request.messages),
            user_query[:200],
        )

        agent = get_or_create_agent()
        try:
            # RetrievalQA returns dict with 'result' and 'source_documents'
            response = agent({"query": user_query})
            answer = response.get("result", "")
            sources = response.get("source_documents", [])
            logger.info("Chat request completed request_id=%s answer_chars=%d sources=%d", request_id, len(answer), len(sources))
        except Exception:
            logger.exception("Chat request failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail="Agent execution failed")
        logger.info(
            "Chat request received request_id=%s model=%s messages=%d query=%s",
            request_id,
            request.model,
            len(request.messages),
            user_query[:200],
        )

        agent = get_or_create_agent()
        try:
            # RetrievalQA returns dict with 'result' and 'source_documents'
            response = agent({"query": user_query})
            answer = response.get("result", "")
            sources = response.get("source_documents", [])
            logger.info("Chat request completed request_id=%s answer_chars=%d sources=%d", request_id, len(answer), len(sources))
        except Exception:
            logger.exception("Chat request failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail="Agent execution failed")

    # Build response in OpenAI format
    completion_id = f"chatcmpl-{int(time.time() * 1000)}"
    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=len(user_query.split()),
            completion_tokens=len(answer.split()),
            total_tokens=len(user_query.split()) + len(answer.split()),
        ),
    )


# --- Main ---
if __name__ == "__main__":
    import uvicorn
    # Run FastAPI server
    port = int(os.getenv("PORT", 8091))
    uvicorn.run(app, host="0.0.0.0", port=port)
