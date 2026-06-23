import os
import time
import uuid

from fastapi import FastAPI, HTTPException

from agents.core.settings import AppConfig
from agents.core.logging import configure_logging
from agents.services.intent_router import IntentRouter
from agents.services.runtime import AgentRuntime
from agents.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    Message,
)

logger = configure_logging()
config = AppConfig.from_env()
router = IntentRouter(config)
runtime = AgentRuntime(config=config, logger=logger)

app = FastAPI(
    title="RAG Agent API",
    version="2.0.0",
    description="OpenAI-compatible API for hybrid RAG + schema-tool routing through Ollama",
)


@app.on_event("startup")
def startup_event() -> None:
    """
    FastAPI startup lifecycle hook.

    Called once when the service process starts. Delegates to runtime warm-up.
    """
    runtime.warm_start()


@app.get("/healthz")
def healthz() -> dict:
    """
    Kubernetes liveness/readiness endpoint.

    Called by probes and load balancers to confirm process health.
    """
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict:
    """
    OpenAI-compatible model discovery endpoint.

    Called by OpenWebUI/client startup flows to list available models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": config.llm_display_name,
                "object": "model",
                "owned_by": "local",
                "permission": [],
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Main OpenAI-compatible chat endpoint.

    Called per user message. Lifecycle:
    1) parse latest user message
    2) classify intent
    3) execute routed runtime path
    4) return response in OpenAI schema
    """
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

    intent = router.route(user_query)
    logger.info("Chat routing request_id=%s intent=%s", request_id, intent)

    try:
        if intent == "doc":
            answer, sources = runtime.answer_doc_query(user_query)
        else:
            answer, sources = runtime.answer_chat_query(user_query)

        logger.info(
            "Chat request completed request_id=%s intent=%s answer_chars=%d sources=%d",
            request_id,
            intent,
            len(answer),
            len(sources),
        )
    except Exception:
        logger.exception("Chat request failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Agent execution failed")

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8091))
    uvicorn.run(app, host="0.0.0.0", port=port)
