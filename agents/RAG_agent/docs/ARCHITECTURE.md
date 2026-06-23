# RAG Agent Architecture

## Goals

This service provides an OpenAI-compatible chat API backed by:

- A deterministic RAG path for internal knowledge queries.
- A direct chat path for general conversation.
- Optional schema-guided tool decision for controlled tool usage.

## High-Level Flow

1. Client sends `POST /v1/chat/completions`.
2. The service extracts the latest user message.
3. `IntentRouter` classifies intent as `doc` or `chat`.
4. `AgentRuntime` executes the selected path:
   - `doc`: vector retrieval from Qdrant + LLM answer synthesis.
   - `chat`: direct LLM answer, optionally with schema-tool decision first.
5. Service returns OpenAI-compatible response JSON.

## Modules and Responsibilities

### `agents/agent.py`

- Thin compatibility entrypoint so deployment command `uvicorn agents.agent:app` stays unchanged.

### `agents/api/app.py`

- FastAPI app creation and endpoint handlers.
- Request logging, intent dispatch, and OpenAI-compatible response mapping.

### `agents/core/settings.py`

- Typed environment configuration (`AppConfig`).
- Intent typing contract.

### `agents/api/schemas.py`

- Pydantic schemas for HTTP request/response contracts.

### `agents/services/tool_models.py`

- Pydantic models for tool decision payloads and tool arguments.

### `agents/services/intent_router.py`

- Query intent router (`doc` vs `chat`).
- Metadata-prompt guardrails for OpenWebUI helper tasks.

### `agents/services/runtime.py`

- Runtime orchestration (`AgentRuntime`).
- Lazy singleton lifecycle for RAG chain and general LLM.
- Execution paths for doc queries and chat queries.

### `agents/services/llm_client.py`

- Provider adapter (`QwenLLM`) for Ollama/OpenAI-compatible endpoints.

### `agents/services/tool_decision.py`

- Schema-guided tool-call decision helpers.
- JSON extraction and strict validation of model output.

### `agents/core/logging.py`

- Central logging initialization.

### `rag_tools/rag.py`

- Document loading and parsing.
- Text chunking strategy.
- Embedding and Qdrant upsert.
- Retrieval chain creation.

## Routing Policy

`doc` route is selected when:

- `ROUTER_FORCE_RAG=true`, or
- query contains configured doc keywords.

`chat` route is selected for:

- OpenWebUI metadata/helper prompts (for example `### Task:`), or
- non-document/general requests.

## Design Principles Applied

- Separation of concerns: config, routing, runtime, transport split by class/function.
- Deterministic retrieval for internal knowledge correctness.
- Safe fallback behavior when schema-tool parsing fails.
- Structured logging for each major stage.
- Stable singleton lifecycle for expensive components.

## API Compatibility

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

All are OpenAI-compatible for OpenWebUI and similar clients.
