from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    OpenAI-compatible chat message model.

    Used by request parsing and response formatting in HTTP handlers.
    """

    role: str = Field(..., description="Role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """
    Incoming request payload for POST /v1/chat/completions.

    Parsed by FastAPI before the endpoint function executes.
    """

    model: str = Field(default="qwen", description="Model identifier")
    messages: List[Message] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, description="Max tokens in response")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Not supported, always false")


class ChatCompletionChoice(BaseModel):
    """Single completion choice in OpenAI-compatible response format."""

    index: int
    message: Message
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    """Token-like usage counters returned to OpenAI-compatible clients."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """
    Final response schema returned by POST /v1/chat/completions.

    Built after orchestration selects a path and generates an answer.
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


