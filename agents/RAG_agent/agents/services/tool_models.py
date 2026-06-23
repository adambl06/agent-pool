from typing import Any, Dict

from pydantic import BaseModel, Field


class QueryDocumentsArgs(BaseModel):
    """
    Validated arguments for the query_documents tool.

    Used after the schema tool-decision step decides that document retrieval is
    required during request processing.
    """

    query: str
    top_k: int = Field(default=4, ge=1, le=20)


class ToolDecision(BaseModel):
    """
    Structured model decision for tool usage.

    Produced by decide_tool_call_with_schema during chat-path execution when
    ENABLE_SCHEMA_TOOLS=true.
    """

    use_tool: bool
    tool_name: str = "none"
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
