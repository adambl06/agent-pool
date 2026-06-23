import json

from langchain_core.language_models import LLM

from agents.services.tool_models import ToolDecision


def _extract_json_block(text: str) -> str:
    """
    Extract the outermost JSON payload from model output.

    Called by decide_tool_call_with_schema to tolerate models that add prose
    around the expected JSON object.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def decide_tool_call_with_schema(llm: LLM, user_query: str) -> ToolDecision:
    """
    Ask the model for a strict tool decision and validate the response.

    Called only on the chat path when schema-tools are enabled.
    Returns a validated ToolDecision used by runtime orchestration.
    """
    prompt = (
        "You are a routing assistant. Decide whether to call the tool query_documents.\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "use_tool": boolean,\n'
        '  "tool_name": "query_documents" | "none",\n'
        '  "arguments": {"query": string, "top_k": integer},\n'
        '  "reason": string\n'
        "}\n"
        "Rules:\n"
        "- Use query_documents for internal docs/runbooks/procedures/CNIP commands.\n"
        "- Use none for general chit-chat or broad knowledge not requiring internal docs.\n"
        "- If use_tool is false, tool_name must be none and arguments can be {}.\n"
        f"User query: {user_query}"
    )
    raw = llm.invoke(prompt)
    payload = json.loads(_extract_json_block(raw))
    decision = ToolDecision.model_validate(payload)

    if decision.use_tool and decision.tool_name != "query_documents":
        raise ValueError("tool_name must be query_documents when use_tool=true")
    if not decision.use_tool and decision.tool_name != "none":
        raise ValueError("tool_name must be none when use_tool=false")
    return decision
