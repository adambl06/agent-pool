from agents.core.settings import AppConfig, Intent


class IntentRouter:
    """
    Stateless query intent classifier for request routing.

    Called once per chat request by the HTTP endpoint before dispatching to
    runtime execution paths.
    """

    META_PROMPT_PREFIXES = (
        "### task:",
        "suggest 3-5 relevant follow-up questions",
        "generate a concise, 3-5 word title",
        "generate 1-3 broad tags",
    )

    def __init__(self, config: AppConfig):
        """Store router configuration and keyword policy."""
        self._config = config

    def route(self, query: str) -> Intent:
        """
        Return `doc` when retrieval grounding is likely required, else `chat`.

        Evaluation order:
        1) metadata prompt guardrail
        2) force-rag switch
        3) keyword match
        """

        lowered = query.lower().strip()

        if any(lowered.startswith(prefix) for prefix in self.META_PROMPT_PREFIXES):
            return "chat"

        if self._config.router_force_rag:
            return "doc"

        if any(keyword in lowered for keyword in self._config.router_doc_keywords):
            return "doc"
        return "chat"
