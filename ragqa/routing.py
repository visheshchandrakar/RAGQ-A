"""LLM-based direct-versus-web routing."""

from .llm import parse_json
from .types import RouteDecision


class QueryRouter:
    def __init__(self, llm):
        self.llm = llm

    def decide(self, question: str) -> RouteDecision:
        prompt = (
            "You route questions for an assistant with a small local model. Choose WEB when "
            "the answer needs current, changing, niche, precise, or externally verifiable facts. "
            "Choose DIRECT for casual conversation, writing help, reasoning, or stable general "
            "knowledge that the model can reliably answer. For WEB, rewrite the user's question "
            "as a concise search-engine query.\n\n"
            f"Question: {question}\n\n"
            'Return JSON only: {"route":"direct|web","search_query":"...",'
            '"reason":"one short sentence"}'
        )
        response = self.llm.generate(
            [{"role": "user", "content": prompt}], max_new_tokens=160
        )
        try:
            data = parse_json(response)
            route = str(data.get("route", "")).strip().lower()
            if route not in {"direct", "web"}:
                raise ValueError("unknown route")
            query = str(data.get("search_query", "")).strip()
            if route == "web" and not query:
                query = question
            return RouteDecision(route, query, str(data.get("reason", "")).strip())
        except Exception:
            return RouteDecision(
                "web",
                question,
                "The router response was invalid, so web search was selected safely.",
            )
