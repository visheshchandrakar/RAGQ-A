"""Lightweight ARES-style quality evaluation for grounded web answers."""

from .llm import parse_json
from .types import ARESScore, RetrievedChunk


class ARESEvaluator:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _score(value) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.5

    def evaluate(
        self, question: str, answer: str, retrieved: list[RetrievedChunk]
    ) -> ARESScore:
        context = "\n\n".join(
            f"[{index}] {item.chunk.text}"
            for index, item in enumerate(retrieved, start=1)
        )
        prompt = (
            "You are an ARES-style evaluator for a retrieval-augmented answer. "
            "Score each dimension from 0.0 to 1.0.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Definitions:\n"
            "- context_relevance: the retrieved passages help answer the question.\n"
            "- faithfulness: factual claims in the answer are supported by the passages.\n"
            "- answer_relevance: the answer directly addresses the question.\n\n"
            "Return JSON only with context_relevance, faithfulness, answer_relevance, "
            "and a reasoning object containing context, faithfulness, and relevance."
        )
        response = self.llm.generate(
            [{"role": "user", "content": prompt}], max_new_tokens=320
        )
        try:
            data = parse_json(response)
            context_relevance = self._score(data.get("context_relevance"))
            faithfulness = self._score(data.get("faithfulness"))
            answer_relevance = self._score(data.get("answer_relevance"))
            details = data.get("reasoning") or {}
        except Exception:
            context_relevance = faithfulness = answer_relevance = 0.5
            details = {"evaluation": "The evaluator returned an invalid response."}

        return ARESScore(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_relevance=context_relevance,
            overall=round(
                (faithfulness + answer_relevance + context_relevance) / 3, 4
            ),
            details=details,
        )
