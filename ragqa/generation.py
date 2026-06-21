"""Direct and retrieval-grounded response generation."""

from .types import Citation, RetrievedChunk


class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def direct(self, question: str) -> str:
        return self.llm.generate(
            [
                {
                    "role": "system",
                    "content": "Answer concisely and clearly from stable general knowledge.",
                },
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_new_tokens=512,
        ).strip()

    def grounded(
        self, question: str, retrieved: list[RetrievedChunk]
    ) -> tuple[str, list[Citation]]:
        context = "\n\n".join(
            f"[SOURCE_{i}] {item.chunk.title}\nURL: {item.chunk.url}\n{item.chunk.text}"
            for i, item in enumerate(retrieved, start=1)
        )
        answer = self.llm.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied web passages. Cite each factual claim "
                        "with [SOURCE_N]. If the evidence is insufficient, say so. Never invent "
                        "a citation or use outside knowledge. Be concise and complete."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Web passages:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.1,
            max_new_tokens=512,
        ).strip()

        citations = []
        for i, item in enumerate(retrieved, start=1):
            tag = f"[SOURCE_{i}]"
            if tag in answer:
                citations.append(
                    Citation(
                        tag=tag,
                        title=item.chunk.title,
                        url=item.chunk.url,
                        search_rank=item.chunk.search_rank,
                        excerpt=item.chunk.text[:240] + "…",
                        retrieval_score=round(item.score, 4),
                    )
                )
        return answer, citations
