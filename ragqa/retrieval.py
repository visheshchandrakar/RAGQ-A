"""Token-aware chunking and request-local FAISS retrieval."""

import faiss
import numpy as np

from .config import PipelineConfig
from .types import RetrievedChunk, WebChunk, WebPage, WebRAGError


class TemporaryFaissRetriever:
    def __init__(self, embedder, tokenizer, config: PipelineConfig):
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.config = config

    def chunk_pages(self, pages: list[WebPage]) -> list[WebChunk]:
        chunks: list[WebChunk] = []
        step = self.config.chunk_size - self.config.chunk_overlap
        for page in pages:
            tokens = self.tokenizer.encode(page.text)
            for start in range(0, len(tokens), step):
                token_slice = tokens[start : start + self.config.chunk_size]
                text = self.tokenizer.decode(token_slice).strip()
                if len(text) < 80:
                    continue
                chunks.append(
                    WebChunk(
                        text=text,
                        title=page.result.title,
                        url=page.result.url,
                        search_rank=page.result.rank,
                        chunk_id=len(chunks),
                        token_count=len(token_slice),
                    )
                )
        if not chunks:
            raise WebRAGError("Fetched pages produced no usable text chunks.")
        return chunks

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        faiss.normalize_L2(matrix)
        return matrix

    def retrieve(self, question: str, chunks: list[WebChunk]) -> list[RetrievedChunk]:
        matrix = self.embed([chunk.text for chunk in chunks])
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        query_vector = self.embed([question])
        k = min(len(chunks), max(self.config.max_context_chunks * 4, 12))
        scores, indices = index.search(query_vector, k)
        candidates = [
            RetrievedChunk(chunks[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

        selected: list[RetrievedChunk] = []
        token_total = 0
        per_source: dict[str, int] = {}

        def add_candidates(require_new_source: bool) -> None:
            nonlocal token_total
            for candidate in candidates:
                if candidate in selected:
                    continue
                count = per_source.get(candidate.chunk.url, 0)
                if require_new_source and count:
                    continue
                if count >= self.config.max_chunks_per_source:
                    continue
                if token_total + candidate.chunk.token_count > self.config.max_context_tokens:
                    continue
                selected.append(candidate)
                token_total += candidate.chunk.token_count
                per_source[candidate.chunk.url] = count + 1
                if len(selected) >= self.config.max_context_chunks:
                    return

        add_candidates(require_new_source=True)
        if len(selected) < self.config.max_context_chunks:
            add_candidates(require_new_source=False)
        if not selected:
            raise WebRAGError("No retrieved chunks fit within the model context budget.")
        return selected
