"""High-level orchestration for the direct-or-web RAG pipeline."""

from __future__ import annotations

import os
import time
from typing import Callable

import tiktoken
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_CONFIG, PipelineConfig
from .generation import AnswerGenerator
from .llm import LocalQwen3
from .retrieval import TemporaryFaissRetriever
from .routing import QueryRouter
from .types import (
    RAGAnswer,
    RetrievedChunk,
    SearchResult,
    WebChunk,
    WebPage,
)
from .web import SerpApiSearch, WebPageFetcher


class WebRAGEngine:
    """Coordinate routing, web ingestion, retrieval, and answer generation."""

    EMBED_MODEL = DEFAULT_CONFIG.embed_model
    GEN_MODEL = DEFAULT_CONFIG.gen_model
    SEARCH_RESULT_LIMIT = DEFAULT_CONFIG.search_result_limit
    SEARCH_TIMEOUT_SECONDS = DEFAULT_CONFIG.search_timeout_seconds
    FETCH_TIMEOUT_SECONDS = DEFAULT_CONFIG.fetch_timeout_seconds
    MAX_RESPONSE_BYTES = DEFAULT_CONFIG.max_response_bytes
    CHUNK_SIZE = DEFAULT_CONFIG.chunk_size
    CHUNK_OVERLAP = DEFAULT_CONFIG.chunk_overlap
    MAX_CONTEXT_TOKENS = DEFAULT_CONFIG.max_context_tokens
    MAX_CONTEXT_CHUNKS = DEFAULT_CONFIG.max_context_chunks
    MAX_CHUNKS_PER_SOURCE = DEFAULT_CONFIG.max_chunks_per_source

    def __init__(
        self,
        model_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        *,
        llm=None,
        embedder=None,
        serpapi_key: str | None = None,
        search_provider: Callable[[str, str], list[SearchResult]] | None = None,
        page_fetcher: Callable[[SearchResult], WebPage] | None = None,
        config: PipelineConfig = DEFAULT_CONFIG,
    ):
        self.config = config
        self.client = llm or LocalQwen3(model_id, progress_callback)
        if progress_callback:
            progress_callback(1.0, "Loading local embedding model…")
        self.embedder = embedder or SentenceTransformer(config.embed_model)
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.serpapi_key = (
            serpapi_key if serpapi_key is not None else os.getenv("SERPAPI_KEY", "")
        )

        self.router = QueryRouter(self.client)
        self.search = SerpApiSearch(self.serpapi_key, config, search_provider)
        self.fetcher = WebPageFetcher(config, page_fetcher)
        self.retriever = TemporaryFaissRetriever(self.embedder, self.enc, config)
        self.generator = AnswerGenerator(self.client)

    def decide_route(self, question: str):
        return self.router.decide(question)

    # Thin delegates retain the useful testing/extension seams of the original API.
    def _search(self, query: str) -> list[SearchResult]:
        return self.search.search(query)

    def _fetch_page(self, result: SearchResult) -> WebPage:
        return self.fetcher.fetch(result)

    def _fetch_all(self, results: list[SearchResult]) -> list[WebPage]:
        return self.fetcher.fetch_all(results)

    def _chunk_pages(self, pages: list[WebPage]) -> list[WebChunk]:
        return self.retriever.chunk_pages(pages)

    def _embed(self, texts: list[str]):
        return self.retriever.embed(texts)

    def _retrieve(self, question: str, chunks: list[WebChunk]) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, chunks)

    def _generate_web_answer(self, question: str, retrieved: list[RetrievedChunk]):
        return self.generator.grounded(question, retrieved)

    def answer(
        self,
        question: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        started = time.time()

        def report(value: float, message: str) -> None:
            if progress_callback:
                progress_callback(value, message)

        report(0.05, "Choosing between a direct answer and web research…")
        decision = self.router.decide(question)
        if decision.route == "direct":
            report(0.45, "Direct route selected — generating locally…")
            answer = self.generator.direct(question)
            report(1.0, "Direct answer complete ✓")
            return RAGAnswer(
                question=question,
                answer=answer,
                route="direct",
                route_reason=decision.reason,
                search_query="",
                citations=[],
                indexed_source_count=0,
                indexed_chunk_count=0,
                latency_ms=round((time.time() - started) * 1000, 1),
            )

        report(0.18, f'Searching the web for “{decision.search_query}”…')
        results = self.search.search(decision.search_query)
        report(0.32, f"Fetching {len(results)} selected result pages…")
        pages = self.fetcher.fetch_all(results)
        report(0.52, "Extracting, chunking, and embedding page content…")
        chunks = self.retriever.chunk_pages(pages)
        report(0.70, "Searching the temporary FAISS index…")
        retrieved = self.retriever.retrieve(question, chunks)
        report(0.84, "Generating a grounded answer with citations…")
        answer, citations = self.generator.grounded(question, retrieved)
        result = RAGAnswer(
            question=question,
            answer=answer,
            route="web",
            route_reason=decision.reason,
            search_query=decision.search_query,
            citations=citations,
            indexed_source_count=len(pages),
            indexed_chunk_count=len(chunks),
            latency_ms=round((time.time() - started) * 1000, 1),
        )
        # Full pages, chunks, embeddings, and FAISS are request-local. Only compact
        # citation excerpts remain reachable after this method returns.
        report(1.0, "Web RAG pipeline complete ✓")
        return result
