"""High-level orchestration for the direct-or-web RAG pipeline."""

from __future__ import annotations

import os
import time
from typing import Callable

import tiktoken
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_CONFIG, PipelineConfig
from .evaluation import ARESEvaluator
from .generation import AnswerGenerator
from .llm import LocalQwen3
from .retrieval import TemporaryFaissRetriever
from .routing import QueryRouter
from .types import (
    PipelineEvent,
    RAGAnswer,
    RetrievedEvidence,
    RetrievedChunk,
    SearchResult,
    SourceReport,
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
        self.evaluator = ARESEvaluator(self.client)

    def decide_route(self, question: str):
        return self.router.decide(question)

    # Thin delegates retain the useful testing/extension seams of the original API.
    def _search(self, query: str) -> list[SearchResult]:
        return self.search.search(query)

    def _fetch_page(self, result: SearchResult) -> WebPage:
        return self.fetcher.fetch(result)

    def _fetch_all(self, results: list[SearchResult]):
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
        trace: list[PipelineEvent] = []

        def report(
            value: float,
            stage: str,
            status: str,
            message: str,
        ) -> None:
            trace.append(PipelineEvent(stage, status, message))
            if progress_callback:
                progress_callback(value, message)

        report(
            0.03,
            "Routing",
            "started",
            "Asking the local LLM whether this question needs web research…",
        )
        decision = self.router.decide(question)
        report(
            0.10,
            "Routing",
            "completed",
            f"Selected {decision.route.upper()}: {decision.reason or 'No reason returned.'}",
        )
        if decision.route == "direct":
            report(
                0.45,
                "Generation",
                "started",
                "Generating a direct answer with the local model…",
            )
            answer = self.generator.direct(question)
            report(1.0, "Generation", "completed", "Direct answer complete ✓")
            return RAGAnswer(
                question=question,
                answer=answer,
                route="direct",
                route_reason=decision.reason,
                search_query="",
                citations=[],
                source_reports=[],
                retrieved_evidence=[],
                pipeline_trace=trace,
                ares=None,
                indexed_source_count=0,
                indexed_chunk_count=0,
                latency_ms=round((time.time() - started) * 1000, 1),
            )

        report(
            0.14,
            "SerpAPI",
            "started",
            f'Sending search query to SerpAPI: “{decision.search_query}”',
        )
        results = self.search.search(decision.search_query)
        for result in results:
            snippet = f" | {result.snippet[:140]}" if result.snippet else ""
            report(
                0.22,
                "SerpAPI",
                "info",
                f"Result #{result.rank}: {result.title} — {result.url}{snippet}",
            )
        report(
            0.25,
            "SerpAPI",
            "completed",
            f"SerpAPI returned {len(results)} unique organic results.",
        )

        report(
            0.28,
            "Page fetching",
            "started",
            f"Browsing and extracting {len(results)} result pages concurrently…",
        )

        def fetch_event(status: str, message: str) -> None:
            report(0.40, "Page fetching", status, message)

        outcome = self.fetcher.fetch_all(results, fetch_event)
        pages = outcome.pages
        failed_by_url = {failure.result.url: failure.error for failure in outcome.failures}
        source_reports = [
            SourceReport(
                title=result.title,
                url=result.url,
                search_rank=result.rank,
                snippet=result.snippet,
                fetch_status="failed" if result.url in failed_by_url else "succeeded",
                error=failed_by_url.get(result.url, ""),
            )
            for result in results
        ]
        report(
            0.46,
            "Page fetching",
            "completed" if not outcome.failures else "warning",
            f"Kept {len(pages)} successful pages; skipped {len(outcome.failures)} failures.",
        )

        report(
            0.50,
            "Chunking",
            "started",
            "Splitting successful pages into overlapping token-aware chunks…",
        )
        chunks = self.retriever.chunk_pages(pages)
        report(
            0.64,
            "Chunking",
            "completed",
            f"Prepared {len(chunks)} chunks from {len(pages)} successful pages.",
        )
        report(
            0.68,
            "FAISS indexing and retrieval",
            "started",
            "Embedding chunks, storing them in temporary cosine FAISS, and searching with the question embedding…",
        )
        retrieved = self.retriever.retrieve(question, chunks)
        for index, item in enumerate(retrieved, start=1):
            report(
                0.76,
                "FAISS indexing and retrieval",
                "info",
                f"Retrieved #{index} (score {item.score:.4f}): {item.chunk.title} — {item.chunk.url}",
            )
        report(
            0.78,
            "FAISS indexing and retrieval",
            "completed",
            f"Indexed {len(chunks)} chunks and selected {len(retrieved)} within the "
            f"{self.config.max_context_tokens}-token context budget.",
        )
        report(
            0.82,
            "Generation",
            "started",
            "Generating an answer using only the retrieved chunks…",
        )
        answer, citations = self.generator.grounded(question, retrieved)
        report(
            0.90,
            "Generation",
            "completed",
            f"Generated the response with {len(citations)} cited sources.",
        )
        report(
            0.92,
            "ARES evaluation",
            "started",
            "Scoring context relevance, faithfulness, and answer relevance…",
        )
        ares = self.evaluator.evaluate(question, answer, retrieved)
        report(
            1.0,
            "ARES evaluation",
            "completed",
            f"ARES complete — overall {ares.overall:.2f}, faithfulness {ares.faithfulness:.2f}, "
            f"answer relevance {ares.answer_relevance:.2f}, context relevance {ares.context_relevance:.2f}.",
        )
        retrieved_evidence = [
            RetrievedEvidence(
                title=item.chunk.title,
                url=item.chunk.url,
                search_rank=item.chunk.search_rank,
                excerpt=item.chunk.text[:500] + "…",
                retrieval_score=round(item.score, 4),
                token_count=item.chunk.token_count,
            )
            for item in retrieved
        ]
        result = RAGAnswer(
            question=question,
            answer=answer,
            route="web",
            route_reason=decision.reason,
            search_query=decision.search_query,
            citations=citations,
            source_reports=source_reports,
            retrieved_evidence=retrieved_evidence,
            pipeline_trace=trace,
            ares=ares,
            indexed_source_count=len(pages),
            indexed_chunk_count=len(chunks),
            latency_ms=round((time.time() - started) * 1000, 1),
        )
        # Full pages, chunks, embeddings, and FAISS are request-local. Only compact
        # citation excerpts remain reachable after this method returns.
        return result
