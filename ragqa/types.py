"""Shared data contracts and user-facing errors."""

from dataclasses import dataclass
from typing import Literal


class WebRAGError(RuntimeError):
    """A user-facing failure in the search, extraction, or retrieval pipeline."""


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["direct", "web"]
    search_query: str
    reason: str


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int


@dataclass(frozen=True)
class WebPage:
    result: SearchResult
    text: str


@dataclass(frozen=True)
class WebChunk:
    text: str
    title: str
    url: str
    search_rank: int
    chunk_id: int
    token_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: WebChunk
    score: float


@dataclass(frozen=True)
class Citation:
    tag: str
    title: str
    url: str
    search_rank: int
    excerpt: str
    retrieval_score: float


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    route: Literal["direct", "web"]
    route_reason: str
    search_query: str
    citations: list[Citation]
    indexed_source_count: int
    indexed_chunk_count: int
    latency_ms: float
