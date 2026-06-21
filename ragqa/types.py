"""Shared data contracts and user-facing errors."""

from dataclasses import dataclass
from typing import Literal, Optional


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
class FetchFailure:
    result: SearchResult
    error: str


@dataclass(frozen=True)
class FetchOutcome:
    pages: list[WebPage]
    failures: list[FetchFailure]


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
class RetrievedEvidence:
    title: str
    url: str
    search_rank: int
    excerpt: str
    retrieval_score: float
    token_count: int


@dataclass(frozen=True)
class SourceReport:
    title: str
    url: str
    search_rank: int
    snippet: str
    fetch_status: Literal["succeeded", "failed"]
    error: str = ""


@dataclass(frozen=True)
class PipelineEvent:
    stage: str
    status: Literal["started", "completed", "info", "warning"]
    message: str


@dataclass(frozen=True)
class ARESScore:
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    overall: float
    details: dict


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    route: Literal["direct", "web"]
    route_reason: str
    search_query: str
    citations: list[Citation]
    source_reports: list[SourceReport]
    retrieved_evidence: list[RetrievedEvidence]
    pipeline_trace: list[PipelineEvent]
    ares: Optional[ARESScore]
    indexed_source_count: int
    indexed_chunk_count: int
    latency_ms: float
