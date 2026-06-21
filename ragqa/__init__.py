"""Public API for the direct-or-web RAG assistant."""

from .config import DEFAULT_CONFIG, PipelineConfig
from .engine import WebRAGEngine
from .llm import LocalQwen3
from .types import (
    ARESScore,
    Citation,
    FetchFailure,
    FetchOutcome,
    PipelineEvent,
    RAGAnswer,
    RetrievedEvidence,
    RetrievedChunk,
    RouteDecision,
    SearchResult,
    SourceReport,
    WebChunk,
    WebPage,
    WebRAGError,
)

__all__ = [
    "ARESScore",
    "Citation",
    "DEFAULT_CONFIG",
    "FetchFailure",
    "FetchOutcome",
    "LocalQwen3",
    "PipelineEvent",
    "PipelineConfig",
    "RAGAnswer",
    "RetrievedChunk",
    "RetrievedEvidence",
    "RouteDecision",
    "SearchResult",
    "SourceReport",
    "WebChunk",
    "WebPage",
    "WebRAGEngine",
    "WebRAGError",
]
