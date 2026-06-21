"""Backward-compatible imports for the modular :mod:`ragqa` package.

New code should import from ``ragqa`` or its focused submodules directly.
"""

from ragqa import (
    ARESScore,
    Citation,
    DEFAULT_CONFIG,
    FetchFailure,
    FetchOutcome,
    LocalQwen3,
    PipelineConfig,
    PipelineEvent,
    RAGAnswer,
    RetrievedChunk,
    RetrievedEvidence,
    RouteDecision,
    SearchResult,
    SourceReport,
    WebChunk,
    WebPage,
    WebRAGEngine,
    WebRAGError,
)

__all__ = [
    "ARESScore",
    "Citation",
    "DEFAULT_CONFIG",
    "FetchFailure",
    "FetchOutcome",
    "LocalQwen3",
    "PipelineConfig",
    "PipelineEvent",
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
