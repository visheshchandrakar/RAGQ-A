"""Public API for the direct-or-web RAG assistant."""

from .config import DEFAULT_CONFIG, PipelineConfig
from .engine import WebRAGEngine
from .llm import LocalQwen3
from .types import (
    Citation,
    RAGAnswer,
    RetrievedChunk,
    RouteDecision,
    SearchResult,
    WebChunk,
    WebPage,
    WebRAGError,
)

__all__ = [
    "Citation",
    "DEFAULT_CONFIG",
    "LocalQwen3",
    "PipelineConfig",
    "RAGAnswer",
    "RetrievedChunk",
    "RouteDecision",
    "SearchResult",
    "WebChunk",
    "WebPage",
    "WebRAGEngine",
    "WebRAGError",
]
