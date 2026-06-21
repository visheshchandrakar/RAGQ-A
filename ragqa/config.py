"""Central configuration for the direct-or-web RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gen_model: str = "Qwen3-8B (4-bit)"
    search_result_limit: int = 5
    search_timeout_seconds: int = 15
    fetch_timeout_seconds: int = 12
    max_response_bytes: int = 5 * 1024 * 1024
    chunk_size: int = 300
    chunk_overlap: int = 50
    max_context_tokens: int = 1400
    max_context_chunks: int = 6
    max_chunks_per_source: int = 2


DEFAULT_CONFIG = PipelineConfig()
