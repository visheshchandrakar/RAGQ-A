"""Network-free tests for the direct-or-web RAG pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ragqa import (
    SearchResult,
    WebChunk,
    WebPage,
    WebRAGEngine,
    WebRAGError,
)


class FakeLLM:
    def __init__(self, route="direct", malformed=False):
        self.route = route
        self.malformed = malformed
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        content = messages[-1]["content"]
        if "You route questions" in content:
            if self.malformed:
                return "this is not JSON"
            return json.dumps(
                {
                    "route": self.route,
                    "search_query": "latest python release" if self.route == "web" else "",
                    "reason": "test route",
                }
            )
        if "Web passages:" in content:
            return "The retrieved pages agree on the result [SOURCE_1] [SOURCE_2]."
        if "ARES-style evaluator" in content:
            return json.dumps(
                {
                    "context_relevance": 0.88,
                    "faithfulness": 0.92,
                    "answer_relevance": 0.9,
                    "reasoning": {
                        "context": "Useful passages",
                        "faithfulness": "Claims are supported",
                        "relevance": "The question is answered",
                    },
                }
            )
        return "This is a direct answer."


class FakeEmbedder:
    """Small deterministic embedding model suitable for FAISS tests."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    lowered.count("python") + 0.1,
                    lowered.count("release") + 0.1,
                    lowered.count("database") + 0.1,
                    len(lowered.split()) / 1000 + 0.1,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def result(rank, url=None):
    return SearchResult(
        title=f"Result {rank}",
        url=url or f"https://example{rank}.com/article",
        snippet="A search result snippet",
        rank=rank,
    )


def page(search_result):
    text = (
        "Python release information and documented changes for developers. " * 80
    )
    return WebPage(search_result, text)


def make_engine(llm=None, key="test-key", search_provider=None, page_fetcher=None):
    return WebRAGEngine(
        llm=llm or FakeLLM(),
        embedder=FakeEmbedder(),
        serpapi_key=key,
        search_provider=search_provider,
        page_fetcher=page_fetcher,
    )


def test_direct_route_never_calls_search_or_fetch():
    search_calls = []

    def search_provider(query, key):
        search_calls.append(query)
        raise AssertionError("Direct route must not search")

    engine = make_engine(FakeLLM("direct"), search_provider=search_provider)
    answer = engine.answer("Explain recursion simply")

    assert answer.route == "direct"
    assert answer.answer == "This is a direct answer."
    assert answer.citations == []
    assert search_calls == []


def test_malformed_router_output_falls_back_to_web():
    decision = make_engine(FakeLLM(malformed=True)).decide_route("What happened today?")
    assert decision.route == "web"
    assert decision.search_query == "What happened today?"
    assert "invalid" in decision.reason


def test_web_route_searches_indexes_retrieves_and_cites():
    searched = []
    results = [result(1), result(2)]

    def search_provider(query, key):
        searched.append((query, key))
        return results

    engine = make_engine(
        FakeLLM("web"), search_provider=search_provider, page_fetcher=page
    )
    answer = engine.answer("What is in the latest Python release?")

    assert answer.route == "web"
    assert searched == [("latest python release", "test-key")]
    assert answer.indexed_source_count == 2
    assert answer.indexed_chunk_count > 2
    assert {citation.url for citation in answer.citations} == {
        results[0].url,
        results[1].url,
    }
    assert answer.indexed_chunk_count > len(answer.citations)
    assert len(answer.source_reports) == 2
    assert answer.retrieved_evidence
    assert answer.ares.overall == 0.9
    assert any(event.stage == "SerpAPI" for event in answer.pipeline_trace)
    fetch_events = [
        event
        for event in answer.pipeline_trace
        if event.stage == "Page fetching" and event.status == "completed"
    ]
    assert {event.message for event in fetch_events} == {
        f"{item.url} — {item.title}" for item in results
    }


def test_missing_key_only_fails_when_web_route_is_used():
    direct = make_engine(FakeLLM("direct"), key="").answer("Say hello")
    assert direct.route == "direct"

    with pytest.raises(WebRAGError, match="SERPAPI_KEY"):
        make_engine(FakeLLM("web"), key="").answer("What happened today?")


def test_search_filters_invalid_urls_and_deduplicates():
    raw = [
        result(1, "https://example.com/page#section"),
        result(2, "https://example.com/page"),
        result(3, "ftp://example.com/file"),
        result(4, "https://other.example/page"),
    ]
    engine = make_engine(search_provider=lambda query, key: raw)
    selected = engine._search("query")
    assert [item.url for item in selected] == [
        "https://example.com/page",
        "https://other.example/page",
    ]


def test_page_failure_is_reported_while_successful_pages_continue():
    results = [result(1), result(2)]

    def fetcher(item):
        if item.rank == 2:
            raise WebRAGError(f"Could not fetch {item.url}: timeout")
        return page(item)

    engine = make_engine(page_fetcher=fetcher)
    outcome = engine._fetch_all(results)
    assert [item.result.url for item in outcome.pages] == [results[0].url]
    assert len(outcome.failures) == 1
    assert outcome.failures[0].result.url == results[1].url


def test_all_page_failures_still_stop_ingestion():
    results = [result(1), result(2)]

    def fetcher(item):
        raise WebRAGError(f"Could not fetch {item.url}: timeout")

    engine = make_engine(page_fetcher=fetcher)
    with pytest.raises(WebRAGError, match="None of the selected web pages"):
        engine._fetch_all(results)


def test_chunking_uses_overlap_and_assigns_metadata():
    engine = make_engine()
    source = result(1)
    source_text = "alpha beta gamma delta epsilon and zeta. " * 140
    source_tokens = engine.enc.encode(source_text)
    chunks = engine._chunk_pages([WebPage(source, source_text)])

    expected_starts = range(
        0, len(source_tokens), engine.CHUNK_SIZE - engine.CHUNK_OVERLAP
    )
    expected_count = sum(
        len(engine.enc.decode(source_tokens[start : start + engine.CHUNK_SIZE]).strip())
        >= 80
        for start in expected_starts
    )
    assert len(chunks) == expected_count
    assert chunks[0].token_count == engine.CHUNK_SIZE
    assert chunks[1].token_count == engine.CHUNK_SIZE
    assert engine.CHUNK_SIZE - engine.CHUNK_OVERLAP == 250
    assert all(chunk.url == source.url for chunk in chunks)


def test_retrieval_obeys_context_budget_and_source_diversity():
    engine = make_engine()
    chunks = []
    for i in range(12):
        source_number = 1 if i < 8 else 2
        chunks.append(
            WebChunk(
                text=("Python release " * 150) + str(i),
                title=f"Source {source_number}",
                url=f"https://source{source_number}.example",
                search_rank=source_number,
                chunk_id=i,
                token_count=300,
            )
        )

    retrieved = engine._retrieve("Python release", chunks)
    assert sum(item.chunk.token_count for item in retrieved) <= engine.MAX_CONTEXT_TOKENS
    assert len({item.chunk.url for item in retrieved}) == 2
    assert all(
        sum(item.chunk.url == url for item in retrieved) <= engine.MAX_CHUNKS_PER_SOURCE
        for url in {item.chunk.url for item in retrieved}
    )
