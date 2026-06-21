"""Lightweight Streamlit surface tests; no models or network are loaded."""

from streamlit.testing.v1 import AppTest

from ragqa import (
    ARESScore,
    Citation,
    PipelineEvent,
    RAGAnswer,
    RetrievedEvidence,
    SourceReport,
)


def test_initial_ui_is_web_first_and_has_no_pdf_uploader():
    app = AppTest.from_file("app.py").run(timeout=10)

    assert not app.exception
    assert "Direct-or-Web RAG Assistant" in [title.value for title in app.title]
    assert "Load local models" in [button.label for button in app.button]
    assert len(app.get("file_uploader")) == 0


def test_web_answer_renders_route_status_index_details_and_clickable_source():
    answer = RAGAnswer(
        question="What changed?",
        answer="A grounded fact [SOURCE_1].",
        route="web",
        route_reason="The answer needs current information.",
        search_query="latest changes",
        citations=[
            Citation(
                tag="[SOURCE_1]",
                title="Example source",
                url="https://example.com/source",
                search_rank=1,
                excerpt="Grounding excerpt",
                retrieval_score=0.91,
            )
        ],
        source_reports=[
            SourceReport(
                title="Example source",
                url="https://example.com/source",
                search_rank=1,
                snippet="A useful search result",
                fetch_status="succeeded",
            ),
            SourceReport(
                title="Blocked source",
                url="https://example.com/blocked",
                search_rank=2,
                snippet="A blocked result",
                fetch_status="failed",
                error="The page could not be parsed",
            ),
        ],
        retrieved_evidence=[
            RetrievedEvidence(
                title="Example source",
                url="https://example.com/source",
                search_rank=1,
                excerpt="Retrieved grounding chunk",
                retrieval_score=0.91,
                token_count=250,
            )
        ],
        pipeline_trace=[
            PipelineEvent("Routing", "completed", "Selected WEB"),
            PipelineEvent("Page fetching", "warning", "Skipped one failed page"),
        ],
        ares=ARESScore(0.9, 0.8, 0.85, 0.85, {"faithfulness": "Supported"}),
        indexed_source_count=1,
        indexed_chunk_count=3,
        latency_ms=12,
    )
    app = AppTest.from_file("app.py")
    app.session_state["engine"] = object()
    app.session_state["history"] = [answer]
    app.run(timeout=10)

    markdown = "\n".join(item.value for item in app.markdown)
    assert not app.exception
    assert "🌐 Web research" in markdown
    assert "latest changes" in markdown
    assert "1 sources · 3 chunks" in markdown
    assert "[Example source](https://example.com/source)" in markdown
    expander_labels = [item.label for item in app.expander]
    assert any("Background pipeline trace" in label for label in expander_labels)
    assert any("Retrieved FAISS chunks" in label for label in expander_labels)
    assert len(app.metric) == 4
