"""Lightweight Streamlit surface tests; no models or network are loaded."""

from streamlit.testing.v1 import AppTest

from ragqa import Citation, RAGAnswer


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
