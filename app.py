"""Streamlit UI for the direct-or-web RAG assistant."""

from __future__ import annotations

import os

import streamlit as st

from ragqa import RAGAnswer, WebRAGEngine, WebRAGError


st.set_page_config(
    page_title="Direct-or-Web RAG Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main .block-container { padding-top: 1.5rem; max-width: 1050px; }
.route-direct, .route-web {
  display:inline-block; border-radius:5px; padding:3px 9px;
  font-size:12px; font-weight:650; margin-bottom:8px;
}
.route-direct { background:#e0f2fe; color:#075985; }
.route-web { background:#dcfce7; color:#166534; }
</style>
""",
    unsafe_allow_html=True,
)


def get_serpapi_key() -> str:
    """Environment takes precedence; Streamlit secrets are an optional fallback."""
    key = os.getenv("SERPAPI_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("SERPAPI_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def get_engine() -> WebRAGEngine | None:
    return st.session_state.get("engine")


def init_engine(progress_callback=None) -> None:
    st.session_state["engine"] = WebRAGEngine(
        progress_callback=progress_callback,
        serpapi_key=get_serpapi_key(),
    )


def render_answer(answer: RAGAnswer) -> None:
    route_class = "route-web" if answer.route == "web" else "route-direct"
    route_label = "🌐 Web research" if answer.route == "web" else "⚡ Direct answer"
    st.markdown(f'<span class="{route_class}">{route_label}</span>', unsafe_allow_html=True)
    st.markdown(f"**Q: {answer.question}**")

    rendered = answer.answer
    for number, citation in enumerate(answer.citations, start=1):
        rendered = rendered.replace(
            citation.tag,
            f"[[{number}]]({citation.url} \"{citation.title}\")",
        )
    st.markdown(rendered)

    with st.expander("Routing details", expanded=False):
        st.markdown(f"**Decision:** `{answer.route}`")
        st.markdown(f"**Reason:** {answer.route_reason or 'No reason returned.'}")
        if answer.search_query:
            st.markdown(f"**Search query:** `{answer.search_query}`")
            st.markdown(
                f"**Temporary index:** {answer.indexed_source_count} sources · "
                f"{answer.indexed_chunk_count} chunks"
            )

    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})", expanded=True):
            for number, citation in enumerate(answer.citations, start=1):
                st.markdown(f"**[{number}] [{citation.title}]({citation.url})**")
                st.caption(
                    f"Google rank {citation.search_rank} · "
                    f"retrieval score {citation.retrieval_score:.4f}"
                )
                st.caption(citation.excerpt)
                st.divider()
    elif answer.route == "web":
        st.warning("The generated answer did not reference any retrieved source tags.")

    st.caption(f"Completed in {answer.latency_ms:.0f} ms")
    st.divider()


if "history" not in st.session_state:
    st.session_state["history"] = []


with st.sidebar:
    st.title("🌐 Web RAG Assistant")
    st.caption("Local Qwen3 · SerpAPI · temporary FAISS retrieval")
    st.markdown("---")

    if get_engine() is None:
        st.caption("The language and embedding models run locally.")
        if st.button("Load local models", type="primary"):
            progress = st.progress(0.0, text="Preparing local models…")

            def show_model_progress(value: float, message: str) -> None:
                progress.progress(max(0.0, min(value, 1.0)), text=message)

            init_engine(show_model_progress)
            progress.progress(1.0, text="Local models are ready ✓")
            st.rerun()
    else:
        st.success("Local models loaded ✓")

    if get_serpapi_key():
        st.success("SERPAPI_KEY configured ✓")
    else:
        st.warning("SERPAPI_KEY is not configured. Direct answers still work.")
        st.caption("Set it as an environment variable or Streamlit secret.")

    st.markdown("---")
    with st.expander("Pipeline settings"):
        st.markdown(
            f"""
| Parameter | Value |
|---|---|
| Generation model | `{WebRAGEngine.GEN_MODEL}` |
| Embedding model | `{WebRAGEngine.EMBED_MODEL}` |
| Search results | {WebRAGEngine.SEARCH_RESULT_LIMIT} |
| Chunking | {WebRAGEngine.CHUNK_SIZE} / {WebRAGEngine.CHUNK_OVERLAP} tokens |
| Context budget | {WebRAGEngine.MAX_CONTEXT_TOKENS} tokens |
| Vector index | `IndexFlatIP` (cosine) |
| Persistence | Per-query only |
"""
        )

    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()


st.title("Direct-or-Web RAG Assistant")
st.caption(
    "The local model decides whether to answer directly or research the web, "
    "then grounds web answers in a temporary FAISS index."
)

with st.expander("Pipeline architecture", expanded=False):
    st.code(
        """Question → local LLM router
  ├─ DIRECT → local answer
  └─ WEB → SerpAPI → fetch top pages → extract and chunk
                                  → embed → temporary cosine FAISS
                                  → retrieve within token budget
                                  → grounded answer + URL citations

The temporary page content and FAISS index are discarded after each answer.""",
        language="text",
    )

st.markdown("---")
engine = get_engine()
if engine is None:
    st.warning("Load the local models in the sidebar to begin.")
else:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Ask anything",
            placeholder="e.g. What changed in Python's latest stable release?",
            height=90,
        )
        submitted = st.form_submit_button("Ask ↗", type="primary")

    if submitted and question.strip():
        progress = st.progress(0.0, text="Starting…")

        def show_pipeline_progress(value: float, message: str) -> None:
            progress.progress(max(0.0, min(value, 1.0)), text=message)

        try:
            result = engine.answer(question, show_pipeline_progress)
            st.session_state["history"].append(result)
            st.rerun()
        except WebRAGError as exc:
            progress.empty()
            st.error(str(exc))
        except Exception as exc:
            progress.empty()
            st.error(f"Unexpected error: {exc}")

for item in reversed(st.session_state["history"]):
    render_answer(item)
