"""Streamlit UI for the direct-or-web RAG assistant."""

from __future__ import annotations

import os

from dotenv import load_dotenv
import streamlit as st

from ragqa import RAGAnswer, WebRAGEngine, WebRAGError

load_dotenv()


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
.stStatus > div:nth-child(n+2) {
  font-size: 12px;
  color: rgba(0, 0, 0, 0.5);
}
.stStatus > div:nth-child(n+2) code {
  color: rgba(0, 0, 0, 0.4);
}
.pipeline-trace-container {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: #f9f9f9;
  height: 400px;
  overflow-y: auto;
  padding: 16px;
  font-family: monospace;
  font-size: 13px;
  scroll-behavior: smooth;
}
.pipeline-trace-event {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 10px 0;
  border-left: 3px solid transparent;
  padding-left: 12px;
  margin-bottom: 4px;
}
.pipeline-trace-event.started {
  border-left-color: #3b82f6;
  background: rgba(59, 130, 246, 0.05);
  padding: 8px 12px;
  border-radius: 4px;
}
.pipeline-trace-event.completed {
  border-left-color: #10b981;
  background: rgba(16, 185, 129, 0.05);
  padding: 8px 12px;
  border-radius: 4px;
}
.pipeline-trace-event.info {
  border-left-color: #6366f1;
  background: rgba(99, 102, 241, 0.05);
  padding: 8px 12px;
  border-radius: 4px;
}
.pipeline-trace-event.warning {
  border-left-color: #f59e0b;
  background: rgba(245, 158, 11, 0.05);
  padding: 8px 12px;
  border-radius: 4px;
}
.pipeline-trace-icon {
  font-size: 16px;
  min-width: 24px;
  flex-shrink: 0;
}
.pipeline-trace-content {
  flex: 1;
  min-width: 0;
}
.pipeline-trace-stage {
  font-weight: 600;
  color: #1f2937;
}
.pipeline-trace-message {
  color: #4b5563;
  margin-top: 2px;
  font-size: 12px;
  word-break: break-word;
}
</style>
<script>
function autoScrollTraceContainer() {
  const container = document.querySelector('.pipeline-trace-container');
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}
function autoScrollStatusBox() {
  const statusContent = document.querySelector('.stStatus > div > div:last-child');
  if (statusContent) {
    statusContent.scrollTop = statusContent.scrollHeight;
  }
}
setTimeout(autoScrollTraceContainer, 100);
setInterval(autoScrollStatusBox, 500);
</script>

<style>
.stStatus {
  max-width: 100% !important;
  height: 300px !important;
  display: flex !important;
  flex-direction: column !important;
  border: 1px solid #e0e0e0 !important;
  border-radius: 8px !important;
}
.stStatus > div {
  display: flex !important;
  flex-direction: column !important;
  height: 100% !important;
}
.stStatus > div > div:first-child {
  flex-shrink: 0;
  padding: 12px 16px !important;
  border-bottom: 1px solid #e0e0e0;
  background: #ffffff;
  border-radius: 8px 8px 0 0;
}
.stStatus > div > div:last-child {
  flex: 1;
  overflow-y: auto !important;
  scroll-behavior: smooth;
  padding: 12px 16px !important;
  background: #f9f9f9;
  border-radius: 0 0 8px 8px;
}
.stStatus > div > div:last-child > div {
  font-family: monospace;
  font-size: 13px;
  color: #4b5563;
  padding: 8px 12px;
  border-left: 3px solid #3b82f6;
  background: rgba(59, 130, 246, 0.03);
  border-radius: 4px;
  margin-bottom: 6px;
  word-break: break-word;
}
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

    if answer.pipeline_trace:
        with st.expander(
            f"Background pipeline trace ({len(answer.pipeline_trace)} events)",
            expanded=False,
        ):
            icons = {
                "started": "🔄",
                "completed": "✅",
                "info": "ℹ️",
                "warning": "⚠️",
            }

            trace_html = '<div class="pipeline-trace-container">'
            for event in answer.pipeline_trace:
                icon = icons.get(event.status, "•")
                status_class = event.status.lower()
                trace_html += f'''
                <div class="pipeline-trace-event {status_class}">
                    <div class="pipeline-trace-icon">{icon}</div>
                    <div class="pipeline-trace-content">
                        <div class="pipeline-trace-stage">{event.stage}</div>
                        <div class="pipeline-trace-message">{event.message}</div>
                    </div>
                </div>
                '''
            trace_html += '</div>'

            st.markdown(trace_html, unsafe_allow_html=True)

            st.markdown(
                f"<div style='text-align: right; font-size: 12px; color: #999; margin-top: 8px;'>"
                f"Latest: {answer.pipeline_trace[-1].stage}</div>",
                unsafe_allow_html=True,
            )

    if answer.source_reports:
        succeeded = sum(
            source.fetch_status == "succeeded" for source in answer.source_reports
        )
        failed = len(answer.source_reports) - succeeded
        with st.expander(
            f"SerpAPI results and page browsing ({succeeded} succeeded, {failed} skipped)",
            expanded=failed > 0,
        ):
            for source in answer.source_reports:
                icon = "✅" if source.fetch_status == "succeeded" else "⚠️"
                st.markdown(
                    f"{icon} **#{source.search_rank} [{source.title}]({source.url})** — "
                    f"`{source.fetch_status}`"
                )
                if source.snippet:
                    st.caption(source.snippet)
                if source.error:
                    st.warning(source.error)

    if answer.retrieved_evidence:
        with st.expander(
            f"Retrieved FAISS chunks ({len(answer.retrieved_evidence)})",
            expanded=False,
        ):
            for index, evidence in enumerate(answer.retrieved_evidence, start=1):
                st.markdown(
                    f"**#{index} [{evidence.title}]({evidence.url})** · "
                    f"score `{evidence.retrieval_score:.4f}` · "
                    f"{evidence.token_count} tokens"
                )
                st.caption(evidence.excerpt)
                st.divider()

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

    if answer.ares is not None:
        with st.expander("ARES evaluation", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faithfulness", f"{answer.ares.faithfulness * 100:.0f}%")
            col2.metric(
                "Answer relevance", f"{answer.ares.answer_relevance * 100:.0f}%"
            )
            col3.metric(
                "Context relevance", f"{answer.ares.context_relevance * 100:.0f}%"
            )
            col4.metric("Overall", f"{answer.ares.overall * 100:.0f}%")
            for dimension, reasoning in answer.ares.details.items():
                st.caption(f"**{dimension}:** {reasoning}")

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
        live_status = st.status("Running the RAG pipeline…", expanded=True)

        def show_pipeline_progress(value: float, message: str) -> None:
            progress.progress(max(0.0, min(value, 1.0)), text=message)
            live_status.write(message)

        try:
            result = engine.answer(question, show_pipeline_progress)
            live_status.update(
                label="Pipeline complete ✓", state="complete", expanded=True
            )
            st.session_state["history"].append(result)
            st.rerun()
        except WebRAGError as exc:
            progress.empty()
            live_status.update(label="Pipeline failed", state="error", expanded=True)
            st.error(str(exc))
        except Exception as exc:
            progress.empty()
            live_status.update(label="Pipeline failed", state="error", expanded=True)
            st.error(f"Unexpected error: {exc}")

for item in reversed(st.session_state["history"]):
    render_answer(item)
