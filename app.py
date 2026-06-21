"""Streamlit UI for the direct-or-web RAG assistant."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from dotenv import load_dotenv
import streamlit as st

from ragqa import RAGAnswer, WebRAGEngine, WebRAGError
from ragqa.types import PipelineEvent

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
.main .block-container { padding-top: 0 !important; max-width: 1050px; }
[data-testid="stSidebar"] { min-width: 230px !important; max-width: 230px !important; }
.main .block-container > div:first-child { margin-top: 0 !important; }
h1:first-of-type {
  margin-top: 0 !important;
  padding-top: 0 !important;
  margin-bottom: 0 !important;
  font-size: 1.6rem !important;
  line-height: 1.2 !important;
}
h1:first-of-type + div p {
  margin-top: 0 !important;
  font-size: 0.8rem !important;
}
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
.pipeline-trace-group {
  border: 1px solid #e5e7eb;
  border-radius: 0;
  margin-bottom: 8px;
  background: #ffffff;
  overflow: hidden;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}
.pipeline-trace-group.started { border-left: 3px solid #3b82f6; }
.pipeline-trace-group.completed { border-left: 3px solid #10b981; }
.pipeline-trace-group.warning { border-left: 3px solid #f59e0b; }
.pipeline-trace-group-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  background: #f3f4f6;
}
.pipeline-trace-group-header .pipeline-trace-stage {
  font-size: 13px;
}
.pipeline-trace-group-body {
  padding: 2px 10px 4px 10px;
}
.pipeline-trace-event {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 4px 0;
  border-left: 3px solid transparent;
  padding-left: 12px;
  margin-bottom: 2px;
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
.domain-badge {
  display: inline-block;
  background: #ede9fe;
  color: #5b21b6;
  border-radius: 999px;
  padding: 1px 9px;
  font-size: 11px;
  font-weight: 600;
  text-transform: lowercase;
}
.question-display {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  border-radius: 10px;
  margin: 16px 0;
  font-size: 20px;
  line-height: 1.6;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  word-break: break-word;
}
.loader-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  padding: 40px 20px;
  background: #f8f9fa;
  border-radius: 10px;
  border: 1px solid #e9ecef;
  margin: 16px 0;
}
.spinner {
  display: inline-block;
  width: 40px;
  height: 40px;
  border: 4px solid #e9ecef;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.loader-text {
  font-size: 14px;
  color: #666;
  font-weight: 500;
}
[data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
  align-items: flex-end !important;
}
[data-testid="stFormSubmitButton"] {
  margin: 4px 0 0 0 !important;
  width: 200px !important;
  height: auto !important;
  align-self: flex-end !important;
  flex: 0 0 auto !important;
}
[data-testid="stFormSubmitButton"] button {
  border-radius: 20px;
  width: 200px !important;
  white-space: nowrap;
  padding: 6px 16px;
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


def render_answer_simple(answer: RAGAnswer) -> None:
    """Render just the question and answer text."""
    route_class = "route-web" if answer.route == "web" else "route-direct"
    route_label = "🌐 Web research" if answer.route == "web" else "⚡ Direct answer"
    st.markdown(f'<span class="{route_class}">{route_label}</span>', unsafe_allow_html=True)
    st.markdown(f"<span style='font-size: 1.3em; font-weight: 600;'>Q: {answer.question}</span>", unsafe_allow_html=True)

    rendered = answer.answer
    for number, citation in enumerate(answer.citations, start=1):
        rendered = rendered.replace(
            citation.tag,
            f"[[{number}]]({citation.url} \"{citation.title}\")",
        )
    st.markdown(rendered)
    st.caption(f"Completed in {answer.latency_ms:.0f} ms")
    st.divider()


_URL_RE = re.compile(r"https?://\S+")
_SCORE_RE = re.compile(r"score (\d+\.\d+)")


def _site_name(url: str) -> str:
    netloc = urlparse(url).netloc.removeprefix("www.")
    labels = netloc.split(".")
    return ".".join(labels[:-1]) if len(labels) > 1 else netloc


def _urls_to_domain_badges(text: str) -> str:
    return _URL_RE.sub(
        lambda match: f'<span class="domain-badge">{_site_name(match.group(0))}</span>',
        text,
    )


def render_pipeline_timeline(events: list[PipelineEvent]) -> str:
    """Render pipeline events grouped by stage as a scrollable timeline (HTML)."""
    icon = {"started": "⏳", "completed": "✅", "info": "ℹ️", "warning": "⚠️"}
    if not events:
        return (
            '<div class="pipeline-trace-container">'
            '<div class="pipeline-trace-message">No pipeline activity yet.</div>'
            "</div>"
        )

    groups: list[tuple[str, list[PipelineEvent]]] = []
    for event in events:
        if groups and groups[-1][0] == event.stage:
            groups[-1][1].append(event)
        else:
            groups.append((event.stage, [event]))

    group_html = []
    for stage, group_events in groups:
        statuses = {item.status for item in group_events}
        group_status = (
            "warning" if "warning" in statuses
            else "completed" if "completed" in statuses
            else "started"
        )
        def trimmed_message(event: PipelineEvent) -> str:
            message = event.message
            if stage == "SerpAPI" and event.status == "info":
                message = message.split(" | ", 1)[0]
                if len(message) > 90:
                    message = message[:90].rstrip() + "…"
            if stage == "Retrieval" and event.status == "info":
                prefix, sep, title = message.rpartition(") ")
                if sep and len(title) > 70:
                    title = title[:70].rstrip() + "…"
                message = f"{prefix}{sep}{title}" if sep else message
            if stage in ("SerpAPI", "Page fetching", "Retrieval"):
                message = _urls_to_domain_badges(message)
            if stage == "Retrieval":
                message = _SCORE_RE.sub(
                    lambda m: f"score <strong style='color:#2563eb;'>{m.group(1)}</strong>",
                    message,
                )
            return message

        sub_rows = "".join(
            f'<div class="pipeline-trace-event {event.status}">'
            f'<div class="pipeline-trace-icon">{icon.get(event.status, "•")}</div>'
            f'<div class="pipeline-trace-content">'
            f'<div class="pipeline-trace-message">{trimmed_message(event)}</div>'
            f"</div></div>"
            for event in group_events
        )
        group_html.append(
            f'<div class="pipeline-trace-group {group_status}">'
            f'<div class="pipeline-trace-group-header">'
            f'<div class="pipeline-trace-icon">{icon.get(group_status, "•")}</div>'
            f'<div class="pipeline-trace-stage">{stage} ({len(group_events)})</div>'
            f"</div>"
            f'<div class="pipeline-trace-group-body">{sub_rows}</div>'
            f"</div>"
        )

    return f'<div class="pipeline-trace-container">{"".join(group_html)}</div>'


def render_ares_evaluation(answer: RAGAnswer) -> None:
    """Render ARES evaluation metrics."""
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


def render_answer_details(answer: RAGAnswer) -> None:
    """Render all the detailed information about the answer."""
    if answer.retrieved_evidence:
        with st.expander(
            f"Retrieved FAISS chunks ({len(answer.retrieved_evidence)})",
            expanded=False,
        ):
            for index, evidence in enumerate(answer.retrieved_evidence, start=1):
                st.markdown(
                    f"**#{index} [{evidence.title}]({evidence.url})** · "
                    f"score <strong style='color:#2563eb;'>{evidence.retrieval_score:.4f}</strong> · "
                    f"{evidence.token_count} tokens",
                    unsafe_allow_html=True,
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

if "history" not in st.session_state:
    st.session_state["history"] = []
if "current_result" not in st.session_state:
    st.session_state["current_result"] = None
if "current_question" not in st.session_state:
    st.session_state["current_question"] = None


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
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("Question & Answers")
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area(
                "Ask anything",
                placeholder="e.g. What changed in Python's latest stable release?",
                height=90,
            )
            submitted = st.form_submit_button("Ask ↗", type="primary")

        pending_slot = st.empty()

        st.markdown("---")
        for item in reversed(st.session_state["history"]):
            render_answer_simple(item)
            render_ares_evaluation(item)
            render_answer_details(item)

    with col_right:
        st.subheader("Background steps")
        timeline_slot = st.empty()
        last_trace = (
            st.session_state["current_result"].pipeline_trace
            if st.session_state["current_result"] is not None
            else []
        )
        timeline_slot.markdown(render_pipeline_timeline(last_trace), unsafe_allow_html=True)

    if submitted and question.strip():
        st.session_state["current_result"] = None
        st.session_state["current_question"] = question
        with pending_slot:
            st.markdown(
                f'<div class="question-display">❓ {question}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '''<div class="loader-container">
                    <div class="spinner"></div>
                    <div class="loader-text">Generating answer...</div>
                </div>''',
                unsafe_allow_html=True,
            )

        events: list[PipelineEvent] = []

        def show_pipeline_progress(value: float, stage: str, status: str, message: str) -> None:
            events.append(PipelineEvent(stage, status, message))
            timeline_slot.markdown(render_pipeline_timeline(events), unsafe_allow_html=True)

        try:
            result = engine.answer(question, show_pipeline_progress)
            st.session_state["history"].append(result)
            st.session_state["current_result"] = result
            st.session_state["current_question"] = None
            st.rerun()
        except WebRAGError as exc:
            pending_slot.error(str(exc))
        except Exception as exc:
            pending_slot.error(f"Unexpected error: {exc}")
