"""Streamlit UI for the direct-or-web RAG assistant."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
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

UI_MARKUP_PATH = Path(__file__).with_name("app_ui.html")
st.markdown(UI_MARKUP_PATH.read_text(encoding="utf-8"), unsafe_allow_html=True)

USER_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path d="M12 12.25a4.25 4.25 0 1 0 0-8.5 4.25 4.25 0 0 0 0 8.5Z"/>
  <path d="M4.5 20.25c.45-3.65 3.05-5.75 7.5-5.75s7.05 2.1 7.5 5.75"/>
</svg>
"""

AI_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path d="M12 2.75c.55 3.8 2.7 5.95 6.5 6.5-3.8.55-5.95 2.7-6.5 6.5-.55-3.8-2.7-5.95-6.5-6.5 3.8-.55 5.95-2.7 6.5-6.5Z"/>
  <path d="M18.25 15.25c.25 1.7 1.3 2.75 3 3-1.7.25-2.75 1.3-3 3-.25-1.7-1.3-2.75-3-3 1.7-.25 2.75-1.3 3-3Z"/>
</svg>
"""


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


def render_question(question: str) -> None:
    """Render the question consistently while loading and after completion."""
    st.markdown(
        '<div class="question-row">'
        f'<div class="message-avatar user-avatar" aria-label="You">{USER_ICON}</div>'
        f'<div class="question-display">{html.escape(question)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_answer_simple(answer: RAGAnswer) -> None:
    """Render just the question and answer text."""
    render_question(answer.question)
    route_class = "route-web" if answer.route == "web" else "route-direct"
    route_label = "Web research" if answer.route == "web" else "Direct answer"

    rendered = answer.answer
    for number, citation in enumerate(answer.citations, start=1):
        rendered = rendered.replace(
            citation.tag,
            f"[[{number}]]({citation.url} \"{citation.title}\")",
        )

    st.markdown(
        '<div class="answer-meta-row">'
        f'<div class="message-avatar ai-avatar" aria-label="AI assistant">{AI_ICON}</div>'
        f'<span class="{route_class}">{route_label}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _, answer_column = st.columns([0.07, 0.93], gap="small")
    with answer_column:
        st.markdown(rendered)
        st.caption(f"Completed in {answer.latency_ms / 1000:.1f} s")
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
        progress_spinner = (
            '<div class="pipeline-trace-progress" role="status">'
            '<strong>In progress</strong>'
            '<div class="pipeline-trace-spinner" aria-hidden="true"></div>'
            '</div>'
            if group_status == "started"
            else ""
        )
        group_html.append(
            f'<div class="pipeline-trace-group {group_status}">'
            f'<div class="pipeline-trace-group-header">'
            f'<div class="pipeline-trace-icon">{icon.get(group_status, "•")}</div>'
            f'<div class="pipeline-trace-stage">{stage} ({len(group_events)})</div>'
            f"{progress_spinner}"
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
    st.markdown(
        '<nav class="app-navigation">'
        '<a class="active" href="/">💬 Assistant</a>'
        '<a href="/architecture">🏗️ Architecture</a>'
        "</nav>",
        unsafe_allow_html=True,
    )
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
        st.markdown(
            f"""
<div class="model-ready-card">
  <div class="model-ready-header">
    <span class="model-ready-dot"></span>
    <span>Models ready</span>
  </div>
  <div class="model-ready-item">
    <span class="model-ready-label">Generation</span>
    <strong>{html.escape(WebRAGEngine.GEN_MODEL)}</strong>
  </div>
  <div class="model-ready-item">
    <span class="model-ready-label">Embedding</span>
    <strong>{html.escape(WebRAGEngine.EMBED_MODEL)}</strong>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


st.title("Direct-or-Web RAG Assistant")
st.caption(
    "The local model decides whether to answer directly or research the web, "
    "then grounds web answers in a temporary FAISS index."
)

engine = get_engine()
if engine is None:
    st.warning("Load the local models in the sidebar to begin.")
else:
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("Question & Answers")
        with st.container(height=800, border=False):
            with st.form("question_form", clear_on_submit=True):
                question = st.text_area(
                    "Ask anything",
                    placeholder="e.g. What changed in Python's latest stable release?",
                    height=90,
                )
                submitted = st.form_submit_button("Ask ↗", type="primary")

            submitted_question = question.strip() if submitted else ""
            if submitted_question:
                st.session_state["current_question"] = submitted_question

            st.markdown("---")

            if st.session_state["current_question"]:
                render_question(st.session_state["current_question"])

            pending_slot = st.empty()
            if submitted_question:
                pending_slot.markdown(
                    '''<div class="loader-container">
                        <div class="spinner"></div>
                        <div class="loader-text">Generating answer...</div>
                    </div>''',
                    unsafe_allow_html=True,
                )

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

    if submitted_question:
        st.session_state["current_result"] = None

        events: list[PipelineEvent] = []

        def show_pipeline_progress(value: float, stage: str, status: str, message: str) -> None:
            events.append(PipelineEvent(stage, status, message))
            timeline_slot.markdown(render_pipeline_timeline(events), unsafe_allow_html=True)

        try:
            result = engine.answer(submitted_question, show_pipeline_progress)
            st.session_state["history"].append(result)
            st.session_state["current_result"] = result
            st.session_state["current_question"] = None
            st.rerun()
        except WebRAGError as exc:
            pending_slot.error(str(exc))
        except Exception as exc:
            pending_slot.error(f"Unexpected error: {exc}")
