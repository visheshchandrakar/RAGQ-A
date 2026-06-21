"""
app.py  –  Self-RAG PDF Chatbot (Streamlit UI)
================================================
Run:  streamlit run app.py
"""

from __future__ import annotations

import io

import streamlit as st

from rag_engine import SelfRAGEngine, RAGAnswer

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main area */
.main .block-container { padding-top: 1.5rem; max-width: 1100px; }

/* Citation badge */
.citation-badge {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    border: 1px solid #c4b5fd;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
    font-family: monospace;
}

/* Score bar */
.score-bar-outer {
    background: #f3f4f6;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin-top: 4px;
}
.score-bar-inner {
    height: 10px;
    border-radius: 6px;
    transition: width 0.4s ease;
}

/* Self-RAG badge */
.selfrag-yes { background:#d1fae5; color:#065f46; border-radius:4px; padding:2px 8px; font-size:12px; }
.selfrag-no  { background:#fee2e2; color:#991b1b; border-radius:4px; padding:2px 8px; font-size:12px; }
.selfrag-warn{ background:#fef3c7; color:#92400e; border-radius:4px; padding:2px 8px; font-size:12px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────
def get_engine() -> SelfRAGEngine | None:
    return st.session_state.get("engine")

def init_engine(progress_callback=None):
    st.session_state["engine"] = SelfRAGEngine(
        progress_callback=progress_callback,
    )

if "history" not in st.session_state:
    st.session_state["history"] = []   # list[RAGAnswer]
if "ingested" not in st.session_state:
    st.session_state["ingested"] = []  # filenames


# ─── Helpers ──────────────────────────────────────────────────────────────────
def score_color(v: float) -> str:
    if v >= 0.75: return "#10b981"
    if v >= 0.50: return "#f59e0b"
    return "#ef4444"

def score_bar(label: str, value: float):
    pct = int(value * 100)
    color = score_color(value)
    st.markdown(f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
            <span>{label}</span><span style="color:{color};font-weight:600">{pct}%</span>
          </div>
          <div class="score-bar-outer">
            <div class="score-bar-inner" style="width:{pct}%;background:{color};"></div>
          </div>
        </div>
    """, unsafe_allow_html=True)

def selfrag_badge(value: bool | None, yes_label="Yes", no_label="No") -> str:
    if value is True:
        return f'<span class="selfrag-yes">✓ {yes_label}</span>'
    if value is False:
        return f'<span class="selfrag-no">✗ {no_label}</span>'
    return f'<span class="selfrag-warn">? Unknown</span>'

def render_answer(ans: RAGAnswer):
    """Render a complete RAGAnswer in the chat panel."""
    st.markdown(f"**Q: {ans.question}**")

    # Main answer text with highlighted citation tags
    rendered = ans.answer
    for cit in ans.citations:
        badge = f'<span class="citation-badge">{cit["tag"]}</span>'
        rendered = rendered.replace(cit["tag"], badge)

    st.markdown(rendered, unsafe_allow_html=True)

    # ── Self-RAG decision panel ────────────────────────────────────────────
    with st.expander("🔍 Self-RAG reflection tokens", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**[Retrieve]**")
            st.markdown(selfrag_badge(ans.self_rag.retrieve, "Yes – retrieved", "No – parametric"), unsafe_allow_html=True)
        with col2:
            st.markdown("**[IsRel]**")
            n_rel = sum(ans.self_rag.is_relevant)
            n_tot = len(ans.self_rag.is_relevant)
            st.markdown(f"{n_rel}/{n_tot} chunks relevant")
        with col3:
            st.markdown("**[IsSup]**")
            st.markdown(selfrag_badge(ans.self_rag.is_supported, "Grounded", "Unsupported"), unsafe_allow_html=True)
        with col4:
            st.markdown("**[IsUse]**")
            st.markdown(selfrag_badge(ans.self_rag.is_useful, "Useful", "Not useful"), unsafe_allow_html=True)

        if ans.self_rag.critique:
            st.info(f"💬 Critique: {ans.self_rag.critique}")

    # ── Citations ──────────────────────────────────────────────────────────
    if ans.citations:
        with st.expander(f"📎 Citations ({len(ans.citations)})", expanded=False):
            for cit in ans.citations:
                with st.container():
                    st.markdown(
                        f'<span class="citation-badge">{cit["tag"]}</span> '
                        f'**{cit["source"]}** · Page {cit["page"]} · '
                        f'Rerank score: `{cit["rerank_score"]:.3f}`',
                        unsafe_allow_html=True
                    )
                    st.caption(f'"{cit["excerpt"]}"')
                    st.divider()

    # ── Retrieved chunks detail ────────────────────────────────────────────
    if ans.retrieved:
        with st.expander(f"📦 Retrieved chunks (FAISS → reranked)", expanded=False):
            for i, rc in enumerate(ans.retrieved):
                is_rel = ans.self_rag.is_relevant[i] if i < len(ans.self_rag.is_relevant) else None
                rel_str = "✅ Relevant" if is_rel else ("❌ Filtered" if is_rel is False else "")
                st.markdown(
                    f"**Chunk #{rc.chunk.chunk_id}** · {rc.chunk.source} · Page {rc.chunk.page}  \n"
                    f"FAISS L2: `{rc.faiss_score:.3f}` · Cosine rerank: `{rc.rerank_score:.4f}` · {rel_str}"
                )
                st.caption(rc.chunk.text)
                st.divider()

    # ── ARES evaluation ────────────────────────────────────────────────────
    with st.expander("📊 ARES Evaluation Metrics", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faithfulness",      f"{ans.ares.faithfulness*100:.0f}%")
        c2.metric("Answer Relevance",  f"{ans.ares.answer_relevance*100:.0f}%")
        c3.metric("Context Relevance", f"{ans.ares.context_relevance*100:.0f}%")
        c4.metric("Overall ARES",      f"{ans.ares.overall*100:.0f}%")

        score_bar("Faithfulness",      ans.ares.faithfulness)
        score_bar("Answer Relevance",  ans.ares.answer_relevance)
        score_bar("Context Relevance", ans.ares.context_relevance)

        if ans.ares.details:
            st.caption("**Reasoning:**")
            for k, v in ans.ares.details.items():
                st.caption(f"• *{k}*: {v}")

    st.caption(f"⏱ Latency: {ans.latency_ms:.0f} ms")
    st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📚 Self-RAG Chatbot")
    st.caption("Lewis (2020) · Gao (2024) · Asai (2023)")

    st.markdown("---")

    if get_engine() is None:
        st.caption("Runs locally with Qwen3-8B 4-bit. No API key is required.")
        if st.button("Load local models", type="primary"):
            progress = st.progress(0.0, text="Preparing local models…")

            def show_progress(value: float, message: str):
                progress.progress(max(0.0, min(value, 1.0)), text=message)

            init_engine(progress_callback=show_progress)
            progress.progress(1.0, text="Local models are ready ✓")
            st.rerun()
    else:
        st.success("Local models loaded ✓")

    st.markdown("---")

    # PDF upload
    st.subheader("Upload PDFs")
    uploaded = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded and get_engine() is not None:
        engine = get_engine()
        for f in uploaded:
            if f.name not in st.session_state["ingested"]:
                with st.spinner(f"Indexing {f.name}…"):
                    n = engine.ingest_pdf(io.BytesIO(f.read()), f.name)
                st.session_state["ingested"].append(f.name)
                st.success(f"✓ {f.name} → {n} chunks")

    if st.session_state["ingested"]:
        st.markdown("**Indexed documents:**")
        for name in st.session_state["ingested"]:
            st.markdown(f"- {name}")
        engine = get_engine()
        if engine:
            st.caption(f"Total chunks in FAISS: **{engine.num_chunks}**")

    st.markdown("---")

    # Settings info
    with st.expander("Pipeline settings"):
        e = get_engine()
        if e:
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Embedding model | `{e.EMBED_MODEL}` |
| Generation model | `{e.GEN_MODEL}` |
| Chunk size | {e.CHUNK_SIZE} tokens |
| Chunk overlap | {e.CHUNK_OVERLAP} tokens |
| FAISS top-k | {e.TOP_K} |
| Rerank top-k | {e.RERANK_K} |
| FAISS index | `IndexFlatL2` |
| Reranker | Cosine (local embeddings) |
""")
        else:
            st.info("Load the local models to see settings.")

    if st.button("🗑 Clear history"):
        st.session_state["history"] = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Self-RAG PDF Chatbot")
st.caption(
    "Implements: FAISS retrieval · cosine reranking · "
    "Self-RAG reflection tokens [Retrieve / IsRel / IsSup / IsUse] · "
    "inline citations · ARES evaluation"
)

# ── Architecture diagram ──────────────────────────────────────────────────────
with st.expander("📐 Pipeline architecture", expanded=False):
    st.markdown("""
```
PDF(s)
  │
  ├─ [Chunking]  sliding window, 400 tok / 60 tok overlap  (Gao §3.2)
  │
  ├─ [Embedding] all-MiniLM-L6-v2 → 384-dim vectors
  │
  └─ [FAISS]     IndexFlatL2 — exact L2 nearest-neighbor index

Query
  │
  ├─ [Retrieve?]  Self-RAG [Retrieve] token — skip retrieval if not needed
  │
  ├─ [FAISS search]  top-6 candidates by L2
  │
  ├─ [Rerank]     cosine similarity → keep top-3       (Gao §3.3 Advanced RAG)
  │
  ├─ [IsRel]      Self-RAG relevance filter per chunk  (Asai §3)
  │
  ├─ [Generate]   Qwen3-8B 4-bit + context + [SOURCE_N] citations     (Lewis §4)
  │
  ├─ [IsSup]      Self-RAG groundedness check          (Asai §3)
  ├─ [IsUse]      Self-RAG utility check               (Asai §3)
  │
  └─ [ARES]       Faithfulness / Answer Relevance / Context Relevance  (Gao §5)
```
""")

st.markdown("---")

# ── Render history ─────────────────────────────────────────────────────────────
# ── Question input ─────────────────────────────────────────────────────────────
engine = get_engine()

if engine is None:
    st.warning("⬅ Load the local models in the sidebar to get started.")
elif engine.num_chunks == 0:
    st.info("⬅ Upload one or more PDFs to begin. The engine is ready.")
else:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a question about your documents",
            placeholder="e.g. What are the main findings of the study?",
            height=80,
        )
        submitted = st.form_submit_button("Ask ↗", type="primary")

    if submitted and question.strip():
        progress = st.progress(0.0, text="Starting Self-RAG pipeline…")

        def show_pipeline_progress(value: float, message: str):
            progress.progress(max(0.0, min(value, 1.0)), text=message)

        try:
            ans = engine.answer(
                question.strip(),
                progress_callback=show_pipeline_progress,
            )
            st.session_state["history"].append(ans)
            st.rerun()
        except Exception as ex:
            progress.empty()
            st.error(f"Error: {ex}")

# ── Render history ───────────────────────────────────────────────────────────────────────────
for ans in st.session_state["history"]:
    render_answer(ans)
