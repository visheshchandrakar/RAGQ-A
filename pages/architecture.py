"""Architecture page for the Direct-or-Web RAG Assistant."""

from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Architecture | Direct-or-Web RAG Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui_markup_path = Path(__file__).resolve().parent.parent / "app_ui.html"
st.markdown(ui_markup_path.read_text(encoding="utf-8"), unsafe_allow_html=True)

ARCH_STYLES = [
    ("#6366f1", "🧱 Layered", "Presentation, orchestration, domain services, and infrastructure adapters stay separated."),
    ("#10b981", "🔗 Pipeline", "Each RAG stage's output becomes the next stage's input."),
    ("#f59e0b", "🧪 Dependency injection", "The engine accepts alternative LLMs, embedders, or fetchers — tests swap these in."),
]

with st.sidebar:
    st.markdown(
        '<nav class="app-navigation">'
        '<a href="/">💬 Assistant</a>'
        '<a class="active" href="/architecture">🏗️ Architecture</a>'
        "</nav>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("🏛️ Architectural style")
    legend_items = "".join(
        f'<div class="arch-sidebar-legend-item" style="--legend-color:{color}">'
        f'<div class="arch-sidebar-legend-title">{title}</div>'
        f'<div class="arch-sidebar-legend-desc">{desc}</div>'
        f"</div>"
        for color, title, desc in ARCH_STYLES
    )
    st.markdown(
        f'<div class="arch-sidebar-legend">{legend_items}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("🎓 For discussion")
    for question in [
        "Should routing be an LLM call, a classifier, or rules?",
        "Does cosine similarity imply credibility?",
        "Same model generates *and* grades the answer — is that sound?",
    ]:
        st.markdown(f"- {question}")

st.markdown(
    '<div class="arch-hero">'
    '<div class="arch-hero-title">🏗️ Anatomy of a Direct-or-Web RAG Assistant</div>'
    '<div class="arch-hero-sub">A teaching-sized pipeline: a local LLM decides whether to answer from its own '
    "knowledge or to search the web, retrieve, and ground its answer in evidence — entirely on your own machine.</div>"
    "</div>",
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)
metrics = [
    ("300 tok", "Chunk size"),
    ("50 tok", "Overlap"),
    ("6", "Max chunks"),
    ("1,400 tok", "Context budget"),
    ("2", "Max per source"),
]
for col, (value, label) in zip(metric_cols, metrics):
    col.markdown(
        f'<div class="arch-metric-chip">'
        f'<div class="arch-metric-value">{value}</div>'
        f'<div class="arch-metric-label">{label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("### 🧩 Component map")
st.caption("Hover a card — components are grouped by the layer they live in.")

LAYERS = [
    (
        "#3b82f6",
        "rgba(59, 130, 246, 0.05)",
        "🖥️ Presentation",
        [
            ("📜", "app.py", "Streamlit UI", "Loads models on demand, accepts questions, renders answers, sources, and the pipeline trace."),
            ("🎨", "app_ui.html", "Shared styling", "CSS and small JS shared across pages."),
            ("🏗️", "pages/architecture.py", "Architecture page", "This page — a visual tour of the system's structure."),
        ],
    ),
    (
        "#8b5cf6",
        "rgba(139, 92, 246, 0.05)",
        "🧭 Application orchestration",
        [
            ("🧠", "ragqa/engine.py", "WebRAGEngine", "Executes the use case end-to-end and assembles the final RAGAnswer."),
        ],
    ),
    (
        "#10b981",
        "rgba(16, 185, 129, 0.05)",
        "⚙️ Domain — pipeline services",
        [
            ("🚦", "ragqa/routing.py", "QueryRouter", "Decides direct vs. web and produces a search query."),
            ("🔍", "ragqa/retrieval.py", "TemporaryFaissRetriever", "Chunks, embeds, indexes, and retrieves relevant passages."),
            ("✍️", "ragqa/generation.py", "AnswerGenerator", "Produces a direct or evidence-grounded answer with source tags."),
            ("📊", "ragqa/evaluation.py", "ARESEvaluator", "Scores faithfulness, context relevance, and answer relevance."),
        ],
    ),
    (
        "#f59e0b",
        "rgba(245, 158, 11, 0.05)",
        "🔌 Infrastructure — external & model adapters",
        [
            ("🤖", "ragqa/llm.py", "LocalQwen3", "Loads the local model backend and exposes a single generate() call."),
            ("🌐", "ragqa/web.py", "SerpApiSearch", "Searches, validates URLs, and deduplicates results."),
            ("📥", "ragqa/web.py", "WebPageFetcher", "Fetches pages concurrently and extracts HTML/PDF text."),
            ("🗂️", "in-memory", "FAISS index", "Exact cosine-similarity search over the request's chunks."),
        ],
    ),
]

layer_blocks = []
for index, (color, bg, layer_title, cards) in enumerate(LAYERS):
    card_html = "".join(
        f'<figure class="arch-card" style="--layer-color:{color}">'
        f'<div class="arch-card-title"><span class="arch-card-icon">{icon}</span>{name}</div>'
        f'<div class="arch-card-path">{path}</div>'
        f'<figcaption class="arch-card-desc">{desc}</figcaption>'
        f"</figure>"
        for icon, path, name, desc in cards
    )
    layer_blocks.append(
        f'<div class="arch-layer" style="--layer-color:{color}; --layer-bg:{bg}">'
        f'<div class="arch-layer-title">{layer_title}</div>'
        f'<div class="arch-layer-cards">{card_html}</div>'
        f"</div>"
    )
    if index < len(LAYERS) - 1:
        layer_blocks.append('<div class="arch-arrow">↓</div>')

st.markdown(f'<div class="arch-figure-grid">{"".join(layer_blocks)}</div>', unsafe_allow_html=True)

st.markdown("### 🔀 Pick a route")
direct_tab, web_tab = st.tabs(["⚡ Direct route", "🌐 Web RAG route"])

DIRECT_STEPS = [
    ("#3b82f6", "Submit question", "User asks something in the Streamlit UI."),
    ("#6366f1", "Route decision", "QueryRouter asks the local LLM: direct or web?"),
    ("#10b981", "Generate from memory", "The LLM answers from stable general knowledge — no network call."),
    ("#f59e0b", "Display", "UI shows the answer with no sources and no ARES score."),
]

WEB_STEPS = [
    ("#3b82f6", "Submit question", "User asks something in the Streamlit UI."),
    ("#6366f1", "Route + rewrite", "Router picks 'web' and rewrites the query for search."),
    ("#06b6d4", "Search", "SerpAPI returns ranked organic results."),
    ("#0ea5e9", "Fetch pages", "Pages are downloaded concurrently; HTML/PDF text is extracted."),
    ("#10b981", "Chunk + embed", "Text is split into 300-token windows and embedded with MiniLM."),
    ("#22c55e", "Retrieve", "FAISS finds the closest, source-diverse chunks to the original question."),
    ("#f59e0b", "Grounded generation", "The LLM answers using only retrieved passages, citing [SOURCE_N] tags."),
    ("#ef4444", "ARES scoring", "The same LLM grades faithfulness, relevance, and context quality."),
    ("#db2777", "Display", "UI shows the answer, clickable citations, and the full pipeline trace."),
]


def render_route_steps(steps: list[tuple[str, str, str]]) -> str:
    rows = "".join(
        f'<div class="route-step">'
        f'<div class="route-step-num" style="--step-color:{color}">{i}</div>'
        f'<div class="route-step-body">'
        f'<div class="route-step-title">{title}</div>'
        f'<div class="route-step-desc">{desc}</div>'
        f"</div></div>"
        for i, (color, title, desc) in enumerate(steps, start=1)
    )
    return f'<div class="route-step-list">{rows}</div>'


with direct_tab:
    st.markdown(render_route_steps(DIRECT_STEPS), unsafe_allow_html=True)
    st.caption("Cheaper and faster — but limited to what the model already knows.")

with web_tab:
    st.markdown(render_route_steps(WEB_STEPS), unsafe_allow_html=True)
    st.caption("Current and grounded — but slower, and only as trustworthy as the pages it reads.")

st.markdown("### ⚖️ Design trade-offs")

TRADEOFFS = [
    ("🏠 Local generation", "Prompts and retrieved text never leave the machine; no hosted LLM cost.", "Model download size, memory use, and lower capability than some hosted models."),
    ("🧠 LLM-based routing", "Reasons about freshness/intent without a big rule set.", "Adds an inference call and can be inconsistent — mitigated by a web fallback on bad JSON."),
    ("📦 Request-local exact index", "Simple lifecycle, fresh evidence, no vector DB service.", "Repeated downloads/embeddings per question; linear-time exact search."),
    ("🌍 Open-web retrieval", "Current and broad information.", "Pages can be blocked, biased, or malicious — similarity isn't a trust signal."),
    ("🏷️ Prompt-based grounding", "A clear, inspectable evidence-to-answer mechanism.", "Instructions aren't guarantees — the model can omit or misattach a citation."),
]

cols = st.columns(2)
for index, (title, benefit, cost) in enumerate(TRADEOFFS):
    with cols[index % 2]:
        st.markdown(
            f'<div class="tradeoff-card">'
            f'<div class="tradeoff-title">{title}</div>'
            f'<div class="tradeoff-row"><span class="tradeoff-chip benefit">Benefit</span>'
            f'<span class="tradeoff-text">{benefit}</span></div>'
            f'<div class="tradeoff-row"><span class="tradeoff-chip cost">Cost</span>'
            f'<span class="tradeoff-text">{cost}</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.caption("Full write-up, sequence diagrams, and data contracts live in ARCHITECTURE.md.")
