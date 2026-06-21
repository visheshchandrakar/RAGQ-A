# Direct-or-Web RAG Assistant

A local Qwen3 assistant that decides whether each question can be answered directly or needs current web research.

## Architecture

```text
Question → Qwen3 route decision
  ├─ DIRECT → local answer
  └─ WEB → SerpAPI → fetch five organic result pages
                    → extract HTML/PDF text
                    → 300-token chunks with 50-token overlap
                    → local MiniLM embeddings
                    → temporary FAISS cosine index
                    → retrieve within a 1,400-token budget
                    → grounded answer with URL citations
```

Web indexes exist for one request only. Full fetched pages, chunks, embeddings, and the FAISS index are released after the answer; chat history retains only compact citation metadata and excerpts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SERPAPI_KEY="your-key"
streamlit run app.py
```

Instead of an environment variable, the key can be stored in `.streamlit/secrets.toml`:

```toml
SERPAPI_KEY = "your-key"
```

Do not commit that file. The environment variable takes precedence when both are present. A key is only required for questions routed to web search; direct answers continue to work without it.

On first launch, **Load local models** downloads Qwen3 and `all-MiniLM-L6-v2`. Apple Silicon uses `Qwen/Qwen3-8B-MLX-4bit`; other supported platforms use the Transformers model with 4-bit quantization. Set `QWEN_MODEL_ID` to use another compatible checkpoint.

## Web behavior

- SerpAPI supplies organic result metadata and URLs.
- The first five unique HTTP(S) results are fetched concurrently.
- HTML is extracted with `trafilatura`; web-hosted PDF results use `pypdf`.
- Responses are limited to 5 MB and requests have bounded timeouts.
- The pipeline is intentionally strict: if any selected page fails to download or parse, the request stops and reports the failing URL. It does not answer from snippets or partial evidence.
- Generated web answers may use only the chunks included in their prompt and cite them with clickable source links.

## Tests

Tests use fake model, search, fetch, and embedding components, so they consume no SerpAPI quota and download no models:

```bash
pytest -q
```

They cover direct routing, safe router fallback, the complete web path, URL deduplication, missing credentials, strict fetch failure, token-aware chunking, context limits, source diversity, and citations.

## Files

```text
ragqa/
├── engine.py       # Pipeline orchestration
├── routing.py      # Direct-versus-web LLM decision
├── web.py          # SerpAPI discovery and HTML/PDF extraction
├── retrieval.py    # Chunking and temporary FAISS search
├── generation.py   # Direct and grounded generation prompts
├── llm.py          # Local Qwen loading and generation adapter
├── types.py        # Shared data contracts and errors
└── config.py       # Pipeline defaults
rag_engine.py       # Compatibility import facade
app.py              # Streamlit UI
test_engine.py      # Network-free engine tests
test_app.py         # Streamlit surface tests
requirements.txt
```
