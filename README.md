<img width="1920" height="1080" alt="Screenshot 2026-06-14 at 19 47 41 (2)" src="https://github.com/user-attachments/assets/2e0920dc-d18b-430a-b90e-22c59b7aaa1e" />
<img width="1920" height="1080" alt="Screenshot 2026-06-14 at 19 47 32 (2)" src="https://github.com/user-attachments/assets/4ad1f973-830a-4e41-b84f-ddd9f52078f8" />
<img width="1470" height="956" alt="Screenshot 2026-06-14 at 19 47 32" src="https://github.com/user-attachments/assets/c647ef5a-a0f2-4178-97a6-166c04bd618f" />
# Self-RAG PDF Chatbot

**Full pipeline: PDF → FAISS → Self-RAG → GPT-4o-mini → ARES evaluation**

Implements the core ideas from:
- **Lewis et al. (2020)** — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Gao et al. (2024)** — RAG for Large Language Models: A Survey (Advanced RAG: reranking, chunking strategies, evaluation)
- **Asai et al. (2023)** — Self-RAG: Learning to Retrieve, Generate, and Critique (ICLR 2024)

---

## Architecture

```
PDF(s)
  │
  ├─ [Chunking]   Sliding window, 400 tok / 60 tok overlap        ← Gao §3.2
  ├─ [Embedding]  text-embedding-3-small (1536-dim)
  └─ [FAISS]      IndexFlatL2 — exact nearest-neighbor index       ← Lewis §2

Query
  │
  ├─ [Retrieve?]  Self-RAG [Retrieve] token                        ← Asai §3
  ├─ [FAISS]      top-6 candidates by L2 distance
  ├─ [Rerank]     cosine similarity → keep top-3                   ← Gao §3.3
  ├─ [IsRel]      Self-RAG relevance filter (per chunk)            ← Asai §3
  ├─ [Generate]   GPT-4o-mini + [SOURCE_N] citation markers        ← Lewis §4
  ├─ [IsSup]      Self-RAG groundedness check                      ← Asai §3
  ├─ [IsUse]      Self-RAG utility check                           ← Asai §3
  └─ [ARES]       Faithfulness / Answer Relevance / Context Rel.   ← Gao §5
```

---

## Setup

```bash
# 1. Clone / copy files into a directory
cd pdf_rag_selfrag

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY=sk-...    # Windows: set OPENAI_API_KEY=sk-...
```

---

## Run

### Streamlit UI (recommended)
```bash
streamlit run app.py
```
Open `http://localhost:8501`. Use the sidebar to enter your API key and upload PDFs.

### CLI smoke test (no PDF needed)
```bash
python test_engine.py
```
Injects 5 RAG-topic paragraphs directly and runs a full pipeline query.

---

## Features

| Feature | Implementation |
|---|---|
| PDF ingestion | `pypdf` — multi-page, multi-file |
| Chunking | Sliding window (400 tok, 60 tok overlap) via `tiktoken` |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector store | `faiss-cpu` `IndexFlatL2` — exact L2 search |
| Reranking | Cosine similarity re-sort of top-6 FAISS hits → top-3 |
| **Self-RAG [Retrieve]** | LLM decides if retrieval is needed at all |
| **Self-RAG [IsRel]** | Per-chunk relevance filtering |
| **Self-RAG [IsSup]** | Groundedness check on generated answer |
| **Self-RAG [IsUse]** | Utility check on generated answer |
| Citations | Inline `[SOURCE_N]` markers → file + page + excerpt |
| **ARES Faithfulness** | LLM judge: answer supported by context? |
| **ARES Answer Relevance** | LLM judge: answer addresses the question? |
| **ARES Context Relevance** | LLM judge: retrieved context relevant? |
| Generation | `gpt-4o-mini` — cost-efficient |

---

## Cost estimate

| Call type | Per query |
|---|---|
| Query embedding | ~$0.00002 |
| Self-RAG [Retrieve] | ~$0.00005 |
| Self-RAG [IsRel] × 3 | ~$0.00015 |
| Generation | ~$0.0003 |
| Self-RAG critique | ~$0.0001 |
| ARES evaluation | ~$0.0002 |
| **Total per query** | **~$0.0008** |

---

## File structure

```
pdf_rag_selfrag/
├── rag_engine.py    # Core Self-RAG engine (chunking, FAISS, reranking, generation, ARES)
├── app.py           # Streamlit UI
├── test_engine.py   # CLI smoke test
├── requirements.txt
└── README.md
```

---

## Extending

**Swap the vector store** — replace `faiss.IndexFlatL2` with `IndexIVFFlat` for large corpora:
```python
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist=100)
index.train(training_vectors)
```

**Add HyDE** (Hypothetical Document Embeddings, Gao §3.1) — generate a hypothetical answer and embed that instead of the raw query:
```python
def hyde_embed(self, question: str) -> np.ndarray:
    hyp = self.client.chat.completions.create(...)  # generate hypothetical doc
    return self._embed_one(hyp)
```

**Multi-hop retrieval** — iterate: generate intermediate answer → extract sub-question → retrieve again.

**Persistent index** — save/load FAISS index and chunks:
```python
faiss.write_index(self.index, "index.faiss")
# restore: self.index = faiss.read_index("index.faiss")
```
