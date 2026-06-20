"""
test_engine.py
==============
Command-line smoke-test for SelfRAGEngine.
Creates a tiny synthetic PDF-like text corpus in memory and runs one query
through the full pipeline.

Usage:
    python test_engine.py
"""

import io

# We'll inject a fake PDF using pypdf-compatible bytes via reportlab if available,
# or test with direct chunk injection.
from rag_engine import SelfRAGEngine, Chunk
import numpy as np


def inject_test_chunks(engine: SelfRAGEngine):
    """Bypass PDF parsing — inject known chunks directly for testing."""
    texts = [
        (
            "Retrieval-Augmented Generation (RAG) combines a parametric language model "
            "with a non-parametric retrieval component. Lewis et al. (2020) showed that "
            "RAG significantly outperforms purely parametric models on open-domain QA tasks "
            "by grounding generation in retrieved documents."
        ),
        (
            "Gao et al. (2024) survey identifies three generations of RAG systems: "
            "Naive RAG (retrieve once, prepend, generate), Advanced RAG (query rewriting, "
            "reranking, hybrid search), and Modular RAG (adaptive retrieval scheduling, "
            "plug-in components, retrieval-augmented fine-tuning)."
        ),
        (
            "Self-RAG (Asai et al. 2023) extends RAG with four reflection tokens: "
            "[Retrieve] decides if retrieval is necessary, [IsRel] filters irrelevant "
            "passages, [IsSup] verifies that the answer is grounded in retrieved context, "
            "and [IsUse] checks that the response is helpful to the user."
        ),
        (
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity "
            "search over dense vectors. IndexFlatL2 performs exact L2 nearest-neighbor "
            "search. For large corpora, IndexIVFFlat with nprobe tuning provides a "
            "speed-accuracy trade-off."
        ),
        (
            "ARES (Automated RAG Evaluation System) measures three axes: "
            "Context Relevance (are retrieved passages relevant?), "
            "Faithfulness (is the answer supported by the context?), and "
            "Answer Relevance (does the answer address the question?). "
            "Each axis is scored 0-1 using an LLM judge."
        ),
    ]

    # Embed them
    vecs = engine._embed_batch(texts)

    import faiss
    mat = np.array(vecs, dtype=np.float32)
    engine.index = faiss.IndexFlatL2(engine._dim)
    engine.index.add(mat)
    engine.embeddings_matrix = mat

    for i, t in enumerate(texts):
        engine.chunks.append(Chunk(
            text=t,
            source="test_corpus.txt",
            page=i + 1,
            chunk_id=i,
            token_count=len(engine.enc.encode(t)),
        ))

    print(f"  Injected {len(texts)} test chunks into FAISS index.")


def main():
    print("=" * 60)
    print("Self-RAG Engine — CLI Smoke Test")
    print("=" * 60)

    engine = SelfRAGEngine()

    print("\n[1] Injecting test chunks…")
    inject_test_chunks(engine)
    print(f"     Chunks in index: {engine.num_chunks}")

    question = "What are the Self-RAG reflection tokens and what does each one do?"
    print(f"\n[2] Running full pipeline for question:\n    '{question}'\n")

    ans = engine.answer(question)

    print("─" * 60)
    print("ANSWER:")
    print(ans.answer)
    print()

    print("─" * 60)
    print("SELF-RAG DECISIONS:")
    print(f"  [Retrieve]  : {ans.self_rag.retrieve}")
    print(f"  [IsRel]     : {ans.self_rag.is_relevant}")
    print(f"  [IsSup]     : {ans.self_rag.is_supported}")
    print(f"  [IsUse]     : {ans.self_rag.is_useful}")
    print(f"  Critique    : {ans.self_rag.critique}")

    print()
    print("CITATIONS:")
    for c in ans.citations:
        print(f"  {c['tag']} → {c['source']} p.{c['page']}  (rerank={c['rerank_score']:.4f})")
        print(f"    \"{c['excerpt'][:80]}...\"")

    print()
    print("ARES SCORES:")
    print(f"  Faithfulness      : {ans.ares.faithfulness:.3f}")
    print(f"  Answer Relevance  : {ans.ares.answer_relevance:.3f}")
    print(f"  Context Relevance : {ans.ares.context_relevance:.3f}")
    print(f"  Overall           : {ans.ares.overall:.3f}")

    print()
    print(f"Latency: {ans.latency_ms:.0f} ms")
    print("=" * 60)
    print("Smoke test complete ✓")


if __name__ == "__main__":
    main()
