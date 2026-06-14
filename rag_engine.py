
"""
rag_engine.py
=============
Full Self-RAG pipeline with FAISS, reranking, and ARES-style evaluation.

References
----------
Lewis et al. (2020)  – Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Gao et al. (2024)   – RAG for LLMs: A Survey (chunking, reranking, advanced retrieval)
Asai et al. (2023)  – Self-RAG: Learning to Retrieve, Generate, and Critique (ICLR 2024)
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import tiktoken
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk from a PDF."""
    text: str
    source: str          # filename
    page: int            # 1-based page number
    chunk_id: int        # global index in the FAISS store
    token_count: int = 0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    faiss_score: float   # L2 distance (lower = better)
    rerank_score: float  # cosine similarity with query (higher = better)


@dataclass
class SelfRAGDecision:
    """Mirrors the four reflection tokens from Asai et al. (2023)."""
    retrieve: bool                        # [Retrieve] – is retrieval needed?
    is_relevant: list[bool] = field(default_factory=list)   # [IsRel] per chunk
    is_supported: Optional[bool] = None  # [IsSup] – answer grounded in docs?
    is_useful: Optional[bool] = None     # [IsUse] – answer useful to user?
    critique: str = ""                   # free-text self-critique


@dataclass
class ARESScore:
    """ARES-inspired evaluation (Saad-Falcon et al. 2023 / Gao survey §5)."""
    faithfulness: float        # 0-1: answer supported by context
    answer_relevance: float    # 0-1: answer addresses the question
    context_relevance: float   # 0-1: retrieved context relevant to question
    overall: float             # macro-average
    details: dict = field(default_factory=dict)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[dict]           # [{chunk_id, source, page, excerpt}]
    retrieved: list[RetrievedChunk]
    self_rag: SelfRAGDecision
    ares: ARESScore
    latency_ms: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SelfRAGEngine:
    """
    End-to-end Self-RAG pipeline.

    Architecture (Gao et al. 2024 – Advanced RAG):
        PDF → Chunking → Embedding → FAISS index
        Query → [Retrieve?] → top-k FAISS → Rerank → [IsRel] →
        Generate + citations → [IsSup] → [IsUse] → ARES eval
    """

    EMBED_MODEL = "text-embedding-3-small"   # 1536-dim, cost-efficient
    GEN_MODEL   = "gpt-4o-mini"
    CHUNK_SIZE  = 400    # tokens  (Gao §3 recommends 256-512 for dense QA)
    CHUNK_OVERLAP = 60   # tokens  (~15% overlap to avoid boundary loss)
    TOP_K       = 6      # FAISS candidates before reranking
    RERANK_K    = 3      # after reranking, keep top-3

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.chunks: list[Chunk] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.embeddings_matrix: Optional[np.ndarray] = None  # (N, 1536)
        self._dim = 1536

    # ------------------------------------------------------------------
    # 1. PDF ingestion & chunking  (Lewis §3; Gao §3.2)
    # ------------------------------------------------------------------

    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        """Parse PDF, chunk by token count, append to FAISS index."""
        reader = PdfReader(pdf_bytes)
        new_chunks: list[Chunk] = []

        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            raw = self._clean_text(raw)
            page_chunks = self._chunk_text(raw, filename, page_num)
            new_chunks.extend(page_chunks)

        if not new_chunks:
            return 0

        # Assign global IDs
        start_id = len(self.chunks)
        for i, c in enumerate(new_chunks):
            c.chunk_id = start_id + i

        # Embed in batches of 100
        texts = [c.text for c in new_chunks]
        vecs = self._embed_batch(texts)

        # Build / extend FAISS index
        mat = np.array(vecs, dtype=np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self._dim)
            self.embeddings_matrix = mat
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, mat])

        self.index.add(mat)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        return text.strip()

    def _chunk_text(self, text: str, source: str, page: int) -> list[Chunk]:
        """Sliding-window token chunker (Gao et al. §3.2)."""
        tokens = self.enc.encode(text)
        chunks = []
        step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
        for start in range(0, len(tokens), step):
            end = start + self.CHUNK_SIZE
            token_slice = tokens[start:end]
            chunk_text = self.enc.decode(token_slice).strip()
            if len(chunk_text) < 30:
                continue
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                page=page,
                chunk_id=-1,
                token_count=len(token_slice),
            ))
        return chunks

    # ------------------------------------------------------------------
    # 2. Embedding
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in chunks of 100 (API limit safety)."""
        all_vecs = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            # Truncate to 8191 tokens max
            safe = [t[:8000] for t in batch]
            resp = self.client.embeddings.create(model=self.EMBED_MODEL, input=safe)
            all_vecs.extend([r.embedding for r in resp.data])
        return all_vecs

    def _embed_one(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.EMBED_MODEL, input=[text[:8000]])
        return np.array(resp.data[0].embedding, dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. FAISS retrieval + cosine reranking  (Lewis §2; Gao §3.3)
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if self.index is None or len(self.chunks) == 0:
            return []

        q_vec = self._embed_one(query).reshape(1, -1)

        # FAISS L2 top-k
        k = min(self.TOP_K, len(self.chunks))
        distances, indices = self.index.search(q_vec, k)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            candidates.append(RetrievedChunk(
                chunk=self.chunks[idx],
                faiss_score=float(dist),
                rerank_score=0.0,
            ))

        # Cosine reranking against query embedding (Gao §3.3 – reranking stage)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        for rc in candidates:
            doc_vec = self.embeddings_matrix[rc.chunk.chunk_id].reshape(1, -1)
            doc_norm = doc_vec / (np.linalg.norm(doc_vec) + 1e-9)
            rc.rerank_score = float(np.dot(q_norm, doc_norm.T)[0][0])

        # Sort by cosine similarity descending
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        return candidates[:self.RERANK_K]

    # ------------------------------------------------------------------
    # 4. Self-RAG reflection tokens  (Asai et al. §3)
    # ------------------------------------------------------------------

    def _self_rag_should_retrieve(self, question: str) -> bool:
        """[Retrieve] token: decide if retrieval is needed."""
        prompt = (
            "You are a Self-RAG controller. Determine if the following question "
            "requires external document retrieval to answer well, or if it can be "
            "answered from general knowledge alone.\n\n"
            f"Question: {question}\n\n"
            "Respond with JSON only: {\"retrieve\": true} or {\"retrieve\": false} "
            "with a one-sentence \"reason\"."
        )
        resp = self.client.chat.completions.create(
            model=self.GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            return bool(data.get("retrieve", True))
        except Exception:
            return True

    def _self_rag_is_relevant(self, question: str, chunks: list[RetrievedChunk]) -> list[bool]:
        """[IsRel] token: for each chunk, is it relevant?"""
        results = []
        for rc in chunks:
            prompt = (
                "You are a Self-RAG relevance judge.\n"
                f"Question: {question}\n"
                f"Passage: {rc.chunk.text[:600]}\n\n"
                "Is this passage relevant to answering the question? "
                "Respond JSON only: {\"relevant\": true/false, \"reason\": \"...\"}"
            )
            resp = self.client.chat.completions.create(
                model=self.GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            try:
                data = json.loads(resp.choices[0].message.content)
                results.append(bool(data.get("relevant", True)))
            except Exception:
                results.append(True)
        return results

    def _self_rag_critique(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> tuple[bool, bool, str]:
        """[IsSup] + [IsUse] tokens + critique text."""
        prompt = (
            "You are a Self-RAG critic evaluating a generated answer.\n\n"
            f"Question: {question}\n\n"
            f"Context passages:\n{context[:2000]}\n\n"
            f"Generated answer:\n{answer}\n\n"
            "Evaluate:\n"
            "1. is_supported: Is every factual claim in the answer grounded in the context? (true/false)\n"
            "2. is_useful: Does the answer properly address the question? (true/false)\n"
            "3. critique: A 1-2 sentence critique noting any unsupported claims or gaps.\n\n"
            "Respond JSON only: {\"is_supported\": bool, \"is_useful\": bool, \"critique\": \"...\"}"
        )
        resp = self.client.chat.completions.create(
            model=self.GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            return (
                bool(data.get("is_supported", True)),
                bool(data.get("is_useful", True)),
                data.get("critique", ""),
            )
        except Exception:
            return True, True, ""

    # ------------------------------------------------------------------
    # 5. Generation with inline citations  (Lewis §4; Gao §4.2)
    # ------------------------------------------------------------------

    def _generate_with_citations(
        self,
        question: str,
        relevant_chunks: list[RetrievedChunk],
    ) -> tuple[str, list[dict]]:
        """Generate answer with [SOURCE_N] citation markers."""

        context_parts = []
        for i, rc in enumerate(relevant_chunks):
            context_parts.append(
                f"[SOURCE_{i+1}] (File: {rc.chunk.source}, Page {rc.chunk.page}):\n"
                f"{rc.chunk.text}"
            )
        context_str = "\n\n".join(context_parts)

        system = (
            "You are a precise research assistant using Retrieval-Augmented Generation. "
            "Answer questions using ONLY the provided source passages. "
            "Cite sources inline using [SOURCE_N] notation immediately after the claim they support. "
            "If a fact is not in the sources, say so explicitly. "
            "Be concise but complete."
        )
        user = (
            f"Sources:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Provide a well-structured answer with inline citations [SOURCE_N]."
        )

        resp = self.client.chat.completions.create(
            model=self.GEN_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        answer = resp.choices[0].message.content.strip()

        # Build citation list
        citations = []
        for i, rc in enumerate(relevant_chunks):
            tag = f"SOURCE_{i+1}"
            if tag in answer:
                citations.append({
                    "tag": f"[{tag}]",
                    "source": rc.chunk.source,
                    "page": rc.chunk.page,
                    "excerpt": rc.chunk.text[:200] + "…",
                    "chunk_id": rc.chunk.chunk_id,
                    "rerank_score": round(rc.rerank_score, 4),
                })

        return answer, citations

    # ------------------------------------------------------------------
    # 6. ARES-style evaluation  (Saad-Falcon et al. 2023; Gao §5)
    # ------------------------------------------------------------------

    def _ares_evaluate(
        self,
        question: str,
        answer: str,
        retrieved: list[RetrievedChunk],
    ) -> ARESScore:
        """
        ARES measures three axes:
          - Context Relevance:  retrieved passages relevant to the question?
          - Faithfulness:       answer supported by the retrieved context?
          - Answer Relevance:   answer actually answers the question?
        """
        context_texts = "\n\n".join(
            f"[{i+1}] {rc.chunk.text[:400]}" for i, rc in enumerate(retrieved)
        )

        prompt = (
            "You are an ARES-style RAG evaluator. Score the following on three axes, "
            "each from 0.0 to 1.0 (two decimal places).\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context_texts[:3000]}\n\n"
            f"Answer: {answer[:1500]}\n\n"
            "Definitions:\n"
            "- context_relevance (0-1): Are the retrieved passages relevant to the question?\n"
            "- faithfulness (0-1): Is the answer fully supported by the retrieved context with no hallucinations?\n"
            "- answer_relevance (0-1): Does the answer directly address what was asked?\n\n"
            "Respond JSON only:\n"
            "{\n"
            "  \"context_relevance\": 0.0-1.0,\n"
            "  \"faithfulness\": 0.0-1.0,\n"
            "  \"answer_relevance\": 0.0-1.0,\n"
            "  \"reasoning\": {\"context\": \"...\", \"faith\": \"...\", \"relevance\": \"...\"}\n"
            "}"
        )
        resp = self.client.chat.completions.create(
            model=self.GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            cr  = float(data.get("context_relevance", 0.5))
            fth = float(data.get("faithfulness", 0.5))
            ar  = float(data.get("answer_relevance", 0.5))
            overall = round((cr + fth + ar) / 3, 4)
            return ARESScore(
                faithfulness=round(fth, 4),
                answer_relevance=round(ar, 4),
                context_relevance=round(cr, 4),
                overall=overall,
                details=data.get("reasoning", {}),
            )
        except Exception:
            return ARESScore(0.5, 0.5, 0.5, 0.5)

    # ------------------------------------------------------------------
    # 7. Main entry point
    # ------------------------------------------------------------------

    def answer(self, question: str) -> RAGAnswer:
        """Full Self-RAG pipeline for one question."""
        t0 = time.time()

        # [Retrieve] – do we even need documents?
        should_retrieve = self._self_rag_should_retrieve(question)

        retrieved: list[RetrievedChunk] = []
        relevant_chunks: list[RetrievedChunk] = []
        self_rag_decision = SelfRAGDecision(retrieve=should_retrieve)

        if should_retrieve and self.index is not None:
            retrieved = self.retrieve(question)

            # [IsRel] – filter irrelevant chunks
            relevance_flags = self._self_rag_is_relevant(question, retrieved)
            self_rag_decision.is_relevant = relevance_flags
            relevant_chunks = [
                rc for rc, flag in zip(retrieved, relevance_flags) if flag
            ]
            # Fallback: if all filtered out, use top-1
            if not relevant_chunks and retrieved:
                relevant_chunks = retrieved[:1]

        # Generate answer (with or without context)
        if relevant_chunks:
            answer_text, citations = self._generate_with_citations(question, relevant_chunks)
        else:
            # No retrieval / no relevant context — pure parametric answer
            resp = self.client.chat.completions.create(
                model=self.GEN_MODEL,
                messages=[
                    {"role": "system", "content": "Answer concisely from general knowledge. No documents were retrieved."},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
            )
            answer_text = resp.choices[0].message.content.strip()
            citations = []

        # [IsSup] + [IsUse] critique
        context_for_critique = "\n\n".join(rc.chunk.text for rc in relevant_chunks)
        is_sup, is_use, critique = self._self_rag_critique(question, answer_text, context_for_critique)
        self_rag_decision.is_supported = is_sup
        self_rag_decision.is_useful = is_use
        self_rag_decision.critique = critique

        # ARES evaluation
        ares = self._ares_evaluate(question, answer_text, relevant_chunks or retrieved)

        latency_ms = round((time.time() - t0) * 1000, 1)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved=retrieved,
            self_rag=self_rag_decision,
            ares=ares,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def indexed_sources(self) -> list[str]:
        return list(dict.fromkeys(c.source for c in self.chunks))
