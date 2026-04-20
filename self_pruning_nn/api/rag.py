from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class RagDoc:
    source: str
    text: str


def _iter_text_files(root: Path) -> List[Path]:
    exts = {".md", ".txt"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        chunks.append(text[i:j].strip())
        if j == len(text):
            break
        i = max(0, j - overlap)
    return [c for c in chunks if c]


class SimpleRagIndex:
    """
    Minimal RAG index:
      - sentence-transformers embeddings
      - FAISS vector index
      - optional OpenAI-compatible chat completion for answer synthesis
    """

    def __init__(self, docs_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.docs_dir = Path(docs_dir)
        self.embedding_model_name = embedding_model

        self._embedder = None
        self._index = None
        self._docs: List[RagDoc] = []
        self._emb_dim: Optional[int] = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def build(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"docs_dir not found: {self.docs_dir}")

        files = _iter_text_files(self.docs_dir)
        docs: List[RagDoc] = []
        for fp in files:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            for chunk in _chunk_text(raw):
                docs.append(RagDoc(source=str(fp.relative_to(self.docs_dir)), text=chunk))

        if not docs:
            raise ValueError(f"No .md/.txt content found under {self.docs_dir}")

        embedder = self._get_embedder()
        texts = [d.text for d in docs]
        embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)

        import faiss

        dim = embs.shape[1]
        self._emb_dim = dim
        index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized embeddings
        index.add(embs)

        self._index = index
        self._docs = docs

    def is_ready(self) -> bool:
        return self._index is not None and len(self._docs) > 0

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[RagDoc, float]]:
        if not self.is_ready():
            self.build()

        embedder = self._get_embedder()
        q = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype=np.float32)

        scores, idxs = self._index.search(q, top_k)
        out: List[Tuple[RagDoc, float]] = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0 or i >= len(self._docs):
                continue
            out.append((self._docs[i], float(s)))
        return out

    async def answer(self, question: str, top_k: int = 4, use_llm: bool = False) -> Tuple[str, Optional[str], List[Tuple[RagDoc, float]]]:
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n".join([f"[{d.source}] {d.text}" for d, _ in retrieved])

        if not use_llm:
            answer = (
                "Retrieval-only mode (no LLM configured).\n\n"
                "Top relevant context:\n\n"
                f"{context}"
            )
            return answer, None, retrieved

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            answer = (
                "LLM mode requested but `OPENAI_API_KEY` is not set; falling back to retrieval-only.\n\n"
                f"{context}"
            )
            return answer, None, retrieved

        from openai import AsyncOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer using only the provided context. If missing, say you don't know."},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
        return answer, model, retrieved

