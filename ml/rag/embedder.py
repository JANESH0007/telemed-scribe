"""
ml/rag/embedder.py — Embedding backends for the RAG layer.

  - SentenceTransformerEmbedder: all-MiniLM-L6-v2 (plan default; downloads model on first use)
  - HashingEmbedder: deterministic char n-gram vectors, no download. Used in tests and as an
    offline fallback; it is actually good at drug-name typos.

All embedders return L2-normalised float32 arrays so FAISS inner product == cosine similarity.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype("float32")


class HashingEmbedder:
    """Character n-gram hashing embedder (n=2..4)."""

    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (2, 3, 4)):
        self.dim = dim
        self.ngrams = ngrams

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            s = f" {t.lower().strip()} "
            for n in self.ngrams:
                for j in range(len(s) - n + 1):
                    h = int(hashlib.md5(s[j : j + n].encode()).hexdigest()[:8], 16)
                    out[i, h % self.dim] += 1.0
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(v.astype("float32"))


def get_default_embedder() -> Embedder:
    """MiniLM if sentence-transformers is installed and loadable, else hashing fallback."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingEmbedder()
