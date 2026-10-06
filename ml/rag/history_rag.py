"""
ml/rag/history_rag.py — Patient history RAG (Phase 5, Person C)
===============================================================
patient ID + new symptoms in  ->  top-k most similar PAST visits of that patient out.

Per-patient FAISS index built on demand from finalized visits (Store.history_for_patient), so a
patient's history can never leak into another patient's results. Visit vectors are cached by
(consultation_id, text) so only new visits are embedded.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Optional

import faiss
import numpy as np
from pydantic import BaseModel

from ml.rag.embedder import Embedder, get_default_embedder

_FIELDS = ("symptoms", "diagnosis", "prescriptions", "follow_up")


class HistoryHit(BaseModel):
    consultation_id: str
    date: Optional[datetime] = None
    score: float
    symptoms: str = ""
    diagnosis: str = ""
    prescriptions: str = ""
    follow_up: str = ""

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


def visit_text(v: dict[str, Any]) -> str:
    return " | ".join(f"{k.replace('_', ' ')}: {v.get(k, '')}" for k in _FIELDS if v.get(k))


class PatientHistoryRAG:
    def __init__(self, embedder: Optional[Embedder] = None, min_score: float = 0.15):
        self.embedder = embedder or get_default_embedder()
        self.min_score = min_score
        self._vec_cache: dict[tuple[str, str], np.ndarray] = {}

    def _embed_visits(self, visits: list[dict]) -> np.ndarray:
        texts = [visit_text(v) for v in visits]
        keys = [(v["consultation_id"], hashlib.md5(t.encode()).hexdigest()) for v, t in zip(visits, texts)]
        missing = [i for i, k in enumerate(keys) if k not in self._vec_cache]
        if missing:
            vecs = self.embedder.encode([texts[i] for i in missing])
            for i, vec in zip(missing, vecs):
                self._vec_cache[keys[i]] = vec
        return np.vstack([self._vec_cache[k] for k in keys])

    def retrieve_from_visits(self, visits: list[dict], query: str, k: int = 3) -> list[HistoryHit]:
        """Rank the given visits (already scoped to ONE patient) against the query."""
        visits = [v for v in visits if visit_text(v)]
        if not visits or not query.strip():
            return []
        index = faiss.IndexFlatIP(self.embedder.dim)
        index.add(self._embed_visits(visits))
        scores, ids = index.search(self.embedder.encode([query]), min(k, len(visits)))
        hits = []
        for s, j in zip(scores[0], ids[0]):
            if j < 0 or s < self.min_score:
                continue
            v = visits[j]
            hits.append(HistoryHit(
                consultation_id=v["consultation_id"], date=v.get("date"), score=round(float(s), 4),
                **{f: v.get(f, "") for f in _FIELDS},
            ))
        return hits

    def retrieve(self, store, patient_id: str, symptoms: str, diagnosis: str = "",
                 exclude: Optional[str] = None, k: int = 3) -> list[HistoryHit]:
        """store = db.Store. Only finalized past visits of this patient are searched."""
        visits = store.history_for_patient(patient_id, exclude=exclude)
        return self.retrieve_from_visits(visits, f"symptoms: {symptoms} | diagnosis: {diagnosis}".strip(" |"), k)
