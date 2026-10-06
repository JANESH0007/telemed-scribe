"""
ml/rag/medication_rag.py — Medication reference RAG (Phase 4, Person C)
=======================================================================
medication dict/string in  ->  matched reference row + anomaly flags out.

Flow per medication:
  1. Resolve the drug name -> reference row
       a. exact alias/name lookup (deterministic; covers brands like Crocin/Dolo)
       b. FAISS nearest-neighbour over name+alias terms (handles STT typos, e.g. "paracetmol")
       c. confidence gate: embedding score AND fuzzy string ratio must agree, else flagged
  2. Compare dose / daily total / frequency to the reference ranges
  3. Return flags with severity (critical | moderate | minor) and the reference row attached

The reference CSV (ml/rag/data/medications.csv) is a SEED dataset for a student project: general adult
values, not authoritative. A clinician/pharmacist must verify before any real use.
"""

from __future__ import annotations

import csv
import difflib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from ml.rag.embedder import Embedder, get_default_embedder
from ml.rag.parsing import parse_medication, parse_prescriptions
from ml.rag.schemas import (
    ExtractedMedication,
    FlagType,
    MedicationCheckResult,
    MedicationEntry,
    MedicationFlag,
)

logger = logging.getLogger(__name__)

DEFAULT_CSV = Path(__file__).parent / "data" / "medications.csv"
_UNIT_TO_MG = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}


def _norm(s: str) -> str:
    s = re.sub(r"\b(?:tab(?:let)?s?|cap(?:sule)?s?|syrup|inhaler|puffer|hcl|sr|er|xr)\b\.?", " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9% ]", " ", s)).strip()


def _num(x: str) -> Optional[float]:
    x = (x or "").strip()
    return float(x) if x else None


def load_reference(csv_path: str | Path = DEFAULT_CSV) -> list[MedicationEntry]:
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            entries.append(
                MedicationEntry(
                    id=r["id"], name=r["name"],
                    aliases=[a.strip() for a in r["aliases"].split("|") if a.strip()],
                    drug_class=r["drug_class"], dose_unit=r["dose_unit"] or None,
                    dose_min=_num(r["dose_min"]), dose_max=_num(r["dose_max"]),
                    max_daily=_num(r["max_daily"]), max_freq_per_day=_num(r["max_freq_per_day"]),
                    route=r["route"], usual_use=r["usual_use"], notes=r["notes"],
                    high_alert=r["high_alert"].strip() == "1",
                )
            )
    return entries


def _convert(value: float, src: str, dst: str) -> Optional[float]:
    src, dst = src.lower(), dst.lower()
    if src == dst:
        return value
    if src in _UNIT_TO_MG and dst in _UNIT_TO_MG:
        return value * _UNIT_TO_MG[src] / _UNIT_TO_MG[dst]
    return None


class MedicationRAG:
    def __init__(
        self,
        entries: list[MedicationEntry],
        embedder: Embedder,
        match_threshold: float = 0.80,
        low_threshold: float = 0.55,
        fuzzy_threshold: float = 0.75,
    ):
        self.entries = entries
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.low_threshold = low_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self._terms: list[str] = []
        self._term_entry: list[int] = []
        for i, e in enumerate(entries):
            for t in [e.name, *e.aliases]:
                self._terms.append(_norm(t))
                self._term_entry.append(i)
        self._exact = dict(zip(self._terms, self._term_entry))
        self.index: faiss.Index = faiss.IndexFlatIP(embedder.dim)
        self.index.add(embedder.encode(self._terms))

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, csv_path: str | Path = DEFAULT_CSV, embedder: Optional[Embedder] = None, **kw) -> "MedicationRAG":
        return cls(load_reference(csv_path), embedder or get_default_embedder(), **kw)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "medication.faiss"))
        (d / "terms.json").write_text(
            json.dumps({"terms": self._terms, "term_entry": self._term_entry, "dim": self.embedder.dim}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, csv_path: str | Path = DEFAULT_CSV, embedder: Optional[Embedder] = None, **kw) -> "MedicationRAG":
        d = Path(directory)
        meta = json.loads((d / "terms.json").read_text(encoding="utf-8"))
        embedder = embedder or get_default_embedder()
        if meta["dim"] != embedder.dim:
            raise ValueError(f"Index dim {meta['dim']} != embedder dim {embedder.dim}; rebuild the index.")
        obj = cls.__new__(cls)
        obj.entries = load_reference(csv_path)
        obj.embedder = embedder
        obj.match_threshold = kw.get("match_threshold", 0.80)
        obj.low_threshold = kw.get("low_threshold", 0.55)
        obj.fuzzy_threshold = kw.get("fuzzy_threshold", 0.75)
        obj._terms, obj._term_entry = meta["terms"], meta["term_entry"]
        obj._exact = dict(zip(obj._terms, obj._term_entry))
        obj.index = faiss.read_index(str(d / "medication.faiss"))
        return obj

    # ------------------------------------------------------------------ name resolution
    def lookup(self, name: str, k: int = 5) -> tuple[Optional[MedicationEntry], float, str, list[str]]:
        """Return (entry, score, method, candidates). method: exact | embedding | low_confidence | none."""
        q = _norm(name)
        if not q:
            return None, 0.0, "none", []
        if q in self._exact:
            return self.entries[self._exact[q]], 1.0, "exact", []

        scores, ids = self.index.search(self.embedder.encode([q]), min(k * 3, len(self._terms)))
        best: dict[int, tuple[float, str]] = {}
        for s, j in zip(scores[0], ids[0]):
            if j < 0:
                continue
            ei = self._term_entry[j]
            if ei not in best or s > best[ei][0]:
                best[ei] = (float(s), self._terms[j])
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        if not ranked:
            return None, 0.0, "none", []
        ei, (score, term) = ranked[0]
        fuzzy = difflib.SequenceMatcher(None, q, term).ratio()
        cands = [self.entries[i].name for i, _ in ranked[:3]]
        # Confident only when vectors AND spelling agree (typo of a known drug), never on semantics alone.
        if (score >= self.match_threshold and fuzzy >= self.fuzzy_threshold) or (
            fuzzy >= 0.85 and score >= self.low_threshold
        ):
            return self.entries[ei], score, "embedding", []
        if score >= self.low_threshold or fuzzy >= 0.6:
            return self.entries[ei], score, "low_confidence", cands
        return None, score, "none", cands

    # ------------------------------------------------------------------ checks
    def check(self, med: ExtractedMedication | str) -> MedicationCheckResult:
        if isinstance(med, str):
            med = parse_medication(med)
        entry, score, method, cands = self.lookup(med.name)
        res = MedicationCheckResult(medication=med, match_score=round(score, 4), candidates=cands)

        if entry is None:
            res.flags.append(MedicationFlag(
                type=FlagType.UNKNOWN_DRUG, severity="moderate",
                message=f"'{med.name}' not found in reference dataset; verify name and dose manually."))
            return res

        res.matched = entry
        res.match_method = "exact" if method == "exact" else "embedding"
        if method == "embedding":
            res.flags.append(MedicationFlag(
                type=FlagType.NAME_CORRECTED, severity="minor", reference_id=entry.id,
                message=f"'{med.name}' interpreted as '{entry.name}' (spelling/ASR variant); confirm."))
        if method == "low_confidence":
            res.flags.append(MedicationFlag(
                type=FlagType.LOW_CONFIDENCE_MATCH, severity="moderate", reference_id=entry.id,
                message=f"'{med.name}' only loosely matches '{entry.name}' (candidates: {', '.join(cands)}); "
                        "confirm the drug before trusting dose checks."))
            return res  # no dose checks against an uncertain match

        if entry.high_alert:
            res.flags.append(MedicationFlag(
                type=FlagType.HIGH_ALERT_DRUG, severity="minor", reference_id=entry.id,
                message=f"{entry.name} is a high-alert drug. {entry.notes}".strip()))
        res.flags.extend(self._dose_flags(med, entry))
        return res

    def _dose_flags(self, med: ExtractedMedication, e: MedicationEntry) -> list[MedicationFlag]:
        flags: list[MedicationFlag] = []
        has_dose_ref = e.dose_unit and (e.dose_min is not None or e.dose_max is not None)
        if not has_dose_ref:
            return flags

        def add(t: FlagType, sev: str, msg: str):
            flags.append(MedicationFlag(type=t, severity=sev, message=msg, reference_id=e.id))

        if med.dose_value is None:
            add(FlagType.MISSING_DOSE, "minor", f"No dose given for {e.name} (usual {e.dose_min}-{e.dose_max} {e.dose_unit}).")
        else:
            unit = med.dose_unit or e.dose_unit
            dose = _convert(med.dose_value, unit, e.dose_unit)
            if dose is None:
                add(FlagType.UNIT_MISMATCH, "moderate",
                    f"Dose unit '{unit}' is incompatible with reference unit '{e.dose_unit}' for {e.name}.")
            else:
                note = f" (stated as {med.dose_value:g} {unit})" if unit.lower() != e.dose_unit.lower() else ""
                per_dose_flagged = False
                if e.dose_max is not None and dose > e.dose_max:
                    sev = "critical" if dose >= 2 * e.dose_max else "moderate"
                    add(FlagType.DOSE_TOO_HIGH, sev,
                        f"{e.name} {dose:g} {e.dose_unit}{note} exceeds usual max single dose {e.dose_max:g} {e.dose_unit}.")
                    per_dose_flagged = True
                elif e.dose_min is not None and dose < e.dose_min:
                    sev = "moderate" if dose < 0.5 * e.dose_min else "minor"
                    add(FlagType.DOSE_TOO_LOW, sev,
                        f"{e.name} {dose:g} {e.dose_unit}{note} is below usual min single dose {e.dose_min:g} {e.dose_unit}.")
                if (not per_dose_flagged and not med.as_needed and med.frequency_per_day
                        and e.max_daily is not None and dose * med.frequency_per_day > e.max_daily):
                    total = dose * med.frequency_per_day
                    sev = "critical" if total >= 1.5 * e.max_daily else "moderate"
                    add(FlagType.DAILY_DOSE_EXCEEDED, sev,
                        f"{e.name} total {total:g} {e.dose_unit}/day ({dose:g} x {med.frequency_per_day:g}) exceeds usual max {e.max_daily:g} {e.dose_unit}/day.")

        if med.frequency_per_day and e.max_freq_per_day and med.frequency_per_day > e.max_freq_per_day:
            sev = "critical" if med.frequency_per_day >= 2 * e.max_freq_per_day else "moderate"
            add(FlagType.FREQUENCY_UNUSUAL, sev,
                f"{e.name} {med.frequency_per_day:g}x/day exceeds usual max {e.max_freq_per_day:g}x/day.")
        return flags

    def check_many(self, meds: list[ExtractedMedication | str]) -> list[MedicationCheckResult]:
        return [self.check(m) for m in meds]

    def check_prescription_text(self, text: str) -> list[MedicationCheckResult]:
        """Entry point for Person B's string: 'Prescriptions: Paracetamol 500mg twice daily, Cetirizine 10mg'."""
        return self.check_many(parse_prescriptions(text))


# Lazy module-level default so callers can just import the function.
_default: Optional[MedicationRAG] = None


def get_default_rag() -> MedicationRAG:
    global _default
    if _default is None:
        _default = MedicationRAG.build()
    return _default


def check_medications(meds: list[ExtractedMedication | str] | str) -> list[MedicationCheckResult]:
    """Public API: list of meds (objects or phrases) or one raw prescription string."""
    rag = get_default_rag()
    return rag.check_prescription_text(meds) if isinstance(meds, str) else rag.check_many(meds)
