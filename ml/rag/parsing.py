"""
ml/rag/parsing.py — Turn Person B's free-text prescription string into ExtractedMedication objects.

B's T5 output looks like: "Paracetamol 500mg twice daily, Ibuprofen 400mg; Cetirizine 10 mg at night".
Handles: tab/cap prefixes, mg/mcg/g/ml/IU/puffs, OD/BD/TDS/QID, "twice a day", "3 times daily",
"every 8 hours", Indian "1-0-1" notation, SOS/PRN, "for 5 days / 1 week".
"""

from __future__ import annotations

import re
from typing import Optional

from ml.rag.schemas import ExtractedMedication

_FORM_PREFIX = re.compile(r"^\s*(?:tab(?:let)?s?|cap(?:sule)?s?|syp|syrup|inj(?:ection)?|oint(?:ment)?)\.?\s+", re.I)
_DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mcg|µg|ug|mg|g|ml|iu|puffs?)\b", re.I)
_NAME_END = re.compile(r"(?<![A-Za-z])\d|\b(?:once|twice|thrice|daily|bd|bid|od|tds|tid|qid|qds|sos|prn|every|for|at|after|before|x)\b", re.I)
_BARE_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.%/-])")
_PATTERN_101 = re.compile(r"\b(\d)\s*-\s*(\d)\s*-\s*(\d)(?:\s*-\s*(\d))?\b")
_N_TIMES = re.compile(r"\b(\d+)\s*(?:times?|x)\s*(?:a|per|/)?\s*(?:day|daily)\b|\b(\d+)\s*/\s*day\b", re.I)
_EVERY_H = re.compile(r"\bevery\s+(\d+)\s*(?:hours?|hrs?|h)\b", re.I)
_DURATION = re.compile(r"\b(?:for|x)\s*(\d+)\s*(day|days|week|weeks|month|months)\b", re.I)
_FREQ_WORDS = [
    (re.compile(r"\b(?:four times|qid|qds)\b", re.I), 4),
    (re.compile(r"\b(?:thrice|three times|tds|tid)\b", re.I), 3),
    (re.compile(r"\b(?:twice|two times|bd|bid)\b", re.I), 2),
    (re.compile(r"\b(?:once|one time|od|qd|daily|every day|at night|at bedtime|nightly|in the morning|every morning)\b", re.I), 1),
]
_AS_NEEDED = re.compile(r"\b(?:sos|prn|as needed|when required)\b", re.I)
_CONTINUATION_FIRST = {
    "once", "twice", "thrice", "daily", "for", "after", "before", "with", "at", "every", "x", "bd", "bid", "od",
    "tds", "tid", "qid", "qds", "qd", "morning", "night", "evening", "sos", "prn", "per", "a", "one", "two",
    "three", "four", "five", "in", "as", "when",
}


def _frequency(text: str) -> Optional[float]:
    m = _PATTERN_101.search(text)
    if m:
        return float(sum(int(g) for g in m.groups() if g))
    m = _N_TIMES.search(text)
    if m:
        return float(m.group(1) or m.group(2))
    m = _EVERY_H.search(text)
    if m and int(m.group(1)) > 0:
        return round(24 / int(m.group(1)), 2)
    for rx, val in _FREQ_WORDS:
        if rx.search(text):
            return float(val)
    return None


def _duration_days(text: str) -> Optional[int]:
    m = _DURATION.search(text)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    return n * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1)


def parse_medication(text: str) -> ExtractedMedication:
    """Parse ONE medication phrase."""
    raw = text.strip()
    body = _FORM_PREFIX.sub("", raw)
    m = _NAME_END.search(body)
    name = (body[: m.start()] if m else body).strip(" ,.-:")
    dose_value = dose_unit = None
    d = _DOSE.search(body)
    if d:
        dose_value = float(d.group(1))
        u = d.group(2).lower()
        dose_unit = {"µg": "mcg", "ug": "mcg", "iu": "IU", "puff": "puffs"}.get(u, u)
    if dose_value is None:  # bare number with no unit ("Pantoprazole 40 od"): unit assumed from reference later
        rest = body
        for rx in (_DURATION, _PATTERN_101, _N_TIMES, _EVERY_H):
            rest = rx.sub(" ", rest)
        b = _BARE_NUM.search(rest[len(name):] if rest.startswith(name) else rest)
        if b:
            dose_value = float(b.group(1))
    as_needed = bool(_AS_NEEDED.search(body))
    freq = None if as_needed else _frequency(body)
    return ExtractedMedication(
        name=name or raw, dose_value=dose_value, dose_unit=dose_unit,
        frequency_per_day=freq, as_needed=as_needed, duration_days=_duration_days(body), raw=raw,
    )


def split_prescriptions(text: str) -> list[str]:
    """Split a multi-drug string. Commas inside one drug's instructions ('500mg, twice daily') are kept together."""
    text = re.sub(r"^\s*prescriptions?\s*:\s*", "", text.strip(), flags=re.I)
    parts: list[str] = []
    for chunk in re.split(r"[;\n+]|\band\b(?=\s+[A-Za-z]{4,}\s+\d)", text):
        for piece in chunk.split(","):
            piece = piece.strip()
            if not piece:
                continue
            first = re.split(r"\W+", piece.lower(), maxsplit=1)[0]
            if parts and (piece[0].isdigit() or first in _CONTINUATION_FIRST):
                parts[-1] += ", " + piece
            else:
                parts.append(piece)
    return [p for p in parts if p.lower() not in {"none", "none specified", "n/a", "na", "-"}]


def parse_prescriptions(text: str) -> list[ExtractedMedication]:
    return [parse_medication(p) for p in split_prescriptions(text)]
