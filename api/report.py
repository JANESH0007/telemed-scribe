"""
Draft report builder (Phase 6 placeholder, deterministic — no LLM).
Maps extraction + RAG output onto db.models.Report. Swap build_draft_report for an LLM call later;
the output shape (the 4 Report fields) is what the doctor-review UI depends on.
"""
from __future__ import annotations

_SEV = {"minor": 1, "moderate": 2, "critical": 3}


def build_draft_report(extraction: dict, rag_out: dict, transcript: dict | None = None) -> dict:
    prescription, flags = [], []
    for chk in rag_out["medication_checks"]:
        med, matched = chk["medication"], chk.get("matched")
        fl = chk.get("flags", [])
        dose = f"{med['dose_value']:g} {med.get('dose_unit') or (matched or {}).get('dose_unit') or ''}".strip() if med.get("dose_value") is not None else None
        prescription.append({
            "drug": matched["name"] if matched else med["name"],
            "as_written": med["name"],
            "dose": dose,
            "frequency_per_day": med.get("frequency_per_day"),
            "as_needed": med.get("as_needed", False),
            "duration_days": med.get("duration_days"),
            "reference_id": matched["id"] if matched else None,
            "flag_count": len(fl),
            "max_severity": max((f["severity"] for f in fl), key=_SEV.get, default=None),
        })
        flags.extend({"drug": med["name"], "dismissed": False, **f} for f in fl)

    return {
        "clinical_summary": {
            "chief_complaint": extraction.get("symptoms", ""),
            "diagnosis": extraction.get("diagnosis", ""),
            "follow_up": extraction.get("follow_up", ""),
            "language": (transcript or {}).get("language"),
            "note": "AI-generated draft; requires doctor review before use.",
        },
        "draft_prescription": prescription,
        "medication_flags": flags,
        "history_context": rag_out["history_context"],
    }
