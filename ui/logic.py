"""
Pure helpers for the review screen (no Streamlit imports, so they are unit-tested).

The doctor edits a *view* of the report; these functions turn those widget values back into the
minimal PATCH payload, preserving every field the UI doesn't show (reference_id, flag_count, ...).
"""
from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any, Optional

SEVERITY_ORDER = {"critical": 3, "moderate": 2, "minor": 1}
SEVERITY_ICON = {"critical": "🔴", "moderate": "🟠", "minor": "🟡"}
STATUS_LABEL = {
    "created": "⚪ Created", "transcribed": "🔵 Transcribed", "extracted": "🔵 Extracted",
    "pending_review": "🟠 Awaiting review", "finalized": "🟢 Finalized",
}
SUMMARY_FIELDS = (("chief_complaint", "Chief complaint / symptoms"), ("diagnosis", "Diagnosis"), ("follow_up", "Follow-up"))
RX_EDITABLE = ("drug", "dose", "frequency_per_day", "duration_days", "as_needed")
PIPELINE_ORDER = ["transcribe", "extract", "medication_rag", "history_rag", "assemble_report", "persist_report"]
PIPELINE_LABEL = {
    "transcribe": "Speech → English", "extract": "Medical extraction", "medication_rag": "Medication check",
    "history_rag": "Patient history", "assemble_report": "Draft report", "persist_report": "Saved",
}


def fmt_dt(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return iso


def _clean(v: Any) -> Any:
    """data_editor hands back numpy scalars and NaN; normalise to plain JSON values."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        if v != v:  # NaN
            return None
        if v.is_integer():
            return v  # keep floats as floats (frequency_per_day is 2.0 in the stored report)
    return v


def _int_or_none(v: Any) -> Optional[int]:
    v = _clean(v)
    return None if v is None else int(v)


def prescription_rows(rx: list[dict]) -> list[dict]:
    """Rows for st.data_editor. `_i` is a hidden id that lets us tell edited rows from rows the
    doctor added, and keep the non-displayed fields of the originals."""
    return [{"_i": i, **r} for i, r in enumerate(rx)]


def merge_prescription(original: list[dict], edited_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in edited_rows:
        i = _clean(row.get("_i"))
        if i is not None and 0 <= int(i) < len(original):
            item = copy.deepcopy(original[int(i)])
        else:  # a drug the doctor added by hand: nothing was checked against the reference data
            item = {"drug": "", "as_written": "", "dose": None, "frequency_per_day": None, "as_needed": False,
                    "duration_days": None, "reference_id": None, "flag_count": 0, "max_severity": None,
                    "added_by_doctor": True}
        for k in RX_EDITABLE:
            if k not in row:
                continue
            v = _clean(row[k])
            if k == "duration_days":
                v = _int_or_none(v)
            elif k == "as_needed":
                v = bool(v)
            elif k in ("drug", "dose") and v is not None:
                v = str(v).strip() or None
            item[k] = v
        if not item.get("drug"):
            continue  # blank row
        if item.get("added_by_doctor") and not item.get("as_written"):
            item["as_written"] = item["drug"]
        out.append(item)
    return out


def build_edits(report: dict, summary: dict[str, str], rx_rows: list[dict],
                dismissed: dict[int, bool]) -> dict:
    """Only the report fields that actually changed, ready for POST /workflow/review {action: edit}."""
    edits: dict[str, Any] = {}

    cs = copy.deepcopy(report.get("clinical_summary", {}))
    new_cs = {**cs, **{k: v.strip() for k, v in summary.items()}}
    if new_cs != cs:
        edits["clinical_summary"] = new_cs

    new_rx = merge_prescription(report.get("draft_prescription", []), rx_rows)
    if new_rx != report.get("draft_prescription", []):
        edits["draft_prescription"] = new_rx

    flags = copy.deepcopy(report.get("medication_flags", []))
    new_flags = [{**f, "dismissed": bool(dismissed.get(i, f.get("dismissed", False)))} for i, f in enumerate(flags)]
    if new_flags != flags:
        edits["medication_flags"] = new_flags
    return edits


def open_flags(report: dict) -> list[dict]:
    return [f for f in report.get("medication_flags", []) if not f.get("dismissed")]


def flag_indices_by_severity(flags: list[dict]) -> list[int]:
    return sorted(range(len(flags)), key=lambda i: (-SEVERITY_ORDER.get(flags[i].get("severity"), 0), i))


def report_to_text(patient: dict, consultation: dict, report: dict) -> str:
    """Plain-text copy of a (finalized) report for download / printing."""
    cs = report.get("clinical_summary", {})
    lines = [
        "TELEMED-SCRIBE — CONSULTATION REPORT", "=" * 40,
        f"Patient:      {patient.get('name', '?')}  (age {patient.get('age', '—')}, sex {patient.get('sex') or '—'})",
        f"Consultation: {consultation.get('id', '')}",
        f"Date:         {fmt_dt(consultation.get('created_at'))}",
        f"Status:       {report.get('status', '')}",
    ]
    if report.get("approved_by"):
        lines.append(f"Approved by:  {report['approved_by']}  on {fmt_dt(report.get('finalized_at'))}")
    lines += ["", "CLINICAL SUMMARY", "-" * 40]
    for key, label in SUMMARY_FIELDS:
        lines.append(f"{label}: {cs.get(key) or '—'}")
    lines += ["", "PRESCRIPTION", "-" * 40]
    for d in report.get("draft_prescription", []) or [{}]:
        if not d:
            lines.append("(none)")
            continue
        freq = f"{d['frequency_per_day']:g}x/day" if d.get("frequency_per_day") else ("as needed" if d.get("as_needed") else "—")
        dur = f"{d['duration_days']} days" if d.get("duration_days") else "—"
        lines.append(f"- {d.get('drug')}  {d.get('dose') or '—'}  {freq}  for {dur}")
    flags = report.get("medication_flags", [])
    if flags:
        lines += ["", "MEDICATION FLAGS", "-" * 40]
        for f in flags:
            lines.append(f"- [{f.get('severity', '').upper()}] {f.get('drug')}: {f.get('message')}"
                         + ("  (dismissed by doctor)" if f.get("dismissed") else ""))
    lines += ["", "Generated with AI assistance and reviewed by the signing physician."]
    return "\n".join(lines)


_EDGE = re.compile(r"^\s*(\w+)\s+(-->|-\.->)\s+(\w+)\s*;?\s*$", re.M)


def mermaid_to_dot(mermaid: str) -> str:
    """Edges of the API's Mermaid graph -> Graphviz DOT, so the UI can draw it with st.graphviz_chart
    (no JavaScript, no CDN). Only \\w+ node names are accepted, so server text can't inject markup."""
    edges = _EDGE.findall(mermaid or "")
    nodes = sorted({n for a, _, b in edges for n in (a, b)})
    out = ["digraph G {", "rankdir=TB;", 'node [shape=box, style="rounded,filled", fillcolor="#E6F4F1", '
           'color="#0F766E", fontname="Helvetica"];', 'edge [color="#64748B"];']
    for n in nodes:
        if n in ("__start__", "__end__"):
            out.append(f'{n} [label="{n.strip("_").upper()}", shape=oval, fillcolor="#CBD5E1", color="#64748B"];')
        elif n == "human_review":
            out.append(f'{n} [fillcolor="#FEF3C7", color="#B45309"];')  # the pause for the doctor
    for a, arrow, b in edges:
        out.append(f"{a} -> {b}" + (' [style=dashed, label="route"]' if arrow == "-.->" else "") + ";")
    out.append("}")
    return "\n".join(out)
