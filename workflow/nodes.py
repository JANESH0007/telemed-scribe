"""
Graph nodes. Each node is a thin adapter over code that already exists:

    transcribe      -> Services.transcribe  (Person A)  -> Store.save_transcript
    extract         -> Services.extract     (Person B)  -> Store.save_extraction
    medication_rag  -> MedicationRAG        (Person C)
    history_rag     -> PatientHistoryRAG    (Person C)
    assemble_report -> api.report.build_draft_report (Person D)
    persist_report  -> Store.save_extraction / Store.save_report
    human_review / apply_edits / finalize -> Store.update_report / Store.finalize_report

Errors are NOT swallowed: BadInput / NotFoundError / ReportLockedError / FileExistsError propagate
out of graph.invoke so the existing FastAPI handlers keep mapping them to 422 / 404 / 409.

Writes happen only in transcribe, extract, persist_report, apply_edits and finalize. The RAG and
assemble nodes are pure, so re-running them is always safe.
"""
from __future__ import annotations

import os
import time
from functools import wraps
from typing import Any, Callable

from langgraph.types import interrupt

from api.report import build_draft_report
from api.services import BadInput, Services
from db import NotFoundError, ReportLockedError, Store
from db.models import ConsultationStatus, ReportStatus
from workflow.state import ReviewDecision, WorkflowState

EXTRACTION_KEYS = ("symptoms", "diagnosis", "prescriptions", "follow_up", "raw_output", "model")
TRANSCRIPT_META_KEYS = ("language", "language_code", "language_probability", "was_translated")
HISTORY_K = 3


def _low_lang_conf() -> float:
    return float(os.getenv("LOW_LANGUAGE_CONFIDENCE", "0.6"))


def traced(name: str, fn: Callable[[WorkflowState], dict]) -> Callable[[WorkflowState], dict]:
    """Append {node, ms} to the trace when the node completes. Exceptions (including LangGraph's
    interrupt signal) pass straight through, so a paused or failed node leaves no entry."""
    @wraps(fn)
    def wrapper(state: WorkflowState) -> dict:
        t0 = time.perf_counter()
        out = fn(state) or {}
        entry = {"node": name, "ms": round((time.perf_counter() - t0) * 1000)}
        return {**out, "trace": [*out.get("trace", []), entry]}
    return wrapper


def make_nodes(store: Store, svc: Services) -> dict[str, Callable[[WorkflowState], dict]]:
    # ------------------------------------------------------------------ init
    def init(state: WorkflowState) -> dict:
        """Validate, reset per-run fields, load the patient id."""
        cid = state["consultation_id"]
        c = store.db.consultations.find_one({"_id": cid})
        if not c:
            raise NotFoundError(f"consultation {cid}")
        if c["status"] == ConsultationStatus.FINALIZED.value:
            raise ReportLockedError(cid)  # finalized visits are immutable
        if state["entry"] != "review" and not state.get("overwrite"):
            try:
                store.get_report(cid)
            except NotFoundError:
                pass
            else:  # check BEFORE the expensive STT/extraction work, and before anything is overwritten
                raise FileExistsError(cid)
        return {
            "patient_id": c["patient_id"],
            "transcript_meta": {}, "extraction": {}, "medication_checks": [], "medications": [],
            "history_context": [], "report": {}, "warnings": [], "decision": {},
            "trace": [{"_reset": True}],
        }

    # ------------------------------------------------------------------ speech (Person A)
    def transcribe(state: WorkflowState) -> dict:
        path = state.get("audio_path")
        if not path:
            raise BadInput("audio_path is required to start from the audio stage")
        store.save_transcript(state["consultation_id"], svc.transcribe(path, state.get("language")))
        return {"audio_path": None}  # the upload is temporary; don't keep the path in a checkpoint

    # ------------------------------------------------------------------ extraction (Person B)
    def extract(state: WorkflowState) -> dict:
        cid = state["consultation_id"]
        tr = store.get_transcript(cid)  # NotFoundError (404) if the audio stage never ran
        store.save_extraction(cid, svc.extract(tr.get("translated_text", "")))  # BadInput (422) if empty
        ex = store.get_extraction(cid)
        return {
            "transcript_meta": {k: tr.get(k) for k in TRANSCRIPT_META_KEYS},
            "extraction": {k: ex.get(k, "") for k in EXTRACTION_KEYS},
        }

    # ------------------------------------------------------------------ RAG (Person C)
    # Sequential on purpose: both lazily load their own embedding model on first use, and loading
    # two SentenceTransformers from parallel threads is a known source of flaky errors. The work
    # itself is milliseconds, so there is nothing to gain from fanning out.
    def medication_rag(state: WorkflowState) -> dict:
        results = svc.med_rag.check_prescription_text(state["extraction"].get("prescriptions", ""))
        return {
            "medication_checks": [r.to_dict() for r in results],              # every drug, flagged or not
            "medications": [r.medication.model_dump() for r in results],      # persisted on the extraction
        }

    def history_rag(state: WorkflowState) -> dict:
        ex = state["extraction"]
        hits = svc.hist_rag.retrieve(store, state["patient_id"], ex.get("symptoms", ""),
                                     ex.get("diagnosis", ""), exclude=state["consultation_id"], k=HISTORY_K)
        return {"history_context": [h.to_dict() for h in hits]}

    # ------------------------------------------------------------------ report (Person D)
    def assemble_report(state: WorkflowState) -> dict:
        ex, meta = state["extraction"], state["transcript_meta"]
        report = build_draft_report(
            ex,
            {"medication_checks": state["medication_checks"], "history_context": state["history_context"]},
            {"language": meta.get("language")},
        )
        warnings = _quality_warnings(ex, meta, state["medication_checks"])
        if warnings:
            report["clinical_summary"]["warnings"] = warnings
        return {"report": report, "warnings": warnings}

    def persist_report(state: WorkflowState) -> dict:
        cid = state["consultation_id"]
        ex = store.get_extraction(cid)
        # full fields + parsed medications, same as api.pipeline.report_stage
        store.save_extraction(cid, {**{k: ex.get(k, "") for k in EXTRACTION_KEYS},
                                    "medications": state["medications"]})
        store.save_report(cid, state["report"])  # -> consultation status: pending_review
        return {"status": "awaiting_review"}

    # ------------------------------------------------------------------ human in the loop
    def human_review(state: WorkflowState) -> dict:
        """Pause until the doctor responds. On resume LangGraph re-runs this function from the top,
        with interrupt() returning the resume value, so everything above it must be idempotent."""
        cid = state["consultation_id"]
        report = store.get_report(cid)  # NotFoundError (404) if there is nothing to review
        if report["status"] == ReportStatus.FINALIZED.value:
            raise ReportLockedError(cid)
        payload = interrupt({
            "consultation_id": cid,
            "awaiting": "doctor_review",
            "flags": len([f for f in report.get("medication_flags", []) if not f.get("dismissed")]),
            "warnings": report.get("clinical_summary", {}).get("warnings", []),
            "allowed_actions": ["edit", "approve"],
        })
        # Validate here, inside the paused node: if the payload is malformed this raises while the
        # thread is still waiting, so the doctor can simply submit a corrected decision.
        decision = ReviewDecision.model_validate(payload)
        return {"decision": decision.to_state(), "status": "awaiting_review"}

    def apply_edits(state: WorkflowState) -> dict:
        store.update_report(state["consultation_id"], state["decision"]["edits"])  # whitelisted fields only
        return {"decision": {}}

    def finalize(state: WorkflowState) -> dict:
        store.finalize_report(state["consultation_id"], state["decision"]["approved_by"])
        return {"status": "finalized"}

    nodes: dict[str, Callable[[WorkflowState], Any]] = {
        "init": init, "transcribe": transcribe, "extract": extract,
        "medication_rag": medication_rag, "history_rag": history_rag,
        "assemble_report": assemble_report, "persist_report": persist_report,
        "human_review": human_review, "apply_edits": apply_edits, "finalize": finalize,
    }
    return {name: traced(name, fn) for name, fn in nodes.items()}


def _quality_warnings(ex: dict, meta: dict, checks: list[dict]) -> list[str]:
    """Plain-language cautions for the reviewing doctor. They never block the workflow."""
    w: list[str] = []
    prob = meta.get("language_probability")
    if prob is not None and prob < _low_lang_conf():
        w.append(f"Low language-detection confidence ({prob:.2f}); the transcript may be unreliable.")
    if meta.get("was_translated"):
        w.append(f"Transcript was machine-translated from {meta.get('language') or 'the source language'}; "
                 "verify drug names and doses against the original audio.")
    if not (ex.get("symptoms") or ex.get("diagnosis") or ex.get("prescriptions")):
        w.append("The extractor found no symptoms, diagnosis or prescriptions; check the transcript.")
    elif not ex.get("prescriptions"):
        w.append("No prescription was extracted.")
    elif not checks:
        w.append("A prescription was extracted but no medications could be parsed from it.")
    return w
