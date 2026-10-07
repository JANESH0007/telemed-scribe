"""
Graph state + the doctor's review decision.

State rules (they matter because the graph is checkpointed):
  * Only plain JSON-like values. No Store/Services objects (those are closed over in graph.py),
    no Mongo documents (ObjectId / datetime would not survive serialization).
  * MongoDB stays the system of record. State only carries what the *next node* needs.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, model_validator

from api.schemas import ReportPatch

Entry = Literal["audio", "extract", "review"]


def merge_trace(old: Optional[list], new: Optional[list]) -> list:
    """Append-only step log. A leading {"_reset": True} marker starts a fresh log, so that a
    thread reused for a second run (same consultation id) does not show the previous run's steps."""
    new = list(new or [])
    if new and new[0].get("_reset"):
        return new[1:]
    return (old or []) + new


class WorkflowState(TypedDict, total=False):
    # ---- inputs ----
    consultation_id: str
    entry: Entry                      # where this run starts (see graph.route_entry)
    audio_path: Optional[str]
    language: Optional[str]           # force an ISO code, e.g. "hi"
    overwrite: bool                   # allow replacing an existing draft report

    # ---- produced by nodes ----
    patient_id: str
    transcript_meta: dict[str, Any]   # language, language_code, language_probability, was_translated
    extraction: dict[str, str]        # symptoms / diagnosis / prescriptions / follow_up / raw_output / model
    medication_checks: list[dict]
    medications: list[dict]
    history_context: list[dict]
    report: dict[str, Any]
    warnings: list[str]

    # ---- human review ----
    decision: dict[str, Any]
    status: Literal["awaiting_review", "finalized"]

    # ---- observability ----
    trace: Annotated[list, merge_trace]


class ReviewDecision(BaseModel):
    """What the doctor sends back when the graph is paused for review."""
    action: Literal["edit", "approve"]
    approved_by: Optional[str] = None     # required for approve
    edits: Optional[ReportPatch] = None   # required for edit; same fields/semantics as PATCH /report

    def edit_fields(self) -> dict[str, Any]:
        """Only the fields the doctor actually sent (mirrors api.main.edit_report)."""
        if self.edits is None:
            return {}
        return {k: v for k, v in self.edits.model_dump(exclude_unset=True).items() if v is not None}

    @model_validator(mode="after")
    def _check(self) -> "ReviewDecision":
        if self.action == "approve" and not (self.approved_by or "").strip():
            raise ValueError("approve requires approved_by")
        if self.action == "edit" and not self.edit_fields():
            raise ValueError("edit requires at least one field to change")
        return self

    def to_state(self) -> dict[str, Any]:
        return {"action": self.action, "approved_by": self.approved_by, "edits": self.edit_fields()}
