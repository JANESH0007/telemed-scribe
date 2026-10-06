"""Request bodies. Responses are the stored documents (see api/serialization.py)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PatientIn(BaseModel):
    name: str = Field(min_length=1)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    sex: Optional[str] = None


class ConsultationIn(BaseModel):
    patient_id: str


class ReportPatch(BaseModel):
    """Doctor edits. Send only the fields you changed."""
    clinical_summary: Optional[dict[str, Any]] = None
    draft_prescription: Optional[list[dict[str, Any]]] = None
    medication_flags: Optional[list[dict[str, Any]]] = None   # set "dismissed": true on a flag to dismiss it
    history_context: Optional[list[dict[str, Any]]] = None


class FinalizeIn(BaseModel):
    approved_by: str = Field(min_length=1)
