"""Document shapes. Validation happens before anything is written."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConsultationStatus(str, Enum):
    CREATED = "created"
    TRANSCRIBED = "transcribed"
    EXTRACTED = "extracted"
    PENDING_REVIEW = "pending_review"
    FINALIZED = "finalized"


class ReportStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    FINALIZED = "finalized"


class Patient(BaseModel):
    name: str
    age: Optional[int] = None
    sex: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class Extraction(BaseModel):
    """Person B's T5 output. Fields are free text, exactly as the model emits."""
    symptoms: str = ""
    diagnosis: str = ""
    prescriptions: str = ""
    follow_up: str = ""
    raw_output: str = ""
    model: str = "t5-small-lora"
    # Filled later by the medication parser (Phase 4); [] until then.
    medications: list[dict[str, Any]] = Field(default_factory=list)


class Report(BaseModel):
    clinical_summary: dict[str, Any] = Field(default_factory=dict)
    draft_prescription: list[dict[str, Any]] = Field(default_factory=list)
    medication_flags: list[dict[str, Any]] = Field(default_factory=list)
    history_context: list[dict[str, Any]] = Field(default_factory=list)
