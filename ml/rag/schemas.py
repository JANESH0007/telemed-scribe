"""
ml/rag/schemas.py — Data contracts for the RAG layer (Person C).

Severity strings match ml/speech/evaluation (critical | moderate | minor).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExtractedMedication(BaseModel):
    """One medication as pulled from the extraction step (Person B)."""

    name: str = Field(description="Drug name as written/said (brand or generic)")
    dose_value: Optional[float] = Field(default=None, description="Numeric dose per intake")
    dose_unit: Optional[str] = Field(default=None, description="mg | mcg | g | ml | IU | puffs")
    frequency_per_day: Optional[float] = Field(default=None, description="Intakes per day")
    as_needed: bool = Field(default=False, description="SOS / PRN")
    duration_days: Optional[int] = None
    raw: str = Field(default="", description="Original text this was parsed from")


class MedicationEntry(BaseModel):
    """One row of the reference dataset."""

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    drug_class: str = ""
    dose_unit: Optional[str] = None
    dose_min: Optional[float] = None
    dose_max: Optional[float] = None
    max_daily: Optional[float] = None
    max_freq_per_day: Optional[float] = None
    route: str = ""
    usual_use: str = ""
    notes: str = ""
    high_alert: bool = False


class FlagType(str, Enum):
    UNKNOWN_DRUG = "unknown_drug"
    LOW_CONFIDENCE_MATCH = "low_confidence_match"
    NAME_CORRECTED = "name_corrected"
    DOSE_TOO_HIGH = "dose_too_high"
    DOSE_TOO_LOW = "dose_too_low"
    DAILY_DOSE_EXCEEDED = "daily_dose_exceeded"
    FREQUENCY_UNUSUAL = "frequency_unusual"
    MISSING_DOSE = "missing_dose"
    UNIT_MISMATCH = "unit_mismatch"
    HIGH_ALERT_DRUG = "high_alert_drug"


class MedicationFlag(BaseModel):
    type: FlagType
    severity: str = Field(description="critical | moderate | minor")
    message: str
    reference_id: Optional[str] = Field(default=None, description="Reference row id")


class MedicationCheckResult(BaseModel):
    """Result for one medication — what Person D / the report LLM consumes."""

    medication: ExtractedMedication
    matched: Optional[MedicationEntry] = None
    match_score: float = 0.0
    match_method: str = Field(default="none", description="exact | embedding | none")
    candidates: list[str] = Field(default_factory=list, description="Top alternatives (low-confidence only)")
    flags: list[MedicationFlag] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity in ("critical", "moderate") for f in self.flags)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
