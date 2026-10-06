"""All Mongo reads/writes live here — Person D calls these, never raw pymongo."""
from __future__ import annotations

import uuid
from typing import Any, Optional, Union

from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from db.models import (
    ConsultationStatus, Extraction, Patient, Report, ReportStatus, utcnow,
)


class NotFoundError(Exception):
    pass


class ReportLockedError(Exception):
    """Raised on any write to a finalized report."""


_STATUS_ORDER = [c.value for c in ConsultationStatus]  # created < transcribed < extracted < pending_review < finalized


def _id() -> str:
    return uuid.uuid4().hex


def _dump(obj: Any) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


class Store:
    def __init__(self, db: Database):
        self.db = db

    # ---------------- patients ----------------
    def create_patient(self, patient: Patient) -> str:
        pid = _id()
        self.db.patients.insert_one({"_id": pid, **patient.model_dump()})
        return pid

    def get_patient(self, patient_id: str) -> dict:
        doc = self.db.patients.find_one({"_id": patient_id})
        if not doc:
            raise NotFoundError(f"patient {patient_id}")
        return doc

    # ---------------- consultations ----------------
    def create_consultation(self, patient_id: str) -> str:
        self.get_patient(patient_id)  # fail early on bad id
        cid = _id()
        self.db.consultations.insert_one({
            "_id": cid,
            "patient_id": patient_id,
            "status": ConsultationStatus.CREATED.value,
            "created_at": utcnow(),
        })
        return cid

    def _set_status(self, cid: str, status: ConsultationStatus) -> None:
        """Move status forward only (a re-run must never pull a visit back from pending_review/finalized)."""
        earlier = _STATUS_ORDER[: _STATUS_ORDER.index(status.value)]
        r = self.db.consultations.update_one(
            {"_id": cid, "status": {"$in": earlier}}, {"$set": {"status": status.value}})
        if r.matched_count == 0 and not self.db.consultations.find_one({"_id": cid}, {"_id": 1}):
            raise NotFoundError(f"consultation {cid}")

    def _assert_not_finalized(self, cid: str) -> None:
        c = self.db.consultations.find_one({"_id": cid}, {"status": 1})
        if not c:
            raise NotFoundError(f"consultation {cid}")
        if c["status"] == ConsultationStatus.FINALIZED.value:
            raise ReportLockedError(cid)

    def list_consultations(self, patient_id: str, limit: int = 50) -> list[dict]:
        return list(self.db.consultations.find({"patient_id": patient_id})
                    .sort("created_at", -1).limit(limit))

    # ---------------- transcript (Person A's ConsultationResult) ----------------
    def save_transcript(self, cid: str, result: Union[dict, Any]) -> None:
        """Pass ConsultationResult or its .to_dict(). Keeps original + translated text."""
        self._assert_not_finalized(cid)
        data = _dump(result)
        self.db.transcripts.update_one(
            {"consultation_id": cid},
            {"$set": {"consultation_id": cid, **data, "saved_at": utcnow()}},
            upsert=True,
        )
        self._set_status(cid, ConsultationStatus.TRANSCRIBED)

    def get_transcript(self, cid: str) -> dict:
        doc = self.db.transcripts.find_one({"consultation_id": cid})
        if not doc:
            raise NotFoundError(f"transcript for {cid}")
        return doc

    # ---------------- extraction (Person B) ----------------
    def save_extraction(self, cid: str, extraction: Union[dict, Extraction]) -> None:
        self._assert_not_finalized(cid)
        # exclude_unset: a B-side re-run that omits `medications` must not wipe Phase-4 results
        data = Extraction(**_dump(extraction)).model_dump(exclude_unset=True)
        update = {"$set": {"consultation_id": cid, **data, "saved_at": utcnow()}}
        if "medications" not in data:
            update["$setOnInsert"] = {"medications": []}
        self.db.extractions.update_one({"consultation_id": cid}, update, upsert=True)
        self._set_status(cid, ConsultationStatus.EXTRACTED)

    def get_extraction(self, cid: str) -> dict:
        doc = self.db.extractions.find_one({"consultation_id": cid})
        if not doc:
            raise NotFoundError(f"extraction for {cid}")
        return doc

    # ---------------- report ----------------
    def save_report(self, cid: str, report: Union[dict, Report]) -> None:
        """Create/overwrite the draft. Refuses if already finalized."""
        data = Report(**_dump(report)).model_dump()
        try:
            self.db.reports.update_one(
                {"consultation_id": cid, "status": {"$ne": ReportStatus.FINALIZED.value}},
                {"$set": {"consultation_id": cid, **data,
                          "status": ReportStatus.PENDING_REVIEW.value,
                          "updated_at": utcnow()}},
                upsert=True,
            )
        except DuplicateKeyError:  # filter missed because the existing doc is finalized
            raise ReportLockedError(cid)
        self._set_status(cid, ConsultationStatus.PENDING_REVIEW)

    def update_report(self, cid: str, fields: dict[str, Any]) -> None:
        """Doctor edits. Only whitelisted report fields; blocked once finalized."""
        allowed = set(Report.model_fields)
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown report fields: {sorted(bad)}")
        r = self.db.reports.update_one(
            {"consultation_id": cid, "status": ReportStatus.PENDING_REVIEW.value},
            {"$set": {**fields, "updated_at": utcnow()}},
        )
        if r.matched_count == 0:
            self._raise_missing_or_locked(cid)

    def finalize_report(self, cid: str, approved_by: str) -> None:
        r = self.db.reports.update_one(
            {"consultation_id": cid, "status": ReportStatus.PENDING_REVIEW.value},
            {"$set": {"status": ReportStatus.FINALIZED.value,
                      "approved_by": approved_by, "finalized_at": utcnow()}},
        )
        if r.matched_count == 0:
            self._raise_missing_or_locked(cid)
        self._set_status(cid, ConsultationStatus.FINALIZED)

    def get_report(self, cid: str) -> dict:
        doc = self.db.reports.find_one({"consultation_id": cid})
        if not doc:
            raise NotFoundError(f"report for {cid}")
        return doc

    def _raise_missing_or_locked(self, cid: str) -> None:
        if self.db.reports.find_one({"consultation_id": cid}, {"_id": 1}):
            raise ReportLockedError(cid)
        raise NotFoundError(f"report for {cid}")

    # ---------------- bundle + history (feeds RAG / UI) ----------------
    def get_bundle(self, cid: str) -> dict:
        """Everything for one consultation; missing parts come back as None."""
        c = self.db.consultations.find_one({"_id": cid})
        if not c:
            raise NotFoundError(f"consultation {cid}")
        pick = lambda coll: self.db[coll].find_one({"consultation_id": cid})
        return {"consultation": c, "transcript": pick("transcripts"),
                "extraction": pick("extractions"), "report": pick("reports")}

    def history_for_patient(self, patient_id: str, exclude: Optional[str] = None) -> list[dict]:
        """Past *finalized* visits only (unreviewed AI output must not feed history RAG).
        One row per visit: date, symptoms, diagnosis, prescriptions, follow_up."""
        out = []
        for c in self.list_consultations(patient_id, limit=500):
            if c["_id"] == exclude or c["status"] != ConsultationStatus.FINALIZED.value:
                continue
            ex = self.db.extractions.find_one({"consultation_id": c["_id"]}) or {}
            out.append({
                "consultation_id": c["_id"], "date": c["created_at"],
                **{k: ex.get(k, "") for k in ("symptoms", "diagnosis", "prescriptions", "follow_up")},
            })
        return out