"""
ml/rag/service.py — One call for Person D's LangGraph nodes.

    out = run_rag(store, cid, med_rag, hist_rag)
    store.save_report(cid, {..., **out})   # keys match db.models.Report: medication_flags, history_context

Does not write to Mongo itself (writing via save_extraction would also reset consultation status).
"""

from __future__ import annotations

from typing import Optional

from ml.rag.history_rag import PatientHistoryRAG
from ml.rag.medication_rag import MedicationRAG


def run_rag(store, cid: str, med_rag: MedicationRAG, hist_rag: PatientHistoryRAG, k: int = 3) -> dict:
    ex = store.get_extraction(cid)
    patient_id = store.db.consultations.find_one({"_id": cid})["patient_id"]

    results = med_rag.check_prescription_text(ex.get("prescriptions", ""))
    hits = hist_rag.retrieve(store, patient_id, ex.get("symptoms", ""), ex.get("diagnosis", ""), exclude=cid, k=k)
    return {
        "medication_checks": [r.to_dict() for r in results],                  # every drug, flagged or not
        "medications": [r.medication.model_dump() for r in results],          # for Extraction.medications
        "medication_flags": [r.to_dict() for r in results if r.flags],        # for Report.medication_flags
        "history_context": [h.to_dict() for h in hits],                       # for Report.history_context
    }
