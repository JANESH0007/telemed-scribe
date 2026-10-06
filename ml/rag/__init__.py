"""
TeleMed-Scribe — RAG & Data Layer (Person C)

    from ml.rag import check_medications
    results = check_medications("Paracetamol 500mg twice daily, Amlodipine 5mg")
"""

from ml.rag.medication_rag import MedicationRAG, check_medications
from ml.rag.history_rag import PatientHistoryRAG
from ml.rag.schemas import ExtractedMedication, MedicationCheckResult, MedicationFlag

__all__ = ["MedicationRAG", "PatientHistoryRAG", "check_medications", "ExtractedMedication", "MedicationCheckResult", "MedicationFlag"]
