import mongomock, pytest
from db import ReportLockedError, Store, ensure_indexes
from db.models import Patient

E = {"symptoms": "headache", "diagnosis": "viral", "prescriptions": "Paracetamol 500mg bd", "follow_up": "5 days"}


@pytest.fixture
def s_c():
    s = Store(mongomock.MongoClient()["t"]); ensure_indexes(s.db)
    return s, s.create_consultation(s.create_patient(Patient(name="T")))


def status(s, c): return s.db.consultations.find_one({"_id": c})["status"]


def test_status_never_regresses(s_c):
    s, c = s_c
    s.save_extraction(c, E); s.save_report(c, {})
    s.save_transcript(c, {"original_text": "a", "translated_text": "b"})
    assert status(s, c) == "pending_review"          # was reset to "transcribed" before the fix


def test_finalized_visit_is_immutable(s_c):
    s, c = s_c
    s.save_extraction(c, E); s.save_report(c, {}); s.finalize_report(c, "Dr A")
    with pytest.raises(ReportLockedError): s.save_extraction(c, {**E, "diagnosis": "CHANGED"})
    with pytest.raises(ReportLockedError): s.save_transcript(c, {"original_text": "a"})
    assert s.get_extraction(c)["diagnosis"] == "viral" and status(s, c) == "finalized"


def test_extraction_rerun_keeps_medications(s_c):
    s, c = s_c
    s.save_extraction(c, {**E, "medications": [{"name": "Paracetamol"}]})
    s.save_extraction(c, {**E, "diagnosis": "flu"})   # B re-runs, no medications key
    ex = s.get_extraction(c)
    assert ex["diagnosis"] == "flu" and ex["medications"] == [{"name": "Paracetamol"}]


def test_first_extraction_defaults_medications_empty(s_c):
    s, c = s_c
    s.save_extraction(c, E)
    assert s.get_extraction(c)["medications"] == []