import mongomock
import pytest

from db import NotFoundError, ReportLockedError, Store, ensure_indexes
from db.models import Patient


@pytest.fixture
def store():
    db = mongomock.MongoClient()["t"]
    ensure_indexes(db)
    return Store(db)


@pytest.fixture
def cid(store):
    pid = store.create_patient(Patient(name="Test", age=40))
    return store.create_consultation(pid)


TRANSCRIPT = {"original_text": "सिर दर्द", "translated_text": "headache", "language_code": "hi",
              "was_translated": True, "duration": 3.2, "segments": []}
EXTRACT = {"symptoms": "headache", "diagnosis": "viral", "prescriptions": "Paracetamol 500mg bd",
           "follow_up": "5 days", "raw_output": "x"}


def test_pipeline_status_flow(store, cid):
    status = lambda: store.db.consultations.find_one({"_id": cid})["status"]
    store.save_transcript(cid, TRANSCRIPT); assert status() == "transcribed"
    store.save_extraction(cid, EXTRACT); assert status() == "extracted"
    store.save_report(cid, {"clinical_summary": {"cc": "headache"}}); assert status() == "pending_review"
    store.finalize_report(cid, "Dr Sharma"); assert status() == "finalized"


def test_transcript_keeps_original_and_translation(store, cid):
    store.save_transcript(cid, TRANSCRIPT)
    t = store.get_transcript(cid)
    assert t["original_text"] == "सिर दर्द" and t["translated_text"] == "headache"


def test_rerun_overwrites_not_duplicates(store, cid):
    store.save_extraction(cid, EXTRACT)
    store.save_extraction(cid, {**EXTRACT, "diagnosis": "flu"})
    assert store.db.extractions.count_documents({"consultation_id": cid}) == 1
    assert store.get_extraction(cid)["diagnosis"] == "flu"


def test_finalized_report_is_locked(store, cid):
    store.save_report(cid, {})
    store.finalize_report(cid, "Dr A")
    with pytest.raises(ReportLockedError): store.update_report(cid, {"medication_flags": []})
    with pytest.raises(ReportLockedError): store.save_report(cid, {})
    with pytest.raises(ReportLockedError): store.finalize_report(cid, "Dr A")


def test_update_rejects_unknown_field(store, cid):
    store.save_report(cid, {})
    with pytest.raises(ValueError): store.update_report(cid, {"status": "finalized"})


def test_bad_ids(store):
    with pytest.raises(NotFoundError): store.create_consultation("nope")
    with pytest.raises(NotFoundError): store.get_report("nope")


def test_history_only_finalized_and_excludes_current(store):
    pid = store.create_patient(Patient(name="H"))
    old, draft, cur = (store.create_consultation(pid) for _ in range(3))
    for c in (old, draft):
        store.save_extraction(c, EXTRACT); store.save_report(c, {})
    store.finalize_report(old, "Dr A")  # draft stays unreviewed
    h = store.history_for_patient(pid, exclude=cur)
    assert [r["consultation_id"] for r in h] == [old]
