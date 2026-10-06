"""API tests: fake transcriber/extractor + hashing-embedder RAGs, mongomock. No models, no Mongo server."""
import mongomock
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import BadInput, Services
from db import Store, ensure_indexes
from ml.rag import MedicationRAG, PatientHistoryRAG
from ml.rag.embedder import HashingEmbedder

TRANSCRIPT = {"original_text": "bukhar", "translated_text": "I have fever and cough", "language": "Hindi",
              "language_code": "hi", "language_probability": 0.99, "was_translated": True, "duration": 4.0, "segments": []}
EXTRACTION = {"symptoms": "fever, cough", "diagnosis": "viral infection",
              "prescriptions": "Paracetamol 5000mg bd, Amoxicilin 500mg tds", "follow_up": "5 days", "raw_output": "x"}


def fake_transcriber(path, language):
    if path.endswith(".bad"):
        raise BadInput("unsupported audio format")
    return TRANSCRIPT


@pytest.fixture
def client():
    store = Store(mongomock.MongoClient()["t"]); ensure_indexes(store.db)
    svc = Services(transcriber=fake_transcriber, extractor=lambda t: EXTRACTION,
                   med_rag=MedicationRAG.build(embedder=HashingEmbedder()), hist_rag=PatientHistoryRAG(HashingEmbedder()))
    with TestClient(create_app(store=store, services=svc)) as c:
        yield c


def new_consultation(c, name="Asha"):
    pid = c.post("/patients", json={"name": name, "age": 40}).json()["id"]
    return pid, c.post("/consultations", json={"patient_id": pid}).json()["id"]


def audio(name="a.wav"):
    return {"file": (name, b"RIFFfake", "audio/wav")}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "mongo": True}


def test_patient_validation_and_404(client):
    assert client.post("/patients", json={"name": ""}).status_code == 422
    assert client.post("/patients", json={"name": "A", "age": 500}).status_code == 422
    assert client.get("/patients/nope").status_code == 404
    assert client.post("/consultations", json={"patient_id": "nope"}).status_code == 404


def test_full_process_flow(client):
    pid, cid = new_consultation(client)
    r = client.post(f"/consultations/{cid}/process", files=audio())
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["consultation"]["status"] == "pending_review"
    assert b["transcript"]["translated_text"] == "I have fever and cough"
    rep = b["report"]
    assert rep["clinical_summary"]["chief_complaint"] == "fever, cough"
    drugs = {d["drug"]: d for d in rep["draft_prescription"]}
    assert drugs["paracetamol"]["max_severity"] == "critical"
    assert drugs["amoxicillin"]["as_written"] == "Amoxicilin"
    assert any(f["type"] == "dose_too_high" and f["dismissed"] is False for f in rep["medication_flags"])
    assert len(b["extraction"]["medications"]) == 2          # parsed meds persisted
    assert b["extraction"]["diagnosis"] == "viral infection"  # rest of extraction intact


def test_step_by_step_matches_process(client):
    _, cid = new_consultation(client)
    assert client.post(f"/consultations/{cid}/audio", files=audio()).status_code == 200
    assert client.post(f"/consultations/{cid}/extract").json()["symptoms"] == "fever, cough"
    assert client.post(f"/consultations/{cid}/report/generate").status_code == 200
    assert client.get(f"/consultations/{cid}/report").json()["status"] == "pending_review"


def test_stage_order_errors(client):
    _, cid = new_consultation(client)
    assert client.post(f"/consultations/{cid}/extract").status_code == 404          # no transcript yet
    assert client.post(f"/consultations/{cid}/report/generate").status_code == 404  # no extraction yet
    assert client.post("/consultations/nope/audio", files=audio()).status_code == 404


def test_bad_audio_is_422(client):
    _, cid = new_consultation(client)
    assert client.post(f"/consultations/{cid}/audio", files=audio("x.bad")).status_code == 422


def test_regenerate_needs_overwrite(client):
    _, cid = new_consultation(client)
    client.post(f"/consultations/{cid}/process", files=audio())
    assert client.post(f"/consultations/{cid}/report/generate").status_code == 409
    assert client.post(f"/consultations/{cid}/report/generate?overwrite=true").status_code == 200


def test_review_edit_dismiss_finalize_lock(client):
    _, cid = new_consultation(client)
    client.post(f"/consultations/{cid}/process", files=audio())
    flags = client.get(f"/consultations/{cid}/report").json()["medication_flags"]
    flags[0]["dismissed"] = True
    r = client.patch(f"/consultations/{cid}/report", json={"medication_flags": flags})
    assert r.status_code == 200 and r.json()["medication_flags"][0]["dismissed"] is True
    assert client.patch(f"/consultations/{cid}/report", json={}).status_code == 422
    assert client.post(f"/consultations/{cid}/report/finalize", json={"approved_by": ""}).status_code == 422

    f = client.post(f"/consultations/{cid}/report/finalize", json={"approved_by": "Dr Sharma"})
    assert f.status_code == 200 and f.json()["status"] == "finalized" and f.json()["approved_by"] == "Dr Sharma"
    assert client.patch(f"/consultations/{cid}/report", json={"medication_flags": []}).status_code == 409
    assert client.post(f"/consultations/{cid}/report/finalize", json={"approved_by": "Dr X"}).status_code == 409
    assert client.post(f"/consultations/{cid}/process", files=audio()).status_code == 409  # finalized visit immutable
    assert client.get(f"/consultations/{cid}").json()["consultation"]["status"] == "finalized"


def test_history_appears_on_next_visit_only_after_finalize(client):
    pid, c1 = new_consultation(client)
    client.post(f"/consultations/{c1}/process", files=audio())
    c2 = client.post("/consultations", json={"patient_id": pid}).json()["id"]
    r = client.post(f"/consultations/{c2}/process", files=audio()).json()
    assert r["report"]["history_context"] == []                      # c1 not finalized yet
    client.post(f"/consultations/{c1}/report/finalize", json={"approved_by": "Dr A"})
    r = client.post(f"/consultations/{c2}/report/generate?overwrite=true").json()
    assert [h["consultation_id"] for h in r["history_context"]] == [c1]


def test_upload_size_limit(client, monkeypatch):
    import api.main as m
    monkeypatch.setattr(m, "MAX_UPLOAD_MB", 0)
    _, cid = new_consultation(client)
    assert client.post(f"/consultations/{cid}/audio", files=audio()).status_code == 413


def test_json_has_no_mongo_artifacts(client):
    pid, cid = new_consultation(client)
    b = client.get(f"/consultations/{cid}").json()
    assert "_id" not in b["consultation"] and isinstance(b["consultation"]["created_at"], str)
