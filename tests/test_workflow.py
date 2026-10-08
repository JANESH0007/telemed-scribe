"""LangGraph workflow tests. Same doubles as test_api.py: fake STT/extractor, hashing-embedder RAG, mongomock."""
import mongomock
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import BadInput, Services
from db import NotFoundError, ReportLockedError, Store, ensure_indexes
from ml.rag import MedicationRAG, PatientHistoryRAG
from ml.rag.embedder import HashingEmbedder
from workflow import ReviewDecision, WorkflowRunner

TRANSCRIPT = {"original_text": "bukhar", "translated_text": "I have fever and cough", "language": "Hindi",
              "language_code": "hi", "language_probability": 0.99, "was_translated": True, "duration": 4.0, "segments": []}
EXTRACTION = {"symptoms": "fever, cough", "diagnosis": "viral infection",
              "prescriptions": "Paracetamol 5000mg bd, Amoxicilin 500mg tds", "follow_up": "5 days", "raw_output": "x"}


def make_services(transcript=TRANSCRIPT, extraction=EXTRACTION):
    def transcriber(path, language):
        if path.endswith(".bad"):
            raise BadInput("unsupported audio format")
        return transcript
    return Services(transcriber=transcriber, extractor=lambda t: extraction,
                    med_rag=MedicationRAG.build(embedder=HashingEmbedder()),
                    hist_rag=PatientHistoryRAG(HashingEmbedder()))


@pytest.fixture
def store():
    s = Store(mongomock.MongoClient()["t"]); ensure_indexes(s.db); return s


@pytest.fixture
def runner(store):
    return WorkflowRunner(store, make_services())


def new_cid(store, name="Asha"):
    from db.models import Patient
    pid = store.create_patient(Patient(name=name)); return pid, store.create_consultation(pid)


def approve(by="Dr Sharma"):
    return ReviewDecision(action="approve", approved_by=by)


def steps(res):
    return [s["node"] for s in res["steps"]]


# ---------------- happy path ----------------
def test_full_run_pauses_for_review_with_ordered_steps(runner, store):
    _, cid = new_cid(store)
    res = runner.run(cid, audio_path="a.wav")
    assert res["status"] == "awaiting_review"
    assert steps(res) == ["init", "transcribe", "extract", "medication_rag", "history_rag", "assemble_report", "persist_report"]
    assert store.get_bundle(cid)["consultation"]["status"] == "pending_review"
    rep = res["report"]
    assert rep["clinical_summary"]["chief_complaint"] == "fever, cough"
    assert {d["drug"]: d for d in rep["draft_prescription"]}["paracetamol"]["max_severity"] == "critical"
    assert len(store.get_extraction(cid)["medications"]) == 2
    assert store.get_extraction(cid)["diagnosis"] == "viral infection"   # rest of extraction intact


def test_matches_existing_pipeline_output(store):
    """The graph must produce the same draft as api.pipeline (the code it replaces)."""
    from api import pipeline
    svc = make_services()
    _, a = new_cid(store, "A"); _, b = new_cid(store, "B")
    pipeline.transcribe_stage(store, svc, a, "a.wav", None); pipeline.extract_stage(store, svc, a)
    old = pipeline.report_stage(store, svc, a)
    new = WorkflowRunner(store, svc).run(b, audio_path="a.wav")["report"]
    for k in ("draft_prescription", "medication_flags", "history_context"):
        assert old[k] == new[k], k
    cs_old, cs_new = old["clinical_summary"], new["clinical_summary"]
    assert {k: v for k, v in cs_new.items() if k != "warnings"} == cs_old


def test_warnings_for_translated_transcript(runner, store):
    _, cid = new_cid(store)
    res = runner.run(cid, audio_path="a.wav")
    assert any("machine-translated" in w for w in res["warnings"])
    assert res["report"]["clinical_summary"]["warnings"] == res["warnings"]


def test_low_confidence_and_empty_extraction_warnings(store):
    svc = make_services({**TRANSCRIPT, "language_probability": 0.3, "was_translated": False},
                        {"symptoms": "", "diagnosis": "", "prescriptions": "", "follow_up": "", "raw_output": ""})
    _, cid = new_cid(store)
    w = WorkflowRunner(store, svc).run(cid, audio_path="a.wav")["warnings"]
    assert any("Low language-detection" in x for x in w) and any("no symptoms" in x for x in w)


# ---------------- review loop ----------------
def test_edit_then_approve_locks_report(runner, store):
    _, cid = new_cid(store)
    runner.run(cid, audio_path="a.wav")
    flags = store.get_report(cid)["medication_flags"]; flags[0]["dismissed"] = True
    r = runner.review(cid, ReviewDecision(action="edit", edits={"medication_flags": flags}))
    assert r["status"] == "awaiting_review" and r["report"]["medication_flags"][0]["dismissed"] is True
    r = runner.review(cid, approve())
    assert r["status"] == "finalized" and r["report"]["approved_by"] == "Dr Sharma"
    assert store.get_bundle(cid)["consultation"]["status"] == "finalized"
    with pytest.raises(ReportLockedError):
        runner.review(cid, approve("Dr X"))
    with pytest.raises(ReportLockedError):
        runner.run(cid, audio_path="a.wav")       # finalized visit is immutable


def test_decision_validation():
    with pytest.raises(ValueError):
        ReviewDecision(action="approve", approved_by="  ")
    with pytest.raises(ValueError):
        ReviewDecision(action="edit")
    with pytest.raises(ValueError):
        ReviewDecision(action="edit", edits={})


# ---------------- durability / failure paths ----------------
def test_review_survives_lost_checkpoint(store):
    """Simulates a server restart: a brand-new runner (empty checkpointer) can still finish the review."""
    svc = make_services()
    _, cid = new_cid(store)
    WorkflowRunner(store, svc).run(cid, audio_path="a.wav")
    fresh = WorkflowRunner(store, svc)
    assert fresh.status(cid)["awaiting_review"] is True
    assert fresh.review(cid, approve())["status"] == "finalized"


def test_failed_resume_does_not_strand_the_review(runner, store, monkeypatch):
    """A transient DB error while applying an edit must not trap the review on a stale decision."""
    _, cid = new_cid(store)
    runner.run(cid, audio_path="a.wav")
    real = store.update_report
    calls = {"n": 0}

    def flaky(c, fields):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("mongo blip")
        return real(c, fields)
    monkeypatch.setattr(store, "update_report", flaky)

    edit = ReviewDecision(action="edit", edits={"clinical_summary": {"diagnosis": "flu"}})
    with pytest.raises(RuntimeError):
        runner.review(cid, edit)
    r = runner.review(cid, edit)                                   # retry applies the NEW decision
    assert r["report"]["clinical_summary"]["diagnosis"] == "flu"
    assert runner.review(cid, approve())["status"] == "finalized"  # and approve still works


def test_overwrite_guard_runs_before_any_work(runner, store):
    _, cid = new_cid(store)
    runner.run(cid, audio_path="a.wav")
    before = store.get_transcript(cid)["saved_at"]
    with pytest.raises(FileExistsError):
        runner.run(cid, audio_path="a.wav")
    assert store.get_transcript(cid)["saved_at"] == before          # transcript untouched
    assert runner.run(cid, audio_path="a.wav", overwrite=True)["status"] == "awaiting_review"


def test_rerun_resets_trace(runner, store):
    _, cid = new_cid(store)
    runner.run(cid, audio_path="a.wav")
    res = runner.run(cid, audio_path="a.wav", overwrite=True)
    assert steps(res).count("init") == 1


def test_errors_propagate_for_http_mapping(runner, store):
    _, cid = new_cid(store)
    with pytest.raises(NotFoundError):
        runner.run("nope", audio_path="a.wav")
    with pytest.raises(NotFoundError):
        runner.run(cid)                                              # no audio and no stored transcript
    with pytest.raises(BadInput):
        runner.run(cid, audio_path="x.bad")
    with pytest.raises(NotFoundError):
        runner.review(cid, approve())                               # nothing to review yet


def test_start_from_stored_transcript(runner, store):
    _, cid = new_cid(store)
    store.save_transcript(cid, TRANSCRIPT)
    res = runner.run(cid)
    assert steps(res)[:2] == ["init", "extract"] and res["status"] == "awaiting_review"


def test_history_only_after_finalize(runner, store):
    pid, c1 = new_cid(store)
    runner.run(c1, audio_path="a.wav")
    c2 = store.create_consultation(pid)
    assert runner.run(c2, audio_path="a.wav")["report"]["history_context"] == []
    runner.review(c1, approve("Dr A"))
    r = runner.run(c2, audio_path="a.wav", overwrite=True)
    assert [h["consultation_id"] for h in r["report"]["history_context"]] == [c1]


def test_graph_has_expected_nodes(runner):
    m = runner.mermaid()
    for n in ("init", "transcribe", "extract", "medication_rag", "history_rag", "human_review", "finalize"):
        assert n in m


# ---------------- HTTP ----------------
@pytest.fixture
def client(store):
    with TestClient(create_app(store=store, services=make_services())) as c:
        yield c


def audio(name="a.wav"):
    return {"file": (name, b"RIFFfake", "audio/wav")}


def test_http_run_review_approve(client):
    pid = client.post("/patients", json={"name": "Asha"}).json()["id"]
    cid = client.post("/consultations", json={"patient_id": pid}).json()["id"]
    r = client.post(f"/consultations/{cid}/workflow/run", files=audio())
    assert r.status_code == 200 and r.json()["status"] == "awaiting_review"
    assert client.get(f"/consultations/{cid}/workflow").json()["awaiting_review"] is True
    assert client.post(f"/consultations/{cid}/workflow/run", files=audio()).status_code == 409
    assert client.post(f"/consultations/{cid}/workflow/review", json={"action": "approve"}).status_code == 422
    assert client.post(f"/consultations/{cid}/workflow/review", json={"action": "edit", "edits": {}}).status_code == 422
    e = client.post(f"/consultations/{cid}/workflow/review",
                    json={"action": "edit", "edits": {"clinical_summary": {"diagnosis": "flu"}}})
    assert e.status_code == 200 and e.json()["report"]["clinical_summary"]["diagnosis"] == "flu"
    f = client.post(f"/consultations/{cid}/workflow/review", json={"action": "approve", "approved_by": "Dr Sharma"})
    assert f.status_code == 200 and f.json()["status"] == "finalized"
    assert client.post(f"/consultations/{cid}/workflow/review", json={"action": "approve", "approved_by": "Dr X"}).status_code == 409
    assert client.get("/workflow/graph").status_code == 200


def test_http_error_mapping(client):
    pid = client.post("/patients", json={"name": "A"}).json()["id"]
    cid = client.post("/consultations", json={"patient_id": pid}).json()["id"]
    assert client.post("/consultations/nope/workflow/run", files=audio()).status_code == 404
    assert client.post(f"/consultations/{cid}/workflow/run", files=audio("x.bad")).status_code == 422
    assert client.post(f"/consultations/{cid}/workflow/run").status_code == 404   # no audio, no transcript
    assert client.get("/consultations/nope/workflow").status_code == 404

def test_http_list_patients(client):
    assert client.get("/patients").json() == []
    a = client.post("/patients", json={"name": "Asha"}).json()["id"]
    b = client.post("/patients", json={"name": "Ravi", "age": 30}).json()["id"]
    got = client.get("/patients").json()
    assert [p["id"] for p in got] == [b, a]                       # newest first
    assert "_id" not in got[0] and isinstance(got[0]["created_at"], str)
    assert client.get("/patients?limit=1").json()[0]["id"] == b
