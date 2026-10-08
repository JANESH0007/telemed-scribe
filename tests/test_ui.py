"""UI tests: pure helpers + headless Streamlit (AppTest) flows against the in-memory demo backend."""
import copy
import math
from pathlib import Path

import pytest

from ui.client import ApiError
from ui.logic import (build_edits, fmt_dt, merge_prescription, mermaid_to_dot, prescription_rows, report_to_text)

APP = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")
REVIEW, NEW = "📝 Review & records", "🩺 New consultation"

REPORT = {
    "status": "pending_review",
    "clinical_summary": {"chief_complaint": "fever", "diagnosis": "viral", "follow_up": "5 days", "language": "Hindi", "note": "n"},
    "draft_prescription": [
        {"drug": "paracetamol", "as_written": "Paracetamol", "dose": "5000 mg", "frequency_per_day": 2.0, "as_needed": False,
         "duration_days": 5, "reference_id": "r1", "flag_count": 1, "max_severity": "critical"},
        {"drug": "amoxicillin", "as_written": "Amoxicilin", "dose": "500 mg", "frequency_per_day": 3.0, "as_needed": False,
         "duration_days": None, "reference_id": "r2", "flag_count": 0, "max_severity": None}],
    "medication_flags": [{"drug": "Paracetamol", "dismissed": False, "type": "dose_too_high", "severity": "critical", "message": "m"},
                         {"drug": "Amoxicilin", "dismissed": False, "type": "name_corrected", "severity": "minor", "message": "m2"}],
}
SUMMARY = {"chief_complaint": "fever", "diagnosis": "viral", "follow_up": "5 days"}


def rows(report=REPORT):
    return copy.deepcopy(prescription_rows(report["draft_prescription"]))


# ---------------- pure logic ----------------
def test_no_changes_means_no_edits():
    assert build_edits(REPORT, SUMMARY, rows(), {}) == {}


def test_untouched_roundtrip_survives_nan_and_numpy_types():
    import numpy as np
    r = rows(); r[1]["duration_days"] = float("nan"); r[0]["duration_days"] = np.float64(5.0); r[0]["_i"] = np.int64(0)
    assert build_edits(REPORT, SUMMARY, r, {}) == {}


def test_summary_edit_keeps_other_summary_keys():
    e = build_edits(REPORT, {**SUMMARY, "diagnosis": " flu "}, rows(), {})
    assert set(e) == {"clinical_summary"}
    assert e["clinical_summary"]["diagnosis"] == "flu" and e["clinical_summary"]["note"] == "n"


def test_dose_edit_preserves_hidden_fields():
    r = rows(); r[0]["dose"] = "500 mg"
    rx = build_edits(REPORT, SUMMARY, r, {})["draft_prescription"]
    assert rx[0]["dose"] == "500 mg" and rx[0]["reference_id"] == "r1" and rx[0]["max_severity"] == "critical"
    assert rx[1] == REPORT["draft_prescription"][1]


def test_delete_and_add_rows():
    r = rows()[1:]                                               # doctor deleted paracetamol
    r.append({"_i": float("nan"), "drug": " Cetirizine ", "dose": "10 mg", "frequency_per_day": 1.0,
              "duration_days": 3.0, "as_needed": False})          # ...and added a drug
    rx = build_edits(REPORT, SUMMARY, r, {})["draft_prescription"]
    assert [d["drug"] for d in rx] == ["amoxicillin", "Cetirizine"]
    assert rx[1]["added_by_doctor"] is True and rx[1]["duration_days"] == 3 and rx[1]["as_written"] == "Cetirizine"
    assert rx[1]["reference_id"] is None and rx[1]["flag_count"] == 0


def test_blank_added_row_is_dropped():
    r = rows() + [{"_i": float("nan"), "drug": None, "dose": None, "frequency_per_day": float("nan"),
                   "duration_days": float("nan"), "as_needed": False}]
    assert build_edits(REPORT, SUMMARY, r, {}) == {}


def test_dismiss_flags():
    e = build_edits(REPORT, SUMMARY, rows(), {0: True, 1: False})
    assert [f["dismissed"] for f in e["medication_flags"]] == [True, False]
    assert e["medication_flags"][0]["type"] == "dose_too_high"


def test_merge_does_not_mutate_input():
    snap = copy.deepcopy(REPORT); r = rows(); r[0]["dose"] = "1 mg"
    merge_prescription(REPORT["draft_prescription"], r)
    assert REPORT == snap


def test_report_text_and_dates():
    t = report_to_text({"name": "Asha", "age": 40}, {"id": "abc", "created_at": "2026-10-07T10:30:00+00:00"},
                       {**REPORT, "approved_by": "Dr S", "finalized_at": "2026-10-07T11:00:00+00:00"})
    assert "Asha" in t and "Approved by:  Dr S" in t and "paracetamol" in t and "[CRITICAL]" in t and "2x/day" in t
    assert fmt_dt("2026-10-07T10:30:00+00:00") == "07 Oct 2026, 10:30" and fmt_dt(None) == "—" and fmt_dt("junk") == "junk"


def test_mermaid_to_dot_is_injection_safe():
    dot = mermaid_to_dot('a --> b;\nb -.-> c;\nx --> y"];}evil{;\n<script>alert(1)</script> --> z;')
    assert "a -> b;" in dot and "b -> c [style=dashed" in dot
    assert "evil" not in dot and "script" not in dot and "y" not in dot.replace("style", "")


def test_client_error_formatting():
    import httpx
    from ui.client import _detail
    r = httpx.Response(422, json={"detail": [{"loc": ["body", "approved_by"], "msg": "field required"}]})
    assert _detail(r) == "approved_by: field required"
    r = httpx.Response(422, json={"detail": [{"loc": ["body"], "msg": "Value error, approve requires approved_by"}]})
    assert _detail(r) == "approve requires approved_by"                       # model-level error: no stray ": "
    assert str(ApiError(409, "locked")) == "locked"


# ---------------- Streamlit flows (headless) ----------------
@pytest.fixture
def env(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import ui.demo_backend as demo
    st.cache_resource.clear()
    monkeypatch.setenv("TELEMED_DEMO", "1")
    client = demo.build_demo_client()
    monkeypatch.setattr(demo, "build_demo_client", lambda *a, **k: client)  # app + test share one backend

    def app(page, pid=None, cid=None):
        at = AppTest.from_file(APP, default_timeout=30)
        at.session_state["nav"] = page
        if pid:
            at.session_state["rev_pid"] = pid
        if cid:
            at.session_state["rev_cid"] = cid
        return at.run()

    def draft(size=3):  # size%3 picks the demo scenario: 0 = paracetamol 5000mg (critical), 2 = diabetes
        pid = client.list_patients()[0]["id"]
        cid = client.create_consultation(pid)["id"]
        client.run(cid, ("a.wav", b"x" * size, "audio/wav"))
        return pid, cid

    yield client, app, draft
    st.cache_resource.clear()


def btn(at, text):
    return next(b for b in at.button if text in b.label)


def test_pages_render_without_errors(env):
    client, app, _ = env
    for page in (NEW, REVIEW, "⚙️ System"):
        at = app(page)
        assert not at.exception and not at.error, page


def test_review_gate_then_approve(env):
    client, app, draft = env
    pid, cid = draft(3)
    at = app(REVIEW, pid, cid)
    assert not at.exception
    assert any("machine-translated" in w.value for w in at.warning)
    assert any("critical flag" in e.value for e in at.error)
    approve = lambda: btn(at, "Approve")
    assert approve().disabled
    next(t for t in at.text_input if t.label == "Approving doctor").input("Dr Sharma").run()
    next(c for c in at.checkbox if "I have reviewed" in c.label).check().run()
    assert approve().disabled                                    # critical flag still needs its own acknowledgement
    next(c for c in at.checkbox if "acknowledge" in c.label).check().run()
    assert not approve().disabled
    approve().click().run()
    assert not at.exception
    b = client.bundle(cid)
    assert b["consultation"]["status"] == "finalized" and b["report"]["approved_by"] == "Dr Sharma"
    assert any("Finalized by" in s.value for s in at.success)
    assert len(at.get("download_button")) == 2 and not [x for x in at.button if "Approve" in x.label]


def test_dismiss_flag_then_save(env):
    client, app, draft = env
    pid, cid = draft(3)
    at = app(REVIEW, pid, cid)
    assert btn(at, "Save").disabled
    next(c for c in at.checkbox if c.label == "Dismiss").check().run()   # first = most severe
    assert not btn(at, "Save").disabled
    btn(at, "Save").click().run()
    flags = client.bundle(cid)["report"]["medication_flags"]
    assert sum(f["dismissed"] for f in flags) == 1 and flags[0]["dismissed"] is True
    assert client.bundle(cid)["consultation"]["status"] == "pending_review"   # saving never finalizes
    assert btn(at, "Save").disabled                                           # clean again after the save


def test_approve_blocked_with_unsaved_edits(env):
    client, app, draft = env
    pid, cid = draft(2)                                           # diabetes: no critical flags
    at = app(REVIEW, pid, cid)
    next(t for t in at.text_area if t.label == "Diagnosis").input("type 2 diabetes, newly diagnosed").run()
    next(t for t in at.text_input if t.label == "Approving doctor").input("Dr A").run()
    next(c for c in at.checkbox if "I have reviewed" in c.label).check().run()
    assert btn(at, "Approve").disabled and btn(at, "Save").disabled is False
    btn(at, "Save").click().run()
    assert client.bundle(cid)["report"]["clinical_summary"]["diagnosis"] == "type 2 diabetes, newly diagnosed"


def test_finalized_visit_is_read_only(env):
    client, app, _ = env
    pid = next(p["id"] for p in client.list_patients() if p["name"] == "Asha Verma")   # seeded + finalized
    cid = client.list_consultations(pid)[0]["id"]
    at = app(REVIEW, pid, cid)
    assert not at.exception
    assert all(t.disabled for t in at.text_area) and all(c.disabled for c in at.checkbox)
    assert not [b for b in at.button if "Save" in b.label or "Approve" in b.label]


def test_history_shown_after_prior_finalized_visit(env):
    client, app, _ = env
    pid = next(p["id"] for p in client.list_patients() if p["name"] == "Asha Verma")
    cid = client.create_consultation(pid)["id"]
    client.run(cid, ("a.wav", b"xxx", "audio/wav"))               # same scenario as her finalized seed visit
    at = app(REVIEW, pid, cid)
    assert any("similarity" in m.value for m in at.markdown)


def test_resume_stopped_pipeline(env):
    client, app, _ = env
    pid = client.list_patients()[0]["id"]
    cid = client.create_consultation(pid)["id"]
    client.http.post(f"/consultations/{cid}/audio", files={"file": ("a.wav", b"xxx", "audio/wav")})
    at = app(REVIEW, pid, cid)
    assert any("stopped after" in w.value for w in at.warning)
    btn(at, "Resume").click().run()
    assert not at.exception and client.bundle(cid)["consultation"]["status"] == "pending_review"


def test_add_patient_and_start_consultation(env):
    client, app, _ = env
    at = app(NEW)
    # st.form commits its values together with the submit click, so no intermediate rerun (like a browser)
    next(t for t in at.text_input if t.label == "Full name").input("Meera Nair")
    btn(at, "Add patient").click().run()
    assert any(p["name"] == "Meera Nair" for p in client.list_patients())
    assert next(s for s in at.selectbox if s.label == "Patient").value == client.list_patients()[0]["id"]  # auto-selected
    btn(at, "Start a new consultation").click().run()
    pid = client.list_patients()[0]["id"]
    cons = client.list_consultations(pid)
    assert len(cons) == 1 and cons[0]["status"] == "created" and at.session_state["active_cid"] == cons[0]["id"]
    assert btn(at, "Process audio").disabled                      # nothing to process until audio is added
    assert not at.exception


def test_empty_name_rejected(env):
    client, app, _ = env
    n = len(client.list_patients())
    at = app(NEW)
    btn(at, "Add patient").click().run()
    assert any("Name is required" in e.value for e in at.error) and len(client.list_patients()) == n
