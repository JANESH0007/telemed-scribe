"""
TeleMed-Scribe — Streamlit interface.

    streamlit run ui/streamlit_app.py                       # uses http://localhost:8000, or demo mode if no API is up
    TELEMED_API_URL=http://host:8000 streamlit run ui/streamlit_app.py
    TELEMED_DEMO=1 streamlit run ui/streamlit_app.py        # force in-memory demo (no MongoDB / models needed)

The UI is a pure client of the REST API (api/main.py + api/workflow_routes.py); it holds no patient data itself.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `ui`, `api`, `workflow` import

import httpx
import pandas as pd
import streamlit as st

from ui.client import ApiError, TeleMedClient
from ui.logic import (
    PIPELINE_LABEL, PIPELINE_ORDER, SEVERITY_ICON, STATUS_LABEL, SUMMARY_FIELDS, build_edits, flag_indices_by_severity,
    fmt_dt, mermaid_to_dot, open_flags, prescription_rows, report_to_text,
)

st.set_page_config(page_title="TeleMed-Scribe", page_icon="🩺", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1250px}</style>", unsafe_allow_html=True)

NEW, REVIEW, SYSTEM = "🩺 New consultation", "📝 Review & records", "⚙️ System"
LANGUAGES = {"Auto-detect": None, "English": "en", "Hindi": "hi"}
OPEN_STATES = ("created", "transcribed", "extracted")


# ============================================================ connection
@st.cache_resource(show_spinner=False)
def _demo_client() -> TeleMedClient:
    from ui.demo_backend import build_demo_client
    return build_demo_client()


@st.cache_resource(show_spinner=False)
def _http_client(url: str) -> TeleMedClient:
    return TeleMedClient.from_url(url)


def _api_up(url: str) -> bool:
    try:
        return httpx.get(f"{url.rstrip('/')}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def connect() -> tuple[TeleMedClient, bool]:
    default_url = os.getenv("TELEMED_API_URL", "http://localhost:8000")
    if "demo_mode" not in st.session_state:  # decided once per session: demo if forced or no API is reachable
        st.session_state["demo_mode"] = os.getenv("TELEMED_DEMO") == "1" or not _api_up(default_url)
    with st.sidebar.expander("Connection"):
        demo = st.toggle("Demo mode (in-memory, stub models)", key="demo_mode")
        url = st.text_input("API URL", default_url, disabled=demo)
    return (_demo_client() if demo else _http_client(url)), demo


def safe(fn, *args, **kwargs):
    """Call the API; show a friendly error instead of a stack trace."""
    try:
        return fn(*args, **kwargs)
    except ApiError as e:
        st.error(e.detail)
        return None


def patient_label(p: dict) -> str:
    bits = [str(p["age"]) + "y" if p.get("age") is not None else None, p.get("sex")]
    extra = ", ".join(b for b in bits if b)
    return f"{p['name']}" + (f" ({extra})" if extra else "") + f" · {p['id'][:6]}"


# ============================================================ shared widgets
def render_steps(steps: list[dict]) -> None:
    done = {s["node"]: s["ms"] for s in steps}
    parts = [f"✅ **{PIPELINE_LABEL[n]}** ({done[n] / 1000:.1f}s)" for n in PIPELINE_ORDER if n in done]
    st.markdown("  →  ".join(parts) if parts else "_No steps recorded in this session._")


def severity_text(sev: str | None) -> str:
    return f"{SEVERITY_ICON.get(sev, '')} {sev}" if sev else ""


def open_review(pid: str, cid: str) -> None:  # on_click callback: runs before the next rerun
    st.session_state.update({"nav": REVIEW, "rev_pid": pid, "rev_cid": cid})


def open_new(pid: str, cid: str) -> None:
    st.session_state.update({"nav": NEW, "new_pid": pid, "active_cid": cid})


def do_run(client: TeleMedClient, cid: str, audio, language, overwrite=False):
    """Run the workflow with a live status box. Returns the result, or None (error already shown)."""
    with st.status("Running the AI pipeline…", expanded=True) as status:
        st.write("Transcribing → translating → extracting → checking medications → drafting the report. "
                 "On CPU this can take a few minutes for long recordings.")
        try:
            res = client.run(cid, audio, language, overwrite)
        except ApiError as e:
            status.update(label="Pipeline stopped", state="error")
            if e.status == 409 and "overwrite" in e.detail:
                st.session_state["confirm_overwrite"] = cid
            else:
                st.error(e.detail)
            return None
        status.update(label="Draft report ready for review", state="complete", expanded=False)
    st.session_state.pop("confirm_overwrite", None)
    return res


# ============================================================ page 1: new consultation
def page_new(client: TeleMedClient) -> None:
    st.header("New consultation")
    st.caption("Pick the patient, add the recording, and the AI prepares a draft for the doctor to review.")

    with st.expander("➕ Add a new patient", expanded=not st.session_state.get("seen_patients", True)):
        with st.form("new_patient", clear_on_submit=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            name = c1.text_input("Full name")
            age = c2.number_input("Age", min_value=0, max_value=130, value=None, step=1)
            sex = c3.selectbox("Sex", ["F", "M", "Other"], index=None, placeholder="Optional")
            if st.form_submit_button("Add patient", type="primary"):
                if not name.strip():
                    st.error("Name is required.")
                else:
                    p = safe(client.create_patient, name.strip(), int(age) if age is not None else None, sex)
                    if p:
                        st.session_state["new_pid"] = p["id"]
                        st.toast(f"Added {p['name']}", icon="✅")

    patients = safe(client.list_patients) or []
    st.session_state["seen_patients"] = bool(patients)
    if not patients:
        st.info("No patients yet. Add one above to begin.")
        return
    labels = {p["id"]: patient_label(p) for p in patients}
    if st.session_state.get("new_pid") not in labels:
        st.session_state.pop("new_pid", None)
    pid = st.selectbox("Patient", list(labels), format_func=labels.get, key="new_pid", index=0)

    consultations = safe(client.list_consultations, pid) or []
    open_ones = [c for c in consultations if c["status"] in OPEN_STATES]
    cid = st.session_state.get("active_cid")
    if cid and cid not in {c["id"] for c in consultations}:
        cid = st.session_state["active_cid"] = None  # belongs to another patient

    st.divider()
    if st.button("▶ Start a new consultation", type="primary" if not cid else "secondary"):
        cid = st.session_state["active_cid"] = safe(client.create_consultation, pid)["id"]
        st.session_state.pop("confirm_overwrite", None)
        st.rerun()
    others = [c for c in open_ones if c["id"] != cid]
    if others:
        pick = st.selectbox("…or continue an unfinished one", [None] + [c["id"] for c in others],
                            format_func=lambda i: "—" if i is None else
                            next(f"{fmt_dt(c['created_at'])} · {STATUS_LABEL[c['status']]}" for c in others if c["id"] == i))
        if pick:
            st.session_state["active_cid"] = pick
            st.rerun()
    if not cid:
        return

    bundle = safe(client.bundle, cid)
    if not bundle:
        return
    status = bundle["consultation"]["status"]
    st.success(f"Active consultation `{cid[:8]}` · {STATUS_LABEL[status]}")
    if status in ("pending_review", "finalized"):
        st.info("This consultation already has a draft report.")
        st.button("Open in review →", on_click=open_review, args=(pid, cid), key="open_existing")
        if status == "finalized":
            return

    # ---- audio ----
    st.subheader("1 · Consultation audio")
    t_up, t_rec = st.tabs(["⬆ Upload a file", "🎙 Record now"])
    audio = None
    with t_rec:
        rec = st.audio_input("Record the consultation")
    with t_up:
        up = st.file_uploader("WAV, MP3, M4A, FLAC or OGG", type=["wav", "mp3", "m4a", "flac", "ogg"])
        if up:
            st.audio(up)
    if up:
        audio = (up.name, up.getvalue(), up.type or "audio/wav")
    elif rec:
        audio = (rec.name or "recording.wav", rec.getvalue(), "audio/wav")
    if up and rec:
        st.caption("Both an upload and a recording are present; the uploaded file will be used.")
    lang_label = st.selectbox("Spoken language", list(LANGUAGES), help="Leave on auto-detect unless detection keeps getting it wrong.")
    language = LANGUAGES[lang_label]

    st.subheader("2 · Process")
    resumable = status in ("transcribed", "extracted")
    c1, c2 = st.columns(2)
    run_new = c1.button("🚀 Process audio", type="primary", disabled=audio is None)
    run_resume = c2.button("⏩ Resume from saved transcript", disabled=not resumable or audio is not None,
                           help="Skips speech-to-text; reuses the transcript already saved for this visit.")
    res = None
    if run_new:
        res = do_run(client, cid, audio, language)
    elif run_resume:
        res = do_run(client, cid, None, None)
    if st.session_state.get("confirm_overwrite") == cid:
        st.warning("This visit already has a draft report. Re-running **replaces it and discards any doctor edits**.")
        if st.button("Replace the existing draft and re-run", disabled=audio is None and not resumable):
            res = do_run(client, cid, audio, language, overwrite=True)
    if res:
        st.session_state.setdefault("last_run", {})[cid] = res

    last = st.session_state.get("last_run", {}).get(cid)
    if last:
        st.subheader("Result")
        render_steps(last["steps"])
        for w in last["warnings"]:
            st.warning(w)
        cs = last["report"]["clinical_summary"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Chief complaint", cs.get("chief_complaint") or "—")
        m2.metric("Diagnosis", cs.get("diagnosis") or "—")
        m3.metric("Open medication flags", len(open_flags(last["report"])))
        st.button("Open in doctor review →", type="primary", on_click=open_review, args=(pid, cid), key="open_after_run")


# ============================================================ page 2: review & records
def page_review(client: TeleMedClient) -> None:
    st.header("Review & records")
    patients = safe(client.list_patients) or []
    if not patients:
        st.info("No patients yet.")
        return
    labels = {p["id"]: patient_label(p) for p in patients}
    if st.session_state.get("rev_pid") not in labels:
        st.session_state.pop("rev_pid", None)
    c1, c2 = st.columns(2)
    pid = c1.selectbox("Patient", list(labels), format_func=labels.get, key="rev_pid", index=0)
    consultations = safe(client.list_consultations, pid) or []
    if not consultations:
        c2.info("No consultations for this patient yet.")
        return
    by_id = {c["id"]: c for c in consultations}
    if st.session_state.get("rev_cid") not in by_id:
        st.session_state.pop("rev_cid", None)
    cid = c2.selectbox("Consultation", list(by_id), key="rev_cid", index=0,
                       format_func=lambda i: f"{fmt_dt(by_id[i]['created_at'])} · {STATUS_LABEL[by_id[i]['status']]}")

    bundle = safe(client.bundle, cid)
    if not bundle:
        return
    status = bundle["consultation"]["status"]
    patient = next(p for p in patients if p["id"] == pid)

    if status == "created":
        st.info("No audio has been processed for this visit yet.")
        st.button("Go to recording →", on_click=open_new, args=(pid, cid))
    elif status in ("transcribed", "extracted"):
        st.warning(f"The pipeline stopped after **{status}**. You can resume it from the saved transcript.")
        render_transcript(bundle)
        if st.button("⏩ Resume pipeline", type="primary"):
            if do_run(client, cid, None, None):
                st.rerun()
    else:
        render_report(client, patient, bundle)


def render_transcript(bundle: dict) -> None:
    tr = bundle.get("transcript")
    if not tr:
        st.caption("No transcript.")
        return
    st.caption(f"Language: **{tr.get('language')}** (confidence {tr.get('language_probability', 0):.0%}) · "
               f"{tr.get('duration', 0):.0f}s of audio")
    if tr.get("was_translated"):
        t1, t2 = st.tabs(["English (used by the AI)", "Original transcript"])
        t1.write(tr.get("translated_text") or "—")
        t2.write(tr.get("original_text") or "—")
    else:
        st.write(tr.get("translated_text") or "—")


def render_report(client: TeleMedClient, patient: dict, bundle: dict) -> None:
    cons, report = bundle["consultation"], bundle["report"]
    cid = cons["id"]
    final = report["status"] == "finalized"
    base = f"{cid}:{report.get('updated_at', '')}"  # new key whenever the stored report changes -> fresh widgets
    cs = report.get("clinical_summary", {})

    h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
    h1.subheader(patient["name"])
    h2.markdown(f"**Status**  \n{STATUS_LABEL[cons['status']]}")
    h3.markdown(f"**Visit**  \n{fmt_dt(cons.get('created_at'))}")
    h4.markdown(f"**Language**  \n{cs.get('language') or '—'}")

    if final:
        st.success(f"✅ Finalized by **{report.get('approved_by')}** on {fmt_dt(report.get('finalized_at'))}. This record is locked.")
    else:
        st.warning("AI-generated draft. It must be reviewed and approved by a doctor before use.")
    for w in cs.get("warnings", []):
        st.warning(w, icon="⚠️")

    left, right = st.columns([3, 2], gap="large")

    # ---------------- right: context ----------------
    with right:
        st.markdown("#### Transcript")
        render_transcript(bundle)
        st.markdown("#### Patient history")
        hist = report.get("history_context", [])
        if not hist:
            st.caption("No similar finalized visits on record.")
        for h in hist:
            with st.container(border=True):
                st.markdown(f"**{fmt_dt(h.get('date'))}** · similarity {h.get('score', 0):.0%}")
                st.caption(f"Symptoms: {h.get('symptoms') or '—'}")
                st.caption(f"Diagnosis: {h.get('diagnosis') or '—'}")
                st.caption(f"Prescribed: {h.get('prescriptions') or '—'}")
        with st.expander("Raw extraction (model output)"):
            st.json(bundle.get("extraction") or {}, expanded=False)

    # ---------------- left: editable report ----------------
    with left:
        st.markdown("#### Clinical summary")
        summary = {}
        for key, label in SUMMARY_FIELDS:
            summary[key] = st.text_area(label, cs.get(key, ""), key=f"{base}:{key}", disabled=final, height=70)

        st.markdown("#### Prescription")
        rx = report.get("draft_prescription", [])
        cols = ["_i", "drug", "dose", "frequency_per_day", "duration_days", "as_needed", "as_written", "max_severity", "flag_count"]
        df = pd.DataFrame(prescription_rows(rx), columns=cols)
        df["_i"] = pd.to_numeric(df["_i"])
        df["frequency_per_day"] = pd.to_numeric(df["frequency_per_day"])
        df["duration_days"] = pd.to_numeric(df["duration_days"])
        df["as_needed"] = df["as_needed"].fillna(False).astype(bool)
        df["max_severity"] = df["max_severity"].map(severity_text)
        edited = st.data_editor(
            df, key=f"{base}:rx", hide_index=True, width="stretch",
            num_rows="fixed" if final else "dynamic",
            disabled=True if final else ["_i", "as_written", "max_severity", "flag_count"],
            column_config={
                "_i": None,
                "drug": st.column_config.TextColumn("Drug", required=True),
                "dose": st.column_config.TextColumn("Dose", help="e.g. 500 mg"),
                "frequency_per_day": st.column_config.NumberColumn("Times / day", min_value=0, step=0.5),
                "duration_days": st.column_config.NumberColumn("Days", min_value=0, step=1),
                "as_needed": st.column_config.CheckboxColumn("SOS"),
                "as_written": st.column_config.TextColumn("As heard"),
                "max_severity": st.column_config.TextColumn("Worst flag"),
                "flag_count": st.column_config.NumberColumn("Flags"),
            })
        rx_rows = edited.to_dict("records")

        st.markdown("#### Medication flags")
        flags = report.get("medication_flags", [])
        dismissed: dict[int, bool] = {}
        if not flags:
            st.success("No medication flags were raised.")
        for i in flag_indices_by_severity(flags):
            f = flags[i]
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"{SEVERITY_ICON.get(f.get('severity'), '')} **{f.get('drug')}** · "
                           f"{f.get('type', '').replace('_', ' ')}  \n{f.get('message')}")
                dismissed[i] = b.checkbox("Dismiss", bool(f.get("dismissed")), key=f"{base}:flag{i}", disabled=final,
                                          help="Mark as reviewed and not a concern.")
        if not final:
            st.caption("Flags describe the AI draft. If you change a drug or dose, check it yourself: flags are not recomputed.")

        # ---------------- actions ----------------
        if final:
            patient_doc = {**patient}
            st.download_button("⬇ Download report (.txt)", report_to_text(patient_doc, cons, report),
                               file_name=f"report_{cid[:8]}.txt", mime="text/plain")
            st.download_button("⬇ Download full record (.json)", json.dumps(bundle, indent=2, ensure_ascii=False),
                               file_name=f"record_{cid[:8]}.json", mime="application/json")
            return

        edits = build_edits(report, summary, rx_rows, dismissed)
        st.divider()
        s1, s2 = st.columns([1, 3])
        if s1.button("💾 Save edits", disabled=not edits, key=f"{base}:save"):
            if safe(client.review_edit, cid, edits):
                st.toast("Edits saved", icon="💾")
                st.rerun()
        s2.caption("You have unsaved changes." if edits else "No unsaved changes.")

        with st.container(border=True):
            st.markdown("#### Approve & finalize")
            crit = [f for f in open_flags(report) if f.get("severity") == "critical"]
            if crit:
                st.error(f"{len(crit)} critical flag(s) are still open. Resolve the prescription or dismiss them after review.")
            doctor = st.text_input("Approving doctor", key=f"{base}:doctor", placeholder="e.g. Dr A. Sharma")
            ok = st.checkbox("I have reviewed the transcript, summary, prescription and flags.", key=f"{base}:ok")
            ack = True
            if crit:
                ack = st.checkbox("I acknowledge the open critical flag(s) and take responsibility for this prescription.",
                                  key=f"{base}:ack")
            if edits:
                st.caption("Save your edits first.")
            if st.button("✅ Approve & finalize", type="primary",
                         disabled=not (doctor.strip() and ok and ack and not edits), key=f"{base}:approve"):
                if safe(client.review_approve, cid, doctor.strip()):
                    st.toast("Report finalized", icon="✅")
                    st.rerun()


# ============================================================ page 3: system
def page_system(client: TeleMedClient, demo: bool) -> None:
    st.header("System")
    h = safe(client.health)
    c1, c2, c3 = st.columns(3)
    c1.metric("API", (h or {}).get("status", "unreachable"))
    c2.metric("MongoDB", "connected" if (h or {}).get("mongo") else "down")
    c3.metric("Mode", "Demo (stub models)" if demo else "Live")
    if demo:
        st.info("Demo mode runs the real API and workflow in this process on an in-memory database, with canned "
                "speech/extraction results. Data disappears when the app restarts.")
    st.subheader("Workflow graph")
    st.caption("The LangGraph state machine that orchestrates the pipeline (served by `GET /workflow/graph`).")
    mermaid = safe(client.graph)
    if mermaid:
        st.graphviz_chart(mermaid_to_dot(mermaid), width="stretch")
        with st.expander("Mermaid source"):
            st.code(mermaid, language="text")


# ============================================================ main
def main() -> None:
    st.sidebar.title("🩺 TeleMed-Scribe")
    st.sidebar.caption("AI-assisted clinical notes for telemedicine")
    page = st.sidebar.radio("Navigate", [NEW, REVIEW, SYSTEM], key="nav", label_visibility="collapsed")
    client, demo = connect()
    if demo:
        st.sidebar.warning("Demo mode")
    st.sidebar.divider()
    st.sidebar.caption("Drafts are AI-generated and must be approved by a licensed physician.")
    if page == NEW:
        page_new(client)
    elif page == REVIEW:
        page_review(client)
    else:
        page_system(client, demo)


main()
