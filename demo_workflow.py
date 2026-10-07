"""
Run the whole LangGraph workflow from the command line (no MongoDB server needed: uses mongomock).

    python demo_workflow.py path/to/audio.mp3          # real models (Whisper / NLLB / T5 / RAG)
    python demo_workflow.py --stub                     # canned STT + extractor, instant, no model downloads
"""
import argparse

import mongomock

from api.services import Services
from db import Store, ensure_indexes
from db.models import Patient
from workflow import ReviewDecision, WorkflowRunner

STUB_TRANSCRIPT = {"original_text": "mujhe bukhar hai", "translated_text": "I have fever and cough", "language": "Hindi",
                   "language_code": "hi", "language_probability": 0.97, "was_translated": True, "duration": 4.0, "segments": []}
STUB_EXTRACTION = {"symptoms": "fever, cough", "diagnosis": "viral infection",
                   "prescriptions": "Paracetamol 5000mg bd, Amoxicilin 500mg tds", "follow_up": "5 days", "raw_output": "stub"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", default="stub.wav")
    ap.add_argument("--stub", action="store_true", help="use canned STT/extractor instead of real models")
    ap.add_argument("--approve-as", default="Dr Demo", help="name recorded on the finalized report")
    args = ap.parse_args()
    if not args.stub and args.audio == "stub.wav":
        ap.error("give an audio file, or pass --stub")

    store = Store(mongomock.MongoClient()["demo"]); ensure_indexes(store.db)
    svc = Services(transcriber=lambda p, l: STUB_TRANSCRIPT, extractor=lambda t: STUB_EXTRACTION) if args.stub else Services()
    if args.stub:  # offline RAG too
        from ml.rag import MedicationRAG, PatientHistoryRAG
        from ml.rag.embedder import HashingEmbedder
        svc = Services(transcriber=svc._transcriber, extractor=svc._extractor,
                       med_rag=MedicationRAG.build(embedder=HashingEmbedder()), hist_rag=PatientHistoryRAG(HashingEmbedder()))

    cid = store.create_consultation(store.create_patient(Patient(name="Demo Patient", age=40)))
    runner = WorkflowRunner(store, svc)

    print(f"\n▶ Running workflow on {args.audio} ...")
    res = runner.run(cid, audio_path=args.audio)
    print("\nSteps:  " + "  →  ".join(f"{s['node']} ({s['ms']}ms)" for s in res["steps"]))
    print(f"Status: {res['status']}  (paused for doctor review)")
    for w in res["warnings"]:
        print(f"  ⚠ {w}")
    rep = res["report"]
    print(f"\nDraft: {rep['clinical_summary']['chief_complaint']} | {rep['clinical_summary']['diagnosis']}")
    for d in rep["draft_prescription"]:
        print(f"  • {d['drug']} {d['dose']} x{d['frequency_per_day']}/day  [flags: {d['flag_count']}, worst: {d['max_severity']}]")

    print(f"\n▶ Doctor approves as '{args.approve_as}' ...")
    res = runner.review(cid, ReviewDecision(action="approve", approved_by=args.approve_as))
    print(f"Status: {res['status']}  approved_by={res['report']['approved_by']}\n")


if __name__ == "__main__":
    main()
