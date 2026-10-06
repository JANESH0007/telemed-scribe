"""Stage functions shared by the single-step endpoints and /process. Each reads/writes only through Store."""
from __future__ import annotations

from typing import Optional

from ml.rag.service import run_rag

from api.report import build_draft_report
from api.services import Services


def transcribe_stage(store, svc: Services, cid: str, audio_path: str, language: Optional[str]) -> dict:
    store.get_bundle(cid)  # 404 early, before running STT
    store.save_transcript(cid, svc.transcribe(audio_path, language))
    return store.get_transcript(cid)


def extract_stage(store, svc: Services, cid: str) -> dict:
    text = store.get_transcript(cid).get("translated_text", "")
    store.save_extraction(cid, svc.extract(text))
    return store.get_extraction(cid)


def report_stage(store, svc: Services, cid: str, overwrite: bool = False) -> dict:
    from db import NotFoundError
    ex = store.get_extraction(cid)
    try:
        store.get_report(cid)
        if not overwrite:
            raise FileExistsError(cid)  # mapped to 409: don't silently discard doctor edits
    except NotFoundError:
        pass
    out = run_rag(store, cid, svc.med_rag, svc.hist_rag)
    transcript = store.db.transcripts.find_one({"consultation_id": cid})
    # persist parsed medications on the extraction (full fields, so any db version keeps the rest)
    store.save_extraction(cid, {**{k: ex.get(k, "") for k in ("symptoms", "diagnosis", "prescriptions", "follow_up", "raw_output", "model")},
                                "medications": out["medications"]})
    store.save_report(cid, build_draft_report(ex, out, transcript))
    return store.get_report(cid)
