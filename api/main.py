"""
TeleMed-Scribe REST API (Person D).   uvicorn api.main:app --reload   ->  docs at /docs

Flow per consultation:  POST /patients -> POST /consultations -> POST .../process (audio) -> review -> finalize
Or step by step:        .../audio -> .../extract -> .../report/generate
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import pipeline
from api.schemas import ConsultationIn, FinalizeIn, PatientIn, ReportPatch
from api.serialization import to_json
from api.services import BadInput, Services
from db import NotFoundError, ReportLockedError, Store, ensure_indexes, get_db
from db.models import Patient

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))


def create_app(store: Optional[Store] = None, services: Optional[Services] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.store is None:
            db = get_db()
            ensure_indexes(db)
            app.state.store = Store(db)
        yield

    app = FastAPI(title="TeleMed-Scribe API", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.services = services or Services()
    app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                       allow_methods=["*"], allow_headers=["*"])

    def get_store(request: Request) -> Store:
        return request.app.state.store

    def get_services(request: Request) -> Services:
        return request.app.state.services

    # ---- error mapping ----
    def err(status: int):
        return lambda _r, e: JSONResponse({"detail": str(e) or e.__class__.__name__}, status_code=status)

    app.add_exception_handler(NotFoundError, lambda _r, e: JSONResponse({"detail": f"not found: {e}"}, status_code=404))
    app.add_exception_handler(ReportLockedError, lambda _r, e: JSONResponse({"detail": f"locked (finalized): {e}"}, status_code=409))
    app.add_exception_handler(FileExistsError, lambda _r, e: JSONResponse(
        {"detail": "report already exists; pass ?overwrite=true to regenerate (discards doctor edits)"}, status_code=409))
    app.add_exception_handler(BadInput, err(422))
    app.add_exception_handler(ValueError, err(422))

    def save_upload(file: UploadFile) -> str:
        suffix = Path(file.filename or "").suffix or ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix)
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    out.close(); os.unlink(path)
                    raise HTTPException(413, f"audio larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        return path

    # ---- health ----
    @app.get("/health")
    def health(s: Store = Depends(get_store)):
        try:
            s.db.command("ping"); mongo = True
        except Exception:
            mongo = False
        return {"status": "ok" if mongo else "degraded", "mongo": mongo}

    # ---- patients ----
    @app.post("/patients", status_code=201)
    def create_patient(body: PatientIn, s: Store = Depends(get_store)):
        pid = s.create_patient(Patient(**body.model_dump()))
        return to_json(s.get_patient(pid))

    @app.get("/patients/{patient_id}")
    def get_patient(patient_id: str, s: Store = Depends(get_store)):
        return to_json(s.get_patient(patient_id))

    @app.get("/patients/{patient_id}/consultations")
    def patient_consultations(patient_id: str, s: Store = Depends(get_store)):
        s.get_patient(patient_id)
        return to_json(s.list_consultations(patient_id))

    # ---- consultations ----
    @app.post("/consultations", status_code=201)
    def create_consultation(body: ConsultationIn, s: Store = Depends(get_store)):
        cid = s.create_consultation(body.patient_id)
        return to_json(s.get_bundle(cid)["consultation"])

    @app.get("/consultations/{cid}")
    def get_consultation(cid: str, s: Store = Depends(get_store)):
        return to_json(s.get_bundle(cid))

    # ---- pipeline stages ----
    @app.post("/consultations/{cid}/audio")
    def upload_audio(cid: str, file: UploadFile = File(...), language: Optional[str] = Query(None, description="force ISO code, e.g. hi"),
                     s: Store = Depends(get_store), svc: Services = Depends(get_services)):
        path = save_upload(file)
        try:
            return to_json(pipeline.transcribe_stage(s, svc, cid, path, language))
        finally:
            Path(path).unlink(missing_ok=True)

    @app.post("/consultations/{cid}/extract")
    def extract(cid: str, s: Store = Depends(get_store), svc: Services = Depends(get_services)):
        return to_json(pipeline.extract_stage(s, svc, cid))

    @app.post("/consultations/{cid}/report/generate")
    def generate_report(cid: str, overwrite: bool = False, s: Store = Depends(get_store), svc: Services = Depends(get_services)):
        return to_json(pipeline.report_stage(s, svc, cid, overwrite))

    @app.post("/consultations/{cid}/process")
    def process(cid: str, file: UploadFile = File(...), language: Optional[str] = Query(None), overwrite: bool = False,
                s: Store = Depends(get_store), svc: Services = Depends(get_services)):
        """Audio in -> transcript -> extraction -> RAG checks -> draft report (status pending_review)."""
        path = save_upload(file)
        try:
            pipeline.transcribe_stage(s, svc, cid, path, language)
        finally:
            Path(path).unlink(missing_ok=True)
        pipeline.extract_stage(s, svc, cid)
        pipeline.report_stage(s, svc, cid, overwrite)
        return to_json(s.get_bundle(cid))

    # ---- review ----
    @app.get("/consultations/{cid}/report")
    def get_report(cid: str, s: Store = Depends(get_store)):
        return to_json(s.get_report(cid))

    @app.patch("/consultations/{cid}/report")
    def edit_report(cid: str, body: ReportPatch, s: Store = Depends(get_store)):
        fields = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
        if not fields:
            raise HTTPException(422, "no fields to update")
        s.update_report(cid, fields)
        return to_json(s.get_report(cid))

    @app.post("/consultations/{cid}/report/finalize")
    def finalize(cid: str, body: FinalizeIn, s: Store = Depends(get_store)):
        s.finalize_report(cid, body.approved_by)
        return to_json(s.get_report(cid))

    return app


app = create_app()
