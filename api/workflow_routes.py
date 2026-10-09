"""
LangGraph workflow endpoints (additive: the original stage endpoints in api/main.py are unchanged).

    POST /consultations/{cid}/workflow/run      audio (optional) -> draft report, pauses for review
    POST /consultations/{cid}/workflow/review   {"action": "edit"|"approve", ...} resumes the pause
    GET  /consultations/{cid}/workflow          status + last run's step trace
    GET  /workflow/graph                        Mermaid source of the graph
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse
from api.serialization import to_json
from db import Store

from workflow import ReviewDecision, WorkflowRunner

_init_lock = threading.Lock()


def get_runner(request: Request) -> WorkflowRunner:
    """Built lazily: the Store may only exist after the app's lifespan has started."""
    app = request.app
    with _init_lock:
        if getattr(app.state, "workflow", None) is None:
            app.state.workflow = WorkflowRunner(app.state.store, app.state.services)
    return app.state.workflow

def get_store(request: Request) -> Store:
    return request.app.state.store

def build_router(save_upload: Callable[[UploadFile], str]) -> APIRouter:
    router = APIRouter(tags=["workflow"])
    @router.get("/patients", tags=["patients"])
    def list_patients(limit: int = Query(200, ge=1, le=1000), s: Store = Depends(get_store)):
        """All patients, newest first (used by the Streamlit UI's patient picker)."""
        return to_json(s.list_patients(limit))

    @router.post("/consultations/{cid}/workflow/run")
    def run(cid: str, file: Optional[UploadFile] = File(None),
            language: Optional[str] = Query(None, description="force ISO code, e.g. hi"),
            overwrite: bool = False, runner: WorkflowRunner = Depends(get_runner)):
        """With an audio file: full pipeline. Without: restart from the stored transcript."""
        path = save_upload(file) if file is not None else None
        try:
            return runner.run(cid, audio_path=path, language=language, overwrite=overwrite)
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    @router.post("/consultations/{cid}/workflow/review")
    def review(cid: str, body: ReviewDecision, runner: WorkflowRunner = Depends(get_runner)):
        return runner.review(cid, body)

    @router.get("/consultations/{cid}/workflow")
    def status(cid: str, runner: WorkflowRunner = Depends(get_runner)):
        return runner.status(cid)

    @router.get("/workflow/graph", response_class=PlainTextResponse)
    def graph(runner: WorkflowRunner = Depends(get_runner)):
        return runner.mermaid()

    return router
