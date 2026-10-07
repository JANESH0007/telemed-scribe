"""
WorkflowRunner: the object the API (and the CLI demo) talk to.

Durability model
----------------
MongoDB is the system of record. The graph's checkpointer (in-memory) only remembers *where the
doctor-review pause is*. If that memory is gone (server restart, another uvicorn worker, a failed
resume) the review is re-attached from MongoDB: a report in `pending_review` is all that is needed.
So nothing is ever stranded, and no extra copy of patient data is written to disk.
"""
from __future__ import annotations

import threading
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from api.services import Services
from api.serialization import to_json
from db import NotFoundError, Store
from db.models import ConsultationStatus
from workflow.graph import build_graph
from workflow.state import ReviewDecision


class WorkflowRunner:
    def __init__(self, store: Store, services: Services, checkpointer: Optional[BaseCheckpointSaver] = None):
        self.store = store
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(store, services, self.checkpointer)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _cfg(self, cid: str) -> dict:
        return {"configurable": {"thread_id": cid}}  # one thread per consultation

    def _lock(self, cid: str) -> threading.Lock:
        """Serialize runs per consultation (two uploads for one visit must not interleave)."""
        with self._guard:
            return self._locks.setdefault(cid, threading.Lock())

    def _paused_at_review(self, cid: str) -> bool:
        snap = self.graph.get_state(self._cfg(cid))
        return snap.next == ("human_review",) and bool(snap.interrupts)

    def _result(self, cid: str, out: dict) -> dict:
        report = self.store.get_report(cid)
        awaiting = bool(out.get("__interrupt__"))
        return {
            "consultation_id": cid,
            "status": "awaiting_review" if awaiting else "finalized",
            "steps": out.get("trace", []),
            "warnings": report.get("clinical_summary", {}).get("warnings", []),
            "report": to_json(report),
        }

    # ------------------------------------------------------------------ public API
    def run(self, cid: str, audio_path: Optional[str] = None, language: Optional[str] = None,
            overwrite: bool = False) -> dict:
        """audio (if given) -> transcript -> extraction -> RAG -> draft report, then pause for review.
        Without audio it starts from the stored transcript (like POST /extract + /report/generate)."""
        with self._lock(cid):
            out = self.graph.invoke(
                {"consultation_id": cid, "entry": "audio" if audio_path else "extract",
                 "audio_path": audio_path, "language": language, "overwrite": overwrite},
                self._cfg(cid))
            return self._result(cid, out)

    def review(self, cid: str, decision: ReviewDecision) -> dict:
        """Resume the paused workflow with the doctor's decision (edit -> stays paused, approve -> done)."""
        with self._lock(cid):
            if not self._paused_at_review(cid):
                # No live pause (restart / other worker / earlier failure): rebuild it from MongoDB.
                # Raises NotFoundError / ReportLockedError if there is nothing reviewable.
                self.graph.invoke({"consultation_id": cid, "entry": "review"}, self._cfg(cid))
            try:
                out = self.graph.invoke(Command(resume=decision.model_dump()), self._cfg(cid))
            except Exception:
                # LangGraph keeps a failed resume value as a pending write and would replay it on the
                # next attempt, trapping the review on a stale decision. Drop the thread: MongoDB still
                # holds the pending_review report, so the next call re-attaches cleanly (see above).
                self._drop_thread(cid)
                raise
            result = self._result(cid, out)
            if result["status"] == "finalized":
                self._forget(cid)
            return result

    def status(self, cid: str) -> dict:
        c = self.store.db.consultations.find_one({"_id": cid})
        if not c:
            raise NotFoundError(f"consultation {cid}")
        return {
            "consultation_id": cid,
            "consultation_status": c["status"],
            # DB truth, so it is right even after a restart: the doctor can act iff a draft is pending
            "awaiting_review": c["status"] == ConsultationStatus.PENDING_REVIEW.value,
            "steps": self.graph.get_state(self._cfg(cid)).values.get("trace", []),  # last run in this process
        }

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def _drop_thread(self, cid: str) -> None:
        delete = getattr(self.checkpointer, "delete_thread", None)
        if delete:
            delete(cid)

    def _forget(self, cid: str) -> None:
        """Finalized visits are immutable: drop the thread so memory doesn't grow forever."""
        self._drop_thread(cid)
        with self._guard:
            self._locks.pop(cid, None)
