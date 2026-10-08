"""Thin HTTP client for the TeleMed-Scribe REST API. Works with a real server (httpx) or an in-process
fastapi TestClient (demo mode), since both expose the same httpx interface."""
from __future__ import annotations

from typing import Any, Optional

import httpx


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail

    def __str__(self) -> str:
        return self.detail


def _detail(resp: httpx.Response) -> str:
    try:
        d = resp.json().get("detail", resp.text)
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"
    if isinstance(d, list):  # FastAPI validation errors
        msgs = []
        for e in d:
            field = ".".join(str(x) for x in e.get("loc", [])[1:])
            msg = str(e.get("msg", "")).removeprefix("Value error, ")
            msgs.append(f"{field}: {msg}" if field else msg)
        return "; ".join(msgs)
    return str(d)


class TeleMedClient:
    def __init__(self, http: httpx.Client):
        self.http = http

    @classmethod
    def from_url(cls, url: str) -> "TeleMedClient":
        # STT + translation can take minutes on CPU, so a generous read timeout
        return cls(httpx.Client(base_url=url.rstrip("/"), timeout=httpx.Timeout(15.0, read=900.0)))

    def _call(self, method: str, path: str, **kw) -> Any:
        try:
            r = self.http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise ApiError(0, f"Cannot reach the API: {e.__class__.__name__}") from e
        if r.status_code >= 400:
            raise ApiError(r.status_code, _detail(r))
        ct = r.headers.get("content-type", "")
        return r.json() if "json" in ct else r.text

    # ---- system ----
    def health(self) -> dict:
        return self._call("GET", "/health")

    def graph(self) -> str:
        return self._call("GET", "/workflow/graph")

    # ---- patients / consultations ----
    def list_patients(self) -> list[dict]:
        return self._call("GET", "/patients")

    def create_patient(self, name: str, age: Optional[int], sex: Optional[str]) -> dict:
        body = {"name": name, "age": age, "sex": sex}
        return self._call("POST", "/patients", json={k: v for k, v in body.items() if v not in (None, "")})

    def list_consultations(self, patient_id: str) -> list[dict]:
        return self._call("GET", f"/patients/{patient_id}/consultations")

    def create_consultation(self, patient_id: str) -> dict:
        return self._call("POST", "/consultations", json={"patient_id": patient_id})

    def bundle(self, cid: str) -> dict:
        return self._call("GET", f"/consultations/{cid}")

    # ---- workflow ----
    def run(self, cid: str, audio: Optional[tuple[str, bytes, str]] = None,
            language: Optional[str] = None, overwrite: bool = False) -> dict:
        params: dict[str, Any] = {"overwrite": str(overwrite).lower()}
        if language:
            params["language"] = language
        files = {"file": audio} if audio else None
        return self._call("POST", f"/consultations/{cid}/workflow/run", params=params, files=files)

    def review_edit(self, cid: str, edits: dict) -> dict:
        return self._call("POST", f"/consultations/{cid}/workflow/review", json={"action": "edit", "edits": edits})

    def review_approve(self, cid: str, approved_by: str) -> dict:
        return self._call("POST", f"/consultations/{cid}/workflow/review",
                          json={"action": "approve", "approved_by": approved_by})
