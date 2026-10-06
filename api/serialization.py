"""Mongo documents -> JSON-safe dicts (datetimes to ISO strings, _id to id)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId


def to_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {("id" if k == "_id" else k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, ObjectId):  # transcripts/extractions/reports use auto _ids
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
