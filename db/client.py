"""Mongo connection + index setup."""
from __future__ import annotations

import os
from functools import lru_cache

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database


@lru_cache(maxsize=1)
def get_db() -> Database:
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    name = os.getenv("MONGO_DB", "telemed_scribe")
    return MongoClient(uri, serverSelectionTimeoutMS=3000)[name]


def ensure_indexes(db: Database) -> None:
    """Idempotent. Call once at app startup."""
    db.consultations.create_index([("patient_id", ASCENDING), ("created_at", DESCENDING)])
    # one transcript / extraction / report per consultation (re-runs overwrite)
    for coll in ("transcripts", "extractions", "reports"):
        db[coll].create_index("consultation_id", unique=True)
    db.reports.create_index("status")
