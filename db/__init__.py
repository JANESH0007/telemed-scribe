"""TeleMed-Scribe — MongoDB data layer (Person C)."""
from db.client import get_db, ensure_indexes
from db.repository import Store, ReportLockedError, NotFoundError

__all__ = ["get_db", "ensure_indexes", "Store", "ReportLockedError", "NotFoundError"]
