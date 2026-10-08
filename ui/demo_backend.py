"""
Demo mode: the real FastAPI app + LangGraph workflow running in-process, on mongomock and canned
"models". Lets the UI be tried (and tested) with no MongoDB, no model downloads and no server.
"""
from __future__ import annotations

import os

import mongomock
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import Services
from db import Store, ensure_indexes
from ml.rag import MedicationRAG, PatientHistoryRAG
from ml.rag.embedder import HashingEmbedder
from ui.client import TeleMedClient

# (original text, english text, language, extraction). Chosen by audio size, so different uploads differ.
SCENARIOS = [
    ("मुझे तीन दिन से बुखार और खांसी है", "I have had fever and cough for three days", "Hindi", "hi", 0.97, True,
     {"symptoms": "fever, cough for 3 days", "diagnosis": "viral upper respiratory infection",
      "prescriptions": "Paracetamol 5000mg bd, Amoxicilin 500mg tds for 5 days", "follow_up": "5 days if not better"}),
    ("I have a headache and my BP was high at home", "I have a headache and my BP was high at home", "English", "en", 0.99, False,
     {"symptoms": "headache, high blood pressure readings", "diagnosis": "hypertension",
      "prescriptions": "Amlodipine 5mg once daily, Paracetamol 500mg SOS", "follow_up": "2 weeks with BP log"}),
    ("sugar high hai, thakan rehti hai", "Sugar is high, I feel tired all the time", "Hindi", "hi", 0.58, True,
     {"symptoms": "fatigue, raised blood sugar", "diagnosis": "type 2 diabetes mellitus",
      "prescriptions": "Metformin 500mg bd after meals", "follow_up": "1 month with HbA1c"}),
]
_BY_ENGLISH = {s[1]: s for s in SCENARIOS}


def _transcriber(path: str, language):
    orig, eng, lang, code, prob, translated, _ = SCENARIOS[os.path.getsize(path) % len(SCENARIOS)]
    return {"original_text": orig, "translated_text": eng, "language": lang, "language_code": code,
            "language_probability": prob, "was_translated": translated, "duration": 42.0, "segments": []}


def _extractor(text: str):
    return {**_BY_ENGLISH[text][6], "raw_output": "demo", "model": "demo-stub"}


def build_demo_client(seed: bool = True) -> TeleMedClient:
    store = Store(mongomock.MongoClient()["demo"])
    ensure_indexes(store.db)
    svc = Services(transcriber=_transcriber, extractor=_extractor,
                   med_rag=MedicationRAG.build(embedder=HashingEmbedder()),
                   hist_rag=PatientHistoryRAG(HashingEmbedder()))
    client = TeleMedClient(TestClient(create_app(store=store, services=svc)))
    if seed:  # one patient with a finalized past visit, so "patient history" has something to show
        pid = client.create_patient("Asha Verma", 42, "F")["id"]
        c1 = client.create_consultation(pid)["id"]
        client.run(c1, ("seed.wav", b"x" * 3, "audio/wav"))      # 3 bytes -> scenario 0
        client.review_approve(c1, "Dr Sharma")
        client.create_patient("Ravi Kumar", 58, "M")
    return client
