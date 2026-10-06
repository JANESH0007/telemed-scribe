"""Tests for ml/rag (medication RAG, history RAG, service). No model download: uses HashingEmbedder."""
import pytest

from ml.rag.embedder import HashingEmbedder
from ml.rag.history_rag import PatientHistoryRAG
from ml.rag.medication_rag import MedicationRAG
from ml.rag.parsing import parse_medication, parse_prescriptions, split_prescriptions


@pytest.fixture(scope="module")
def rag():
    return MedicationRAG.build(embedder=HashingEmbedder())


def types(res):
    return {f.type.value for f in res.flags}


# ---------------- parsing ----------------
def test_split_keeps_instructions_with_their_drug():
    parts = split_prescriptions("Prescriptions: Paracetamol 500mg, twice daily, for 5 days; Tab Cetirizine 10 mg at night, Ibuprofen 400mg")
    assert parts == ["Paracetamol 500mg, twice daily, for 5 days", "Tab Cetirizine 10 mg at night", "Ibuprofen 400mg"]


@pytest.mark.parametrize("text,freq", [
    ("Paracetamol 500mg bd", 2), ("Paracetamol 500mg TDS", 3), ("Paracetamol 500mg once daily", 1),
    ("Paracetamol 500mg 1-0-1", 2), ("Paracetamol 500mg every 8 hours", 3), ("Paracetamol 500mg 4 times a day", 4),
])
def test_frequency_parsing(text, freq):
    assert parse_medication(text).frequency_per_day == freq


def test_parse_fields():
    m = parse_medication("Tab Amoxicillin 500 mg thrice daily for 1 week")
    assert (m.name, m.dose_value, m.dose_unit, m.frequency_per_day, m.duration_days) == ("Amoxicillin", 500, "mg", 3, 7)
    assert parse_medication("Paracetamol SOS").as_needed
    assert parse_medication("Pantoprazole 40 od").dose_value == 40
    assert parse_medication("Vitamin D3 60000 IU").name == "Vitamin D3"


def test_none_prescription():
    assert parse_prescriptions("None") == []


# ---------------- name resolution ----------------
def test_exact_and_brand_alias(rag):
    assert rag.check("Crocin 650 mg bd").matched.name == "paracetamol"
    assert rag.check("Paracetamol 500mg").match_method == "exact"


def test_typo_matched_but_flagged_for_confirmation(rag):
    r = rag.check("Amoxicilin 500mg tds")
    assert r.matched.name == "amoxicillin" and "name_corrected" in types(r)


def test_unknown_drug(rag):
    assert "unknown_drug" in types(rag.check("Xyzzyqq 10mg"))


def test_lookalike_not_trusted(rag):
    r = rag.check("Dexamethasone 4mg")  # not in dataset; must not silently become another drug
    assert "name_corrected" not in types(r) and ({"unknown_drug", "low_confidence_match"} & types(r))


# ---------------- dose checks ----------------
def test_normal_dose_clean(rag):
    assert rag.check("Paracetamol 500mg twice daily").flags == []
    assert rag.check("Amlodipine 5mg once daily").ok


def test_single_dose_too_high_critical(rag):
    r = rag.check("Paracetamol 2000mg")
    assert "dose_too_high" in types(r) and r.flags[0].severity == "critical"


def test_daily_total_exceeded(rag):
    r = rag.check("Ibuprofen 800mg 6 times a day")
    assert "daily_dose_exceeded" in types(r) or "frequency_unusual" in types(r)


def test_unit_error_caught_via_conversion(rag):
    r = rag.check("Levothyroxine 50 mg once daily")  # 50 mg = 50000 mcg
    assert "dose_too_high" in types(r) and r.flags[0].severity == "critical"


def test_dose_too_low_and_missing(rag):
    assert "dose_too_low" in types(rag.check("Amlodipine 0.5mg"))
    assert "missing_dose" in types(rag.check("Metformin"))


def test_prn_skips_daily_total(rag):
    assert "daily_dose_exceeded" not in types(rag.check("Paracetamol 1000mg SOS"))


def test_topical_skips_dose_check(rag):
    assert rag.check("Hydrocortisone cream 1%").flags == []


def test_high_alert_and_reference_attached(rag):
    r = rag.check("Warfarin 5mg")
    assert "high_alert_drug" in types(r) and r.matched.id and r.flags[0].reference_id == r.matched.id


def test_unit_incompatible(rag):
    assert "unit_mismatch" in types(rag.check("Paracetamol 5 ml"))


def test_check_prescription_text(rag):
    res = rag.check_prescription_text("Paracetamol 500mg bd, Cetirizine 10mg")
    assert [r.matched.name for r in res] == ["paracetamol", "cetirizine"]


def test_save_load_roundtrip(rag, tmp_path):
    rag.save(tmp_path)
    loaded = MedicationRAG.load(tmp_path, embedder=HashingEmbedder())
    assert loaded.check("Amoxicilin 500mg").matched.name == "amoxicillin"


def test_load_rejects_dim_mismatch(rag, tmp_path):
    rag.save(tmp_path)
    with pytest.raises(ValueError):
        MedicationRAG.load(tmp_path, embedder=HashingEmbedder(dim=64))


def test_reference_dataset_sane(rag):
    names = [e.name for e in rag.entries]
    assert len(names) == len(set(names)) >= 50
    for e in rag.entries:
        if e.dose_min is not None and e.dose_max is not None:
            assert e.dose_min <= e.dose_max, e.name
        if e.dose_max is not None and e.max_daily is not None:
            assert e.max_daily >= e.dose_max or e.name in {"clopidogrel"}, e.name


# ---------------- history RAG ----------------
VISITS = [
    {"consultation_id": "c1", "date": None, "symptoms": "fever, cough, sore throat", "diagnosis": "viral URTI", "prescriptions": "Paracetamol 500mg", "follow_up": "5 days"},
    {"consultation_id": "c2", "date": None, "symptoms": "knee pain on climbing stairs", "diagnosis": "osteoarthritis", "prescriptions": "Diclofenac 50mg", "follow_up": "2 weeks"},
    {"consultation_id": "c3", "date": None, "symptoms": "high blood pressure headache", "diagnosis": "hypertension", "prescriptions": "Amlodipine 5mg", "follow_up": "1 month"},
]


def test_history_ranks_similar_visit_first():
    h = PatientHistoryRAG(HashingEmbedder())
    hits = h.retrieve_from_visits(VISITS, "cough and fever since 3 days", k=2)
    assert hits[0].consultation_id == "c1" and len(hits) <= 2


def test_history_empty_cases():
    h = PatientHistoryRAG(HashingEmbedder())
    assert h.retrieve_from_visits([], "fever") == [] and h.retrieve_from_visits(VISITS, "  ") == []


def test_history_vector_cache_reuses_embeddings():
    calls = []
    class Spy(HashingEmbedder):
        def encode(self, texts):
            calls.append(len(texts)); return super().encode(texts)
    h = PatientHistoryRAG(Spy())
    h.retrieve_from_visits(VISITS, "fever"); h.retrieve_from_visits(VISITS, "knee pain")
    assert calls.count(3) == 1  # 3 visits embedded once, only queries re-embedded


# ---------------- service + Mongo (needs db/ from the DB layer) ----------------
def test_run_rag_end_to_end_scoped_per_patient():
    mongomock = pytest.importorskip("mongomock")
    from db import Store, ensure_indexes
    from db.models import Patient
    from ml.rag.service import run_rag

    store = Store(mongomock.MongoClient()["t"]); ensure_indexes(store.db)
    pa, pb = store.create_patient(Patient(name="A")), store.create_patient(Patient(name="B"))

    def finalized_visit(pid, ex):
        c = store.create_consultation(pid); store.save_extraction(c, ex); store.save_report(c, {}); store.finalize_report(c, "Dr X"); return c

    finalized_visit(pa, {"symptoms": "fever and cough", "diagnosis": "viral", "prescriptions": "Paracetamol 500mg", "follow_up": "5 days"})
    finalized_visit(pb, {"symptoms": "fever and cough", "diagnosis": "B's private visit", "prescriptions": "x", "follow_up": ""})
    cur = store.create_consultation(pa)
    store.save_extraction(cur, {"symptoms": "cough with fever", "diagnosis": "", "prescriptions": "Paracetamol 5000mg bd, Amoxicilin 500mg tds", "follow_up": ""})

    out = run_rag(store, cur, MedicationRAG.build(embedder=HashingEmbedder()), PatientHistoryRAG(HashingEmbedder()))
    assert len(out["history_context"]) == 1 and out["history_context"][0]["diagnosis"] == "viral"
    assert len(out["medications"]) == 2
    assert any(f["type"] == "dose_too_high" for m in out["medication_flags"] for f in m["flags"])
