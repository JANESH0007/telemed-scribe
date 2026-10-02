"""
ml/speech/evaluation/error_analysis.py — Medical Error Analysis (Phase 9)
===========================================================================
Goes beyond WER to specifically check medical information accuracy.

Tracks errors involving:
  - Medication names (e.g., "paracetamol" vs "paracetol")
  - Dosage numbers (e.g., "500 mg" vs "50 mg")
  - Frequency (e.g., "twice daily" vs "twice a day")
  - Symptoms (e.g., "headache" vs "head ache")
  - Durations (e.g., "3 days" vs "three days")

A dosage error like 500mg → 50mg is clinically critical even if WER
barely changes, so this analysis is essential for a medical project.

Public API:
    errors = analyze_medical_errors(reference, hypothesis)
    report = format_medical_error_report(errors)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ml.speech.schemas import MedicalErrorEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Medical term patterns
# ---------------------------------------------------------------------------

# Common medications encountered in Indian telemedicine
COMMON_MEDICATIONS = [
    "paracetamol", "crocin", "dolo", "azithromycin", "amoxicillin",
    "metformin", "amlodipine", "atorvastatin", "omeprazole", "pantoprazole",
    "cetirizine", "montelukast", "ibuprofen", "diclofenac", "aceclofenac",
    "ranitidine", "domperidone", "ondansetron", "metoprolol", "losartan",
    "telmisartan", "aspirin", "clopidogrel", "insulin", "glimepiride",
    "sitagliptin", "empagliflozin", "levothyroxine", "prednisone",
    "prednisolone", "dexamethasone", "salbutamol", "budesonide",
    "fluticasone", "vitamin", "calcium", "iron", "folic acid", "zinc",
    "multivitamin", "b12", "b-complex", "ors",
]

# Dosage patterns
DOSAGE_PATTERN = re.compile(
    r"\b(\d+\.?\d*)\s*(mg|ml|mcg|g|iu|units?|tablets?|capsules?|drops?|cc)\b",
    re.IGNORECASE,
)

# Frequency patterns
FREQUENCY_TERMS = [
    "once", "twice", "thrice", "daily", "weekly", "monthly",
    "morning", "evening", "night", "afternoon", "bedtime",
    "before food", "after food", "before meals", "after meals",
    "empty stomach", "sos", "prn", "stat", "od", "bd", "tid", "qid",
    "once a day", "twice a day", "three times a day",
]

# Symptom terms
COMMON_SYMPTOMS = [
    "fever", "cough", "cold", "headache", "body ache", "fatigue",
    "nausea", "vomiting", "diarrhea", "diarrhoea", "constipation",
    "chest pain", "breathlessness", "shortness of breath",
    "abdominal pain", "stomach pain", "back pain", "joint pain",
    "sore throat", "runny nose", "congestion", "dizziness",
    "weakness", "weight loss", "weight gain", "swelling",
    "itching", "rash", "burning", "tingling", "numbness",
    "blood pressure", "sugar", "diabetes", "hypertension",
    "cholesterol", "thyroid", "asthma", "infection",
]

# Duration patterns
DURATION_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(days?|weeks?|months?|years?|hours?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _extract_medications(text: str) -> list[str]:
    """Extract medication names found in text."""
    text_lower = text.lower()
    found = []
    for med in COMMON_MEDICATIONS:
        if med in text_lower:
            found.append(med)
    return found


def _extract_dosages(text: str) -> list[str]:
    """Extract dosage patterns from text."""
    return [m.group(0) for m in DOSAGE_PATTERN.finditer(text)]


def _extract_frequencies(text: str) -> list[str]:
    """Extract frequency terms from text."""
    text_lower = text.lower()
    found = []
    for term in FREQUENCY_TERMS:
        if term in text_lower:
            found.append(term)
    return found


def _extract_symptoms(text: str) -> list[str]:
    """Extract symptom terms from text."""
    text_lower = text.lower()
    found = []
    for symptom in COMMON_SYMPTOMS:
        if symptom in text_lower:
            found.append(symptom)
    return found


def _extract_durations(text: str) -> list[str]:
    """Extract duration patterns from text."""
    return [m.group(0) for m in DURATION_PATTERN.finditer(text)]


def _classify_severity(category: str, reference: str, hypothesis: str) -> str:
    """
    Classify error severity based on clinical impact.

    Critical: Could change clinical decision (wrong dosage, wrong medication)
    Moderate: Could cause confusion (wrong symptom, wrong duration)
    Minor: Cosmetic / formatting differences
    """
    if category == "dosage":
        # Extract numbers and compare
        ref_nums = re.findall(r"\d+\.?\d*", reference)
        hyp_nums = re.findall(r"\d+\.?\d*", hypothesis)
        if ref_nums and hyp_nums:
            try:
                ref_val = float(ref_nums[0])
                hyp_val = float(hyp_nums[0])
                ratio = max(ref_val, hyp_val) / max(min(ref_val, hyp_val), 0.001)
                if ratio >= 5:  # 5x or more difference (e.g., 500 vs 50)
                    return "critical"
                elif ratio >= 2:
                    return "critical"
                else:
                    return "moderate"
            except (ValueError, ZeroDivisionError):
                pass
        return "critical"  # Dosage errors are always at least critical

    elif category == "medication":
        return "critical"  # Wrong medication name is always critical

    elif category == "frequency":
        return "moderate"

    elif category == "symptom":
        return "moderate"

    elif category == "duration":
        return "moderate"

    return "minor"


def analyze_medical_errors(
    reference: str,
    hypothesis: str,
    context: str = "",
) -> list[MedicalErrorEntry]:
    """
    Analyze differences between reference and hypothesis transcripts
    specifically for medical terminology errors.

    Args:
        reference: Ground-truth transcript.
        hypothesis: STT model output.
        context: Additional context about the sample.

    Returns:
        List of MedicalErrorEntry objects describing each error found.
    """
    errors = []

    # --- Medication errors ---
    ref_meds = set(_extract_medications(reference))
    hyp_meds = set(_extract_medications(hypothesis))

    # Medications in reference but missing from hypothesis
    for med in ref_meds - hyp_meds:
        errors.append(MedicalErrorEntry(
            category="medication",
            reference=med,
            hypothesis="[missing]",
            severity="critical",
            context=context or f"Medication '{med}' present in reference but not in STT output",
        ))

    # Medications in hypothesis but not in reference (hallucinated)
    for med in hyp_meds - ref_meds:
        errors.append(MedicalErrorEntry(
            category="medication",
            reference="[not present]",
            hypothesis=med,
            severity="critical",
            context=context or f"Medication '{med}' hallucinated by STT model",
        ))

    # --- Dosage errors ---
    ref_dosages = _extract_dosages(reference)
    hyp_dosages = _extract_dosages(hypothesis)

    if ref_dosages and hyp_dosages:
        # Compare dosages in order
        for i, ref_dose in enumerate(ref_dosages):
            if i < len(hyp_dosages):
                hyp_dose = hyp_dosages[i]
                if ref_dose.lower() != hyp_dose.lower():
                    errors.append(MedicalErrorEntry(
                        category="dosage",
                        reference=ref_dose,
                        hypothesis=hyp_dose,
                        severity=_classify_severity("dosage", ref_dose, hyp_dose),
                        context=context or f"Dosage mismatch: '{ref_dose}' → '{hyp_dose}'",
                    ))
            else:
                errors.append(MedicalErrorEntry(
                    category="dosage",
                    reference=ref_dose,
                    hypothesis="[missing]",
                    severity="critical",
                    context=context or f"Dosage '{ref_dose}' missing from STT output",
                ))
    elif ref_dosages and not hyp_dosages:
        for ref_dose in ref_dosages:
            errors.append(MedicalErrorEntry(
                category="dosage",
                reference=ref_dose,
                hypothesis="[missing]",
                severity="critical",
                context=context or f"All dosage information missing from STT output",
            ))

    # --- Frequency errors ---
    ref_freqs = set(_extract_frequencies(reference))
    hyp_freqs = set(_extract_frequencies(hypothesis))

    for freq in ref_freqs - hyp_freqs:
        errors.append(MedicalErrorEntry(
            category="frequency",
            reference=freq,
            hypothesis="[missing]",
            severity=_classify_severity("frequency", freq, ""),
            context=context or f"Frequency '{freq}' missing from STT output",
        ))

    # --- Symptom errors ---
    ref_symptoms = set(_extract_symptoms(reference))
    hyp_symptoms = set(_extract_symptoms(hypothesis))

    for symptom in ref_symptoms - hyp_symptoms:
        errors.append(MedicalErrorEntry(
            category="symptom",
            reference=symptom,
            hypothesis="[missing]",
            severity=_classify_severity("symptom", symptom, ""),
            context=context or f"Symptom '{symptom}' missing from STT output",
        ))

    # --- Duration errors ---
    ref_durations = _extract_durations(reference)
    hyp_durations = _extract_durations(hypothesis)

    if ref_durations and hyp_durations:
        for i, ref_dur in enumerate(ref_durations):
            if i < len(hyp_durations):
                hyp_dur = hyp_durations[i]
                if ref_dur.lower() != hyp_dur.lower():
                    errors.append(MedicalErrorEntry(
                        category="duration",
                        reference=ref_dur,
                        hypothesis=hyp_dur,
                        severity=_classify_severity("duration", ref_dur, hyp_dur),
                        context=context or f"Duration mismatch: '{ref_dur}' → '{hyp_dur}'",
                    ))
    elif ref_durations and not hyp_durations:
        for ref_dur in ref_durations:
            errors.append(MedicalErrorEntry(
                category="duration",
                reference=ref_dur,
                hypothesis="[missing]",
                severity="moderate",
                context=context or f"Duration '{ref_dur}' missing from STT output",
            ))

    logger.info(
        "Medical error analysis: found %d errors (%d critical)",
        len(errors),
        sum(1 for e in errors if e.severity == "critical"),
    )

    return errors


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_medical_error_report(
    all_errors: list[tuple[str, list[MedicalErrorEntry]]],
) -> str:
    """
    Format a medical error analysis report.

    Args:
        all_errors: List of (sample_name, errors) tuples.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MEDICAL TERM ERROR ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    total_errors = sum(len(errs) for _, errs in all_errors)
    critical_count = sum(
        sum(1 for e in errs if e.severity == "critical")
        for _, errs in all_errors
    )
    moderate_count = sum(
        sum(1 for e in errs if e.severity == "moderate")
        for _, errs in all_errors
    )

    lines.append(f"Total samples analyzed:  {len(all_errors)}")
    lines.append(f"Total medical errors:    {total_errors}")
    lines.append(f"  Critical:              {critical_count}")
    lines.append(f"  Moderate:              {moderate_count}")
    lines.append(f"  Minor:                 {total_errors - critical_count - moderate_count}")
    lines.append("")

    # Error breakdown by category
    category_counts = {}
    for _, errs in all_errors:
        for e in errs:
            category_counts[e.category] = category_counts.get(e.category, 0) + 1

    if category_counts:
        lines.append("Errors by category:")
        lines.append(f"  {'Category':<15} {'Count':>6}")
        lines.append(f"  {'-'*15} {'-'*6}")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:<15} {count:>6}")
        lines.append("")

    # Detailed errors per sample
    lines.append("-" * 70)
    lines.append("DETAILED ERRORS")
    lines.append("-" * 70)

    for sample_name, errs in all_errors:
        if not errs:
            continue
        lines.append(f"\n  Sample: {sample_name}")
        for e in errs:
            severity_marker = "🔴" if e.severity == "critical" else ("🟡" if e.severity == "moderate" else "🟢")
            lines.append(f"    {severity_marker} [{e.category}] {e.severity.upper()}")
            lines.append(f"       Reference:  {e.reference}")
            lines.append(f"       Hypothesis: {e.hypothesis}")
            if e.context:
                lines.append(f"       Context:    {e.context}")

    return "\n".join(lines)
