"""
ml/speech/evaluation/wer.py — Word Error Rate Evaluation (Phase 8)
====================================================================
Calculates WER comparing reference transcripts against faster-whisper
output. Evaluates separately for English, Hindi, Hindi-English mixed,
and reports aggregate results.

Does not fabricate values — all metrics come from actual test data.

Public API:
    result = calculate_wer(reference, hypothesis)
    report = evaluate_dataset(samples)
    print_report(report)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jiwer

from ml.speech.schemas import EvaluationSample, WERResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WER Calculation
# ---------------------------------------------------------------------------

def calculate_wer(reference: str, hypothesis: str) -> WERResult:
    """
    Calculate Word Error Rate and related metrics.

    Args:
        reference: Ground-truth transcript.
        hypothesis: STT model output.

    Returns:
        WERResult with WER, MER, WIL, WIP, and edit operation counts.
    """
    if not reference.strip():
        logger.warning("Empty reference transcript — WER undefined.")
        return WERResult(
            wer=1.0 if hypothesis.strip() else 0.0,
            mer=1.0 if hypothesis.strip() else 0.0,
            wil=1.0 if hypothesis.strip() else 0.0,
            wip=0.0,
            substitutions=0,
            deletions=0,
            insertions=len(hypothesis.split()) if hypothesis.strip() else 0,
            reference_length=0,
            hypothesis_length=len(hypothesis.split()) if hypothesis.strip() else 0,
        )

    # Standard text transformations for WER
    transform = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    measures = jiwer.compute_measures(
        reference,
        hypothesis,
        truth_transform=transform,
        hypothesis_transform=transform,
    )

    return WERResult(
        wer=round(measures["wer"], 4),
        mer=round(measures["mer"], 4),
        wil=round(measures["wil"], 4),
        wip=round(measures["wip"], 4),
        substitutions=measures["substitutions"],
        deletions=measures["deletions"],
        insertions=measures["insertions"],
        reference_length=len(reference.split()),
        hypothesis_length=len(hypothesis.split()),
    )


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------

@dataclass
class LanguageWERReport:
    """WER report for a specific language category."""
    language: str
    sample_count: int = 0
    total_wer: float = 0.0
    avg_wer: float = 0.0
    min_wer: float = 1.0
    max_wer: float = 0.0
    individual_results: list[dict] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Full evaluation report across all languages."""
    english: LanguageWERReport = field(default_factory=lambda: LanguageWERReport(language="english"))
    hindi: LanguageWERReport = field(default_factory=lambda: LanguageWERReport(language="hindi"))
    mixed: LanguageWERReport = field(default_factory=lambda: LanguageWERReport(language="mixed"))
    aggregate: LanguageWERReport = field(default_factory=lambda: LanguageWERReport(language="aggregate"))


def evaluate_sample(
    sample: EvaluationSample,
    hypothesis: str,
) -> dict:
    """
    Evaluate a single sample.

    Args:
        sample: EvaluationSample with reference transcript.
        hypothesis: STT output for this sample.

    Returns:
        Dict with sample info and WER result.
    """
    wer_result = calculate_wer(sample.reference_transcript, hypothesis)

    return {
        "audio_file": sample.audio_file,
        "language": sample.language,
        "speaker_context": sample.speaker_context,
        "reference": sample.reference_transcript,
        "hypothesis": hypothesis,
        "wer": wer_result.wer,
        "substitutions": wer_result.substitutions,
        "deletions": wer_result.deletions,
        "insertions": wer_result.insertions,
        "wer_result": wer_result,
    }


def evaluate_dataset(
    results: list[dict],
) -> EvaluationReport:
    """
    Build an evaluation report from a list of sample results.

    Args:
        results: List of dicts from evaluate_sample().

    Returns:
        EvaluationReport with per-language and aggregate metrics.
    """
    report = EvaluationReport()

    for r in results:
        wer = r["wer"]
        lang = r["language"].lower()

        # Route to the right language bucket
        if lang == "english":
            lang_report = report.english
        elif lang == "hindi":
            lang_report = report.hindi
        elif lang in ("mixed", "code-mixed", "hindi-english"):
            lang_report = report.mixed
        else:
            logger.warning("Unknown language category: %s", lang)
            continue

        lang_report.sample_count += 1
        lang_report.total_wer += wer
        lang_report.min_wer = min(lang_report.min_wer, wer)
        lang_report.max_wer = max(lang_report.max_wer, wer)
        lang_report.individual_results.append(r)

        # Aggregate
        report.aggregate.sample_count += 1
        report.aggregate.total_wer += wer
        report.aggregate.min_wer = min(report.aggregate.min_wer, wer)
        report.aggregate.max_wer = max(report.aggregate.max_wer, wer)
        report.aggregate.individual_results.append(r)

    # Compute averages
    for lang_report in [report.english, report.hindi, report.mixed, report.aggregate]:
        if lang_report.sample_count > 0:
            lang_report.avg_wer = round(
                lang_report.total_wer / lang_report.sample_count, 4
            )

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(report: EvaluationReport) -> str:
    """Format the evaluation report as a human-readable table."""
    lines = []
    lines.append("=" * 70)
    lines.append("STT EVALUATION REPORT — Word Error Rate (WER)")
    lines.append("=" * 70)
    lines.append("")

    # Summary table
    lines.append(f"{'Language':<18} {'Samples':>8} {'Avg WER':>10} {'Min WER':>10} {'Max WER':>10}")
    lines.append("-" * 56)

    for lang_report in [report.english, report.hindi, report.mixed, report.aggregate]:
        if lang_report.sample_count > 0:
            name = lang_report.language.capitalize()
            if lang_report.language == "aggregate":
                lines.append("-" * 56)
                name = "AGGREGATE"
            lines.append(
                f"{name:<18} {lang_report.sample_count:>8} "
                f"{lang_report.avg_wer:>10.2%} "
                f"{lang_report.min_wer:>10.2%} "
                f"{lang_report.max_wer:>10.2%}"
            )

    lines.append("")
    lines.append("=" * 70)

    # Detailed results
    lines.append("")
    lines.append("DETAILED RESULTS")
    lines.append("-" * 70)

    for r in report.aggregate.individual_results:
        lines.append(f"\nAudio: {r['audio_file']}")
        lines.append(f"  Language: {r['language']}")
        lines.append(f"  WER:      {r['wer']:.2%}")
        lines.append(f"  S/D/I:    {r['substitutions']}/{r['deletions']}/{r['insertions']}")
        ref_preview = r["reference"][:80] + "..." if len(r["reference"]) > 80 else r["reference"]
        hyp_preview = r["hypothesis"][:80] + "..." if len(r["hypothesis"]) > 80 else r["hypothesis"]
        lines.append(f"  Ref:      {ref_preview}")
        lines.append(f"  Hyp:      {hyp_preview}")

    return "\n".join(lines)


def print_report(report: EvaluationReport) -> None:
    """Print the evaluation report to stdout."""
    print(format_report(report))


def save_report(report: EvaluationReport, output_path: str | Path) -> None:
    """Save the evaluation report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save human-readable report
    with open(path, "w") as f:
        f.write(format_report(report))

    # Save JSON alongside
    json_path = path.with_suffix(".json")
    json_data = {
        lang: {
            "sample_count": getattr(report, lang).sample_count,
            "avg_wer": getattr(report, lang).avg_wer,
            "min_wer": getattr(report, lang).min_wer,
            "max_wer": getattr(report, lang).max_wer,
        }
        for lang in ["english", "hindi", "mixed", "aggregate"]
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info("Report saved to %s and %s", path, json_path)
