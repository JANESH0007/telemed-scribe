"""
evaluation_data/run_evaluation.py — Run full STT evaluation
=============================================================
Runs faster-whisper on all evaluation audio samples, calculates WER
per language, and produces the medical error analysis report.

Usage:
    python -m evaluation_data.run_evaluation

Outputs:
    evaluation_data/reports/wer_report.txt
    evaluation_data/reports/wer_report.json
    evaluation_data/reports/medical_errors.txt
    evaluation_data/reports/comparison.txt (if Vosk is available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = Path(__file__).parent
REFS_DIR = EVAL_DIR / "references"
REPORTS_DIR = EVAL_DIR / "reports"


def run_evaluation():
    """Run the complete evaluation pipeline."""
    from ml.speech.evaluation.error_analysis import (
        analyze_medical_errors,
        format_medical_error_report,
    )
    from ml.speech.evaluation.wer import (
        evaluate_dataset,
        evaluate_sample,
        format_report,
        save_report,
    )
    from ml.speech.schemas import EvaluationSample
    from ml.speech.stt import transcribe

    REPORTS_DIR.mkdir(exist_ok=True)

    all_results = []
    all_medical_errors = []

    # Process each language category
    for ref_file in sorted(REFS_DIR.glob("*.json")):
        print(f"\n{'='*60}")
        print(f"Evaluating: {ref_file.stem}")
        print(f"{'='*60}")

        with open(ref_file) as f:
            samples_data = json.load(f)

        for sample_data in samples_data:
            audio_path = EVAL_DIR / sample_data["audio_file"]

            sample = EvaluationSample(**sample_data)

            if not audio_path.exists():
                print(f"  SKIP (no audio): {audio_path.name}")
                continue

            print(f"\n  Processing: {audio_path.name}")
            print(f"  Language: {sample.language}")

            try:
                # Run STT
                transcript = transcribe(str(audio_path))
                hypothesis = transcript.text

                print(f"  Detected lang: {transcript.language} ({transcript.language_probability:.2f})")
                print(f"  Reference:  {sample.reference_transcript[:60]}...")
                print(f"  Hypothesis: {hypothesis[:60]}...")

                # WER
                result = evaluate_sample(sample, hypothesis)
                all_results.append(result)
                print(f"  WER: {result['wer']:.2%}")

                # Medical error analysis
                errors = analyze_medical_errors(
                    sample.reference_transcript,
                    hypothesis,
                    context=f"{sample.language} - {audio_path.name}",
                )
                if errors:
                    all_medical_errors.append((audio_path.name, errors))
                    print(f"  Medical errors: {len(errors)} ({sum(1 for e in errors if e.severity == 'critical')} critical)")

            except Exception as e:
                print(f"  ERROR: {e}")

    # Generate WER report
    if all_results:
        report = evaluate_dataset(all_results)
        report_path = REPORTS_DIR / "wer_report.txt"
        save_report(report, report_path)
        print(f"\n\nWER report saved to: {report_path}")
        print(format_report(report))

    # Generate medical error report
    if all_medical_errors:
        medical_report = format_medical_error_report(all_medical_errors)
        medical_path = REPORTS_DIR / "medical_errors.txt"
        with open(medical_path, "w") as f:
            f.write(medical_report)
        print(f"\nMedical error report saved to: {medical_path}")
        print(medical_report)

    # Optional: STT comparison
    try:
        _run_comparison(all_results)
    except Exception as e:
        print(f"\nSTT comparison skipped: {e}")


def _run_comparison(wer_results: list[dict]):
    """Run optional STT comparison if Vosk is available."""
    from ml.speech.evaluation.stt_comparison import (
        compare_stt_engines,
        format_comparison_report,
    )

    comparisons = []
    for r in wer_results[:3]:  # Compare on first 3 samples only
        audio_path = EVAL_DIR / r["audio_file"]
        if audio_path.exists():
            comp = compare_stt_engines(
                str(audio_path),
                r["reference"],
                r["language"],
            )
            comparisons.append(comp)

    if comparisons:
        comp_report = format_comparison_report(comparisons)
        comp_path = REPORTS_DIR / "comparison.txt"
        with open(comp_path, "w") as f:
            f.write(comp_report)
        print(f"\nComparison report saved to: {comp_path}")


if __name__ == "__main__":
    run_evaluation()
