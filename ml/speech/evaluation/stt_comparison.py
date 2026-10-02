"""
ml/speech/evaluation/stt_comparison.py — Optional STT Comparison (Phase 10)
=============================================================================
Experimental comparison of faster-whisper against Vosk (or another local
STT engine) on the same audio samples.

Compares:
  - WER
  - Latency (inference time)
  - Resource usage (memory)
  - Hindi / English / code-mixed quality

This is OPTIONAL and does NOT affect the main pipeline.
faster-whisper remains the primary implementation.

Public API:
    results = compare_stt_engines(audio_path, reference)
    report = format_comparison_report(all_results)
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ml.speech.evaluation.wer import calculate_wer
from ml.speech.schemas import WERResult

logger = logging.getLogger(__name__)


@dataclass
class STTBenchmarkResult:
    """Benchmark result for a single STT engine on a single sample."""
    engine: str
    audio_file: str
    language: str
    hypothesis: str = ""
    wer_result: Optional[WERResult] = None
    latency_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    error: Optional[str] = None
    success: bool = True


@dataclass
class ComparisonResult:
    """Comparison result for a single audio sample across engines."""
    audio_file: str
    language: str
    reference: str
    results: list[STTBenchmarkResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine runners
# ---------------------------------------------------------------------------

def _run_faster_whisper(audio_path: str, language: Optional[str] = None) -> str:
    """Run faster-whisper and return the transcript."""
    from ml.speech.stt import transcribe

    result = transcribe(audio_path, language=language)
    return result.text


def _run_vosk(audio_path: str, language: Optional[str] = None) -> str:
    """
    Run Vosk STT and return the transcript.
    Requires vosk and a downloaded model.

    Falls back gracefully if vosk is not installed.
    """
    try:
        import json
        import wave

        import vosk

        # Model paths (user must download these)
        model_paths = {
            "en": "models/vosk-model-small-en-us-0.15",
            "hi": "models/vosk-model-small-hi-0.22",
        }

        lang = language or "en"
        model_path = model_paths.get(lang, model_paths["en"])

        if not Path(model_path).exists():
            return f"[Vosk model not found at {model_path}]"

        model = vosk.Model(model_path)

        wf = wave.open(audio_path, "rb")
        rec = vosk.KaldiRecognizer(model, wf.getframerate())

        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                results.append(result.get("text", ""))

        final = json.loads(rec.FinalResult())
        results.append(final.get("text", ""))

        return " ".join(r for r in results if r).strip()

    except ImportError:
        return "[Vosk not installed — run: pip install vosk]"
    except Exception as e:
        return f"[Vosk error: {e}]"


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _measure_memory() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)  # Convert to MB (macOS reports bytes)
    except Exception:
        return 0.0


def benchmark_engine(
    engine_name: str,
    engine_fn,
    audio_path: str,
    reference: str,
    language: str,
) -> STTBenchmarkResult:
    """
    Benchmark a single STT engine on a single audio file.

    Measures latency, memory, and WER.
    """
    result = STTBenchmarkResult(
        engine=engine_name,
        audio_file=audio_path,
        language=language,
    )

    try:
        mem_before = _measure_memory()
        start_time = time.perf_counter()

        hypothesis = engine_fn(audio_path, language=language)

        result.latency_seconds = round(time.perf_counter() - start_time, 3)
        result.peak_memory_mb = round(_measure_memory() - mem_before, 1)
        result.hypothesis = hypothesis

        # Skip WER if engine returned an error message
        if not hypothesis.startswith("["):
            result.wer_result = calculate_wer(reference, hypothesis)
        else:
            result.success = False
            result.error = hypothesis

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error("Engine %s failed: %s", engine_name, traceback.format_exc())

    return result


def compare_stt_engines(
    audio_path: str | Path,
    reference: str,
    language: str = "english",
    engines: Optional[list[str]] = None,
) -> ComparisonResult:
    """
    Compare STT engines on a single audio sample.

    Args:
        audio_path: Path to audio file.
        reference: Ground-truth transcript.
        language: Language category.
        engines: List of engine names to test. Default: ["faster-whisper", "vosk"]

    Returns:
        ComparisonResult with benchmarks for each engine.
    """
    engines = engines or ["faster-whisper", "vosk"]
    audio_path = str(Path(audio_path).resolve())

    engine_fns = {
        "faster-whisper": _run_faster_whisper,
        "vosk": _run_vosk,
    }

    # Map language category to code
    lang_code_map = {"english": "en", "hindi": "hi", "mixed": "hi"}
    lang_code = lang_code_map.get(language.lower(), "en")

    comparison = ComparisonResult(
        audio_file=audio_path,
        language=language,
        reference=reference,
    )

    for engine_name in engines:
        fn = engine_fns.get(engine_name)
        if fn is None:
            logger.warning("Unknown engine: %s", engine_name)
            continue

        logger.info("Benchmarking %s on %s...", engine_name, Path(audio_path).name)
        result = benchmark_engine(engine_name, fn, audio_path, reference, lang_code)
        comparison.results.append(result)

    return comparison


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_comparison_report(comparisons: list[ComparisonResult]) -> str:
    """Format a comparison report across all samples and engines."""
    lines = []
    lines.append("=" * 75)
    lines.append("STT ENGINE COMPARISON REPORT")
    lines.append("=" * 75)
    lines.append("")

    # Aggregate by engine
    engine_stats: dict[str, dict] = {}
    for comp in comparisons:
        for r in comp.results:
            if r.engine not in engine_stats:
                engine_stats[r.engine] = {
                    "total_wer": 0.0,
                    "total_latency": 0.0,
                    "count": 0,
                    "errors": 0,
                }
            stats = engine_stats[r.engine]
            if r.success and r.wer_result:
                stats["total_wer"] += r.wer_result.wer
                stats["total_latency"] += r.latency_seconds
                stats["count"] += 1
            else:
                stats["errors"] += 1

    # Summary table
    lines.append(f"{'Engine':<20} {'Avg WER':>10} {'Avg Latency':>12} {'Samples':>8} {'Errors':>8}")
    lines.append("-" * 58)

    for engine, stats in engine_stats.items():
        if stats["count"] > 0:
            avg_wer = stats["total_wer"] / stats["count"]
            avg_lat = stats["total_latency"] / stats["count"]
            lines.append(
                f"{engine:<20} {avg_wer:>10.2%} {avg_lat:>10.2f}s "
                f"{stats['count']:>8} {stats['errors']:>8}"
            )
        else:
            lines.append(f"{engine:<20} {'N/A':>10} {'N/A':>12} {0:>8} {stats['errors']:>8}")

    lines.append("")

    # Per-sample details
    lines.append("-" * 75)
    lines.append("PER-SAMPLE DETAILS")
    lines.append("-" * 75)

    for comp in comparisons:
        lines.append(f"\nAudio: {comp.audio_file}")
        lines.append(f"Language: {comp.language}")
        lines.append(f"Reference: {comp.reference[:60]}...")
        for r in comp.results:
            wer_str = f"{r.wer_result.wer:.2%}" if r.wer_result else "N/A"
            lines.append(
                f"  {r.engine:<18} WER={wer_str:<8} "
                f"Latency={r.latency_seconds:.2f}s "
                f"{'✓' if r.success else '✗ ' + (r.error or '')}"
            )

    return "\n".join(lines)
