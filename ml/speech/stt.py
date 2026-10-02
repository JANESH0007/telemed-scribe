"""
ml/speech/stt.py — Speech-to-Text Module (Phase 2-3)
=====================================================
Integrates faster-whisper as the primary STT engine.

Features:
  - English, Hindi, Hindi-English code-mixed speech
  - Automatic language detection
  - Timestamps / segments
  - Configurable model size (tiny, base, small, medium, large-v2, large-v3)
  - CPU-friendly int8 quantization
  - Optional VAD / silence filtering
  - Safe transcript normalization (whitespace, formatting)
  - Raw transcript always preserved

Public API:
    result = transcribe(audio_path)
    result = transcribe(audio_path, model_size="small")

Model initialization is separated from inference — the model is loaded
once and cached as a module-level singleton.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ml.speech.config import WhisperConfig, get_config
from ml.speech.schemas import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton — loaded once, reused across calls
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def _get_model(config: Optional[WhisperConfig] = None):
    """
    Load and cache the faster-whisper model.
    Returns the cached model on subsequent calls.
    """
    from faster_whisper import WhisperModel

    config = config or get_config().whisper
    cache_key = f"{config.model_size}_{config.device}_{config.compute_type}"

    if cache_key not in _model_cache:
        logger.info(
            "Loading Whisper model: size=%s, device=%s, compute_type=%s",
            config.model_size,
            config.device,
            config.compute_type,
        )
        model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )
        _model_cache[cache_key] = model
        logger.info("Whisper model loaded successfully.")

    return _model_cache[cache_key]


def clear_model_cache() -> None:
    """Release all cached models (useful for testing / memory management)."""
    _model_cache.clear()
    logger.info("Whisper model cache cleared.")


# ---------------------------------------------------------------------------
# Transcript normalization (Phase 3)
# ---------------------------------------------------------------------------

def normalize_transcript(text: str) -> str:
    """
    Apply SAFE normalization only:
      - Collapse multiple whitespace/newlines into single space
      - Strip leading/trailing whitespace
      - Normalize Unicode whitespace characters
      - Fix obvious double-punctuation

    Does NOT aggressively rewrite or correct medical terminology.
    The raw transcript remains available in TranscriptResult.
    """
    if not text:
        return text

    # Normalize Unicode whitespace (non-breaking spaces, etc.)
    text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000]", " ", text)

    # Collapse multiple spaces/newlines into single space
    text = re.sub(r"\s+", " ", text)

    # Fix obvious double punctuation (e.g., ".. " → ". ")
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)

    return text.strip()


def join_segments(segments: list[TranscriptSegment]) -> str:
    """Join segment texts into a single string with proper spacing."""
    texts = [seg.text.strip() for seg in segments if seg.text.strip()]
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

# Human-readable language names
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}


def get_language_name(code: str) -> str:
    """Get human-readable language name from ISO code."""
    return LANGUAGE_NAMES.get(code, code.capitalize())


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str | Path,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    config: Optional[WhisperConfig] = None,
) -> TranscriptResult:
    """
    Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to audio file (WAV preferred, 16kHz mono).
        model_size: Override model size (default: from config).
        language: Force language code (default: auto-detect).
                  Use None to let the model detect the language.
        config: Override WhisperConfig.

    Returns:
        TranscriptResult containing:
          - text: normalized full transcript
          - language: detected language code
          - language_probability: detection confidence
          - segments: list of TranscriptSegment
          - duration: audio duration in seconds

    The original per-segment text is preserved exactly as the model
    returned it. The top-level `text` field has only safe normalization
    applied (whitespace cleanup, segment joining).
    """
    config = config or get_config().whisper

    # Override model size if requested
    if model_size:
        config = WhisperConfig(
            model_size=model_size,
            device=config.device,
            compute_type=config.compute_type,
            vad_enabled=config.vad_enabled,
            vad_threshold=config.vad_threshold,
        )

    audio_path = str(Path(audio_path).resolve())
    logger.info("Transcribing: %s (model=%s)", audio_path, config.model_size)

    # Load model (cached)
    model = _get_model(config)

    # Build transcription parameters
    transcribe_kwargs = {
        "beam_size": 5,
        "best_of": 5,
        "word_timestamps": False,
    }

    # VAD filtering
    if config.vad_enabled:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = {
            "threshold": config.vad_threshold,
            "min_silence_duration_ms": 500,
        }

    # Language override
    if language:
        transcribe_kwargs["language"] = language

    # Run transcription
    segments_generator, info = model.transcribe(audio_path, **transcribe_kwargs)

    # Collect segments
    transcript_segments = []
    for i, seg in enumerate(segments_generator):
        transcript_segments.append(
            TranscriptSegment(
                id=i,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text,
                avg_logprob=round(seg.avg_logprob, 4) if seg.avg_logprob else None,
                no_speech_prob=round(seg.no_speech_prob, 4) if seg.no_speech_prob else None,
            )
        )

    # Join and normalize
    raw_joined = join_segments(transcript_segments)
    normalized_text = normalize_transcript(raw_joined)

    result = TranscriptResult(
        text=normalized_text,
        language=info.language,
        language_probability=round(info.language_probability, 4),
        segments=transcript_segments,
        duration=round(info.duration, 3),
        model_size=config.model_size,
    )

    logger.info(
        "Transcription complete: lang=%s (%.2f), duration=%.1fs, segments=%d",
        result.language,
        result.language_probability,
        result.duration,
        len(result.segments),
    )

    return result
