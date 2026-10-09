"""
ml/speech/pipeline.py — End-to-End Speech Pipeline (Phase 6)
==============================================================
Orchestrates:  audio → STT → language detection → translation → result

This is the single entry point for Person B (Micro-LM) and Person D
(FastAPI / Streamlit).

Public API:
    result = process_consultation_audio(audio_path)
    english_text = result.translated_text  # ← ready for Micro-LM
    result_dict = result.to_dict()         # ← ready for FastAPI

The pipeline does NOT invoke or implement the Micro-LM, RAG, or
any web framework components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ml.speech.audio import (
    AudioProcessingError,
    cleanup_temp_files,
    managed_audio,
    process_audio,
)
from ml.speech.config import get_config
from ml.speech.schemas import ConsultationResult, ProcessedAudio, TranscriptResult
from ml.speech.stt import get_language_name, transcribe
from ml.speech.translation import translate_to_english

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the speech pipeline encounters an error."""
    pass


def process_consultation_audio(
    audio_path: str | Path,
    model_size: Optional[str] = None,
    force_language: Optional[str] = None,
    skip_translation: bool = False,
) -> ConsultationResult:
    """
    Full pipeline: audio file → English transcript.

    Flow:
        1. Audio preprocessing (validation, format conversion)
        2. faster-whisper STT (transcription + language detection)
        3. Translation to English (if source is not English)
        4. Package result for downstream consumers

    Args:
        audio_path: Path to the input audio file (WAV, MP3, M4A, etc.)
        model_size: Override Whisper model size (default: from config).
        force_language: Force language detection to a specific code.
        skip_translation: If True, skip the translation step.

    Returns:
        ConsultationResult containing:
          - original_text: raw STT output
          - language: detected language name
          - language_code: ISO 639-1 code
          - translated_text: English text (for Micro-LM)
          - segments: timed transcript segments
          - duration: audio duration

    Usage by Person B:
        result = process_consultation_audio("consultation.mp3")
        micro_lm_input = result.translated_text
        # or equivalently:
        micro_lm_input = result.for_micro_lm()

    Usage by Person D:
        result = process_consultation_audio("consultation.mp3")
        api_response = result.to_dict()
    """
    try:
        logger.info("=" * 60)
        logger.info("PIPELINE START: %s", audio_path)
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # Step 1: Audio preprocessing
        # ------------------------------------------------------------------
        logger.info("Step 1/3: Audio preprocessing")
        with managed_audio(audio_path) as processed:
            audio_file = processed.processed_path
            duration = processed.duration

            logger.info(
                "  Audio ready: %.1fs, %dHz, %dch, converted=%s",
                processed.duration,
                processed.sample_rate,
                processed.channels,
                processed.is_converted,
            )

            # ----------------------------------------------------------
            # Step 2: Speech-to-Text
            # ----------------------------------------------------------
            logger.info("Step 2/3: Speech-to-Text (faster-whisper)")
            transcript = transcribe(
                audio_file,
                model_size=model_size,
                language=force_language,
            )

            logger.info(
                "  STT complete: lang=%s (%.2f), %d segments",
                transcript.language,
                transcript.language_probability,
                len(transcript.segments),
            )

            # ----------------------------------------------------------
            # Step 3: Translation (if needed)
            # ----------------------------------------------------------
            logger.info("Step 3/3: Translation")
            original_text = transcript.text
            lang_code = transcript.language

            if skip_translation or lang_code == "en":
                translated_text = original_text
                was_translated = False
                if lang_code == "en":
                    logger.info("  Source is English — skipping translation.")
                else:
                    logger.info("  Translation skipped (flag set).")
            else:
                logger.info(
                    "  Translating %s → English (segment by segment)",
                    get_language_name(lang_code),
                )
                translated_segments = []
                for seg in transcript.segments:
                    translated_segments.append(translate_to_english(seg.text, lang_code))
                translated_text = " ".join(translated_segments).strip()
                was_translated = True
                logger.info("  Translation complete.")

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        result = ConsultationResult(
            original_text=original_text,
            language=get_language_name(lang_code),
            language_code=lang_code,
            language_probability=transcript.language_probability,
            translated_text=translated_text,
            was_translated=was_translated,
            segments=transcript.segments,
            duration=transcript.duration,
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("  Language: %s (%s)", result.language, result.language_code)
        logger.info("  Translated: %s", result.was_translated)
        logger.info("  Output length: %d chars", len(result.translated_text))
        logger.info("=" * 60)

        return result

    except AudioProcessingError:
        # Re-raise audio errors as-is (they have clear messages)
        raise
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise PipelineError(f"Speech pipeline failed: {e}") from e


# ---------------------------------------------------------------------------
# Convenience functions for downstream consumers
# ---------------------------------------------------------------------------

def get_english_transcript(audio_path: str | Path, **kwargs) -> str:
    """
    Convenience function returning just the English text string.
    Useful for Person B's Micro-LM input.
    """
    result = process_consultation_audio(audio_path, **kwargs)
    return result.translated_text


def get_consultation_dict(audio_path: str | Path, **kwargs) -> dict:
    """
    Convenience function returning the result as a plain dict.
    Useful for Person D's FastAPI response.
    """
    result = process_consultation_audio(audio_path, **kwargs)
    return result.to_dict()
