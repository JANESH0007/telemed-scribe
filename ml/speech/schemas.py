"""
ml/speech/schemas.py — Data models for Person A's speech pipeline.
===================================================================
These are the contracts consumed by Person B (Micro-LM) and
Person D (FastAPI / Streamlit).

All models are Pydantic v2 BaseModels for validation, serialization,
and clean dict/JSON export.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single timed segment from the STT engine."""

    id: int = Field(description="Segment index (0-based)")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Segment text")
    avg_logprob: Optional[float] = Field(
        default=None, description="Average log-probability of tokens"
    )
    no_speech_prob: Optional[float] = Field(
        default=None, description="Probability of no speech in segment"
    )

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


class TranscriptResult(BaseModel):
    """
    Output of the STT engine (Phase 2-3).
    Contains the raw transcript exactly as the model returned it.
    """

    text: str = Field(description="Full transcribed text")
    language: str = Field(description="Detected language code (e.g. 'en', 'hi')")
    language_probability: float = Field(
        description="Confidence of language detection (0.0-1.0)"
    )
    segments: list[TranscriptSegment] = Field(
        default_factory=list, description="Timed transcript segments"
    )
    duration: float = Field(description="Audio duration in seconds")
    model_size: str = Field(default="base", description="Whisper model size used")

    def to_dict(self) -> dict:
        """Export as a plain dict for downstream consumers."""
        return self.model_dump()


class ProcessedAudio(BaseModel):
    """Metadata and path for a preprocessed audio file."""

    original_path: str = Field(description="Original input file path")
    processed_path: str = Field(description="Path to processed (model-ready) file")
    sample_rate: int = Field(description="Sample rate in Hz")
    channels: int = Field(description="Number of audio channels")
    duration: float = Field(description="Duration in seconds")
    format: str = Field(description="File format (e.g. 'wav')")
    is_converted: bool = Field(
        default=False,
        description="True if the file was converted from its original format",
    )


class TranslationResult(BaseModel):
    """Output of the translation layer (Phase 4-5)."""

    original_text: str = Field(description="Text before translation")
    translated_text: str = Field(
        description="English translation (same as original if source is English)"
    )
    source_language: str = Field(description="Source language code")
    was_translated: bool = Field(
        description="True if translation was actually performed"
    )


class ConsultationResult(BaseModel):
    """
    Final output of the full pipeline (Phase 6).
    This is what Person B and Person D consume.

    Usage:
        result = process_consultation_audio("audio.wav")
        micro_lm_input = result.translated_text  # clean English string
    """

    original_text: str = Field(
        description="Raw transcript exactly as returned by the STT model"
    )
    language: str = Field(description="Detected language (e.g. 'Hindi', 'English')")
    language_code: str = Field(description="ISO 639-1 language code (e.g. 'hi', 'en')")
    language_probability: float = Field(description="Language detection confidence")
    translated_text: str = Field(
        description="English text — ready for Micro-LM consumption"
    )
    was_translated: bool = Field(
        description="True if translation was performed (False for English input)"
    )
    segments: list[TranscriptSegment] = Field(
        default_factory=list, description="Timed segments from STT"
    )
    duration: float = Field(description="Audio duration in seconds")

    def to_dict(self) -> dict:
        """Export as a plain dict for downstream consumers."""
        return self.model_dump()

    def for_micro_lm(self) -> str:
        """Return the clean English string for Person B's Micro-LM."""
        return self.translated_text


class WERResult(BaseModel):
    """Word Error Rate evaluation result."""

    wer: float = Field(description="Word Error Rate (0.0-1.0)")
    mer: float = Field(description="Match Error Rate")
    wil: float = Field(description="Word Information Lost")
    wip: float = Field(description="Word Information Preserved")
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = Field(description="Number of words in reference")
    hypothesis_length: int = Field(description="Number of words in hypothesis")


class EvaluationSample(BaseModel):
    """A single sample in the evaluation dataset."""

    audio_file: str = Field(description="Path to audio file")
    reference_transcript: str = Field(description="Ground-truth transcript")
    language: str = Field(description="Language: english | hindi | mixed")
    speaker_context: str = Field(
        default="", description="Speaker/context info (e.g. 'doctor', 'patient')"
    )
    notes: str = Field(default="", description="Additional notes about the sample")


class MedicalErrorEntry(BaseModel):
    """A single medical-term error found during analysis."""

    category: str = Field(
        description="Error category: medication | dosage | frequency | symptom | duration"
    )
    reference: str = Field(description="Correct term from reference transcript")
    hypothesis: str = Field(description="What the STT model produced")
    severity: str = Field(
        description="Error severity: critical | moderate | minor"
    )
    context: str = Field(default="", description="Surrounding sentence for context")
