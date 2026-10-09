"""
ml/speech/config.py — Centralized configuration for Person A's module.
======================================================================
Reads from environment variables / .env file.
Model initialization is separated from inference so models are NOT
repeatedly loaded.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env from project root (if present)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


class WhisperConfig(BaseSettings):
    """Configuration for faster-whisper STT engine."""

    model_size: str = Field(
        default="large-v3",
        description="Whisper model size: tiny | base | small | medium | large-v2 | large-v3",
    )
    device: str = Field(default="cpu", description="Device: cpu | cuda")
    compute_type: str = Field(
        default="float32",
        description="Compute type: int8 (CPU-fast) | float16 (GPU) | float32 | default",
    )
    vad_enabled: bool = Field(default=True, description="Enable VAD filtering")
    vad_threshold: float = Field(
        default=0.5, description="VAD threshold (0.0-1.0)"
    )

    model_config = {"env_prefix": "WHISPER_"}


class TranslationConfig(BaseSettings):
    """Configuration for NLLB-200 translation model."""

    model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="HuggingFace model ID for NLLB-200",
    )
    device: str = Field(default="cpu")
    max_length: int = Field(default=512, description="Max output tokens")

    model_config = {
        "env_prefix": "TRANSLATION_",
        "extra": "ignore"
    }


class AudioConfig(BaseSettings):
    """Configuration for audio preprocessing."""

    target_sample_rate: int = Field(
        default=16000, description="Target sample rate in Hz"
    )
    target_channels: int = Field(default=1, description="Target channel count")
    max_duration: int = Field(
        default=600, description="Max audio duration in seconds (0 = unlimited)"
    )

    # Supported input formats
    supported_formats: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")

    model_config = {
        "env_prefix": "TARGET_",
        "extra": "ignore"
    }


class SpeechConfig(BaseSettings):
    """Top-level configuration aggregating all sub-configs."""

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    log_level: str = Field(default="INFO")
    eval_data_dir: str = Field(default="evaluation_data")

    model_config = {
        "env_prefix": "",
        "extra": "ignore"
    }


@lru_cache(maxsize=1)
def get_config() -> SpeechConfig:
    """Return the singleton SpeechConfig instance."""
    return SpeechConfig()
