"""
ml/speech/audio.py — Audio Preprocessing Module (Phase 1)
==========================================================
Handles:
  - Audio file validation (format, existence, corruption)
  - Duration and metadata extraction
  - Conversion to model-compatible format (16kHz mono WAV)
  - Configurable sample rate / channel handling
  - Temporary file cleanup

Public API:
    processed = process_audio(audio_path)
    processed = process_audio(audio_path, target_sample_rate=16000, target_channels=1)

This module does NOT depend on Streamlit or any web framework.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from pydub import AudioSegment

from ml.speech.config import AudioConfig, get_config
from ml.speech.schemas import ProcessedAudio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AudioProcessingError(Exception):
    """Raised when audio processing fails."""
    pass


class UnsupportedFormatError(AudioProcessingError):
    """Raised for unsupported audio formats."""
    pass


class AudioValidationError(AudioProcessingError):
    """Raised when audio file validation fails."""
    pass


class AudioDurationError(AudioProcessingError):
    """Raised when audio exceeds maximum duration."""
    pass


# ---------------------------------------------------------------------------
# Temporary file management
# ---------------------------------------------------------------------------

_temp_files: list[str] = []


def _register_temp_file(path: str) -> None:
    """Track a temporary file for later cleanup."""
    _temp_files.append(path)


def cleanup_temp_files() -> int:
    """
    Remove all tracked temporary files.
    Returns the number of files cleaned up.
    """
    count = 0
    for path in _temp_files:
        try:
            if os.path.exists(path):
                os.remove(path)
                count += 1
                logger.debug("Cleaned up temp file: %s", path)
        except OSError as e:
            logger.warning("Failed to clean up %s: %s", path, e)
    _temp_files.clear()
    return count


@contextmanager
def managed_audio(audio_path: str | Path) -> Generator[ProcessedAudio, None, None]:
    """
    Context manager that processes audio and cleans up temp files on exit.

    Usage:
        with managed_audio("consultation.mp3") as processed:
            # use processed.processed_path
        # temp files are cleaned up here
    """
    processed = process_audio(audio_path)
    try:
        yield processed
    finally:
        if processed.is_converted:
            try:
                if os.path.exists(processed.processed_path):
                    os.remove(processed.processed_path)
                    logger.debug("Cleaned up converted file: %s", processed.processed_path)
            except OSError as e:
                logger.warning("Cleanup failed for %s: %s", processed.processed_path, e)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_audio_file(
    audio_path: str | Path,
    config: Optional[AudioConfig] = None,
) -> Path:
    """
    Validate that the audio file exists, has a supported extension,
    and is not corrupted.

    Returns the resolved Path on success.
    Raises AudioValidationError or UnsupportedFormatError on failure.
    """
    config = config or get_config().audio
    path = Path(audio_path).resolve()

    # Existence check
    if not path.exists():
        raise AudioValidationError(f"Audio file not found: {path}")

    if not path.is_file():
        raise AudioValidationError(f"Path is not a file: {path}")

    # Format check
    suffix = path.suffix.lower()
    if suffix not in config.supported_formats:
        raise UnsupportedFormatError(
            f"Unsupported audio format '{suffix}'. "
            f"Supported: {', '.join(config.supported_formats)}"
        )

    # Size check — empty or suspiciously small
    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioValidationError(f"Audio file is empty: {path}")
    if file_size < 100:
        raise AudioValidationError(
            f"Audio file too small ({file_size} bytes), likely corrupted: {path}"
        )

    # Corruption check — try to load a small portion
    try:
        _format = _get_pydub_format(suffix)
        audio = AudioSegment.from_file(str(path), format=_format)
        if len(audio) == 0:
            raise AudioValidationError(f"Audio file has zero duration: {path}")
    except Exception as e:
        if isinstance(e, (AudioValidationError, UnsupportedFormatError)):
            raise
        raise AudioValidationError(
            f"Audio file appears corrupted or unreadable: {path}. Error: {e}"
        ) from e

    return path


def _get_pydub_format(suffix: str) -> str:
    """Map file extension to pydub format string."""
    format_map = {
        ".wav": "wav",
        ".mp3": "mp3",
        ".m4a": "m4a",
        ".flac": "flac",
        ".ogg": "ogg",
        ".webm": "webm",
    }
    return format_map.get(suffix, suffix.lstrip("."))


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def get_audio_metadata(audio_path: str | Path) -> dict:
    """
    Extract metadata from an audio file without full conversion.

    Returns:
        dict with keys: duration, sample_rate, channels, format, file_size
    """
    path = Path(audio_path).resolve()
    suffix = path.suffix.lower()
    _format = _get_pydub_format(suffix)

    audio = AudioSegment.from_file(str(path), format=_format)

    return {
        "duration": round(len(audio) / 1000.0, 3),  # ms → seconds
        "sample_rate": audio.frame_rate,
        "channels": audio.channels,
        "format": suffix.lstrip("."),
        "file_size": path.stat().st_size,
        "sample_width": audio.sample_width,  # bytes per sample
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_audio(
    audio_path: str | Path,
    target_sample_rate: Optional[int] = None,
    target_channels: Optional[int] = None,
    config: Optional[AudioConfig] = None,
) -> ProcessedAudio:
    """
    Validate and preprocess an audio file for the STT model.

    Steps:
        1. Validate the file (format, existence, corruption)
        2. Extract metadata (duration, sample rate, channels)
        3. Convert to 16kHz mono WAV if needed
        4. Check duration limits

    Args:
        audio_path: Path to the input audio file.
        target_sample_rate: Override target sample rate (default: from config).
        target_channels: Override target channels (default: from config).
        config: Override AudioConfig (default: from global config).

    Returns:
        ProcessedAudio with the path to the model-ready file.

    Raises:
        AudioValidationError: If the file is invalid or corrupted.
        UnsupportedFormatError: If the format is not supported.
        AudioDurationError: If the audio exceeds max duration.
    """
    config = config or get_config().audio
    sr = target_sample_rate or config.target_sample_rate
    ch = target_channels or config.target_channels

    # Step 1: Validate
    path = validate_audio_file(audio_path, config)
    logger.info("Processing audio: %s", path.name)

    # Step 2: Load audio
    suffix = path.suffix.lower()
    _format = _get_pydub_format(suffix)
    audio = AudioSegment.from_file(str(path), format=_format)

    duration_sec = round(len(audio) / 1000.0, 3)

    # Step 3: Duration check
    if config.max_duration > 0 and duration_sec > config.max_duration:
        raise AudioDurationError(
            f"Audio duration ({duration_sec:.1f}s) exceeds maximum "
            f"({config.max_duration}s)"
        )

    # Step 4: Determine if conversion is needed
    needs_conversion = (
        suffix != ".wav"
        or audio.frame_rate != sr
        or audio.channels != ch
    )

    if needs_conversion:
        logger.info(
            "Converting: %s (%dHz, %dch) → WAV (%dHz, %dch)",
            suffix, audio.frame_rate, audio.channels, sr, ch,
        )

        # Resample
        if audio.frame_rate != sr:
            audio = audio.set_frame_rate(sr)

        # Convert channels
        if audio.channels != ch:
            audio = audio.set_channels(ch)

        # Write to temp WAV
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="telemed_")
        os.close(temp_fd)
        audio.export(temp_path, format="wav")
        _register_temp_file(temp_path)

        logger.info("Converted audio saved to: %s", temp_path)

        return ProcessedAudio(
            original_path=str(path),
            processed_path=temp_path,
            sample_rate=sr,
            channels=ch,
            duration=duration_sec,
            format="wav",
            is_converted=True,
        )
    else:
        logger.info(
            "Audio already in target format (WAV, %dHz, %dch)", sr, ch
        )
        return ProcessedAudio(
            original_path=str(path),
            processed_path=str(path),
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            duration=duration_sec,
            format="wav",
            is_converted=False,
        )
