"""
tests/conftest.py — Shared test fixtures
==========================================
Provides audio file fixtures, mock models, and temp directory management
for all test modules.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Prevent macOS crash due to duplicate OpenMP runtimes between faiss and torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ---------------------------------------------------------------------------
# Audio file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="telemed_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_wav_file(temp_dir: Path) -> Path:
    """Create a valid minimal WAV file (16kHz, mono, 1 second of silence)."""
    path = temp_dir / "test_valid.wav"
    _create_wav_file(path, sample_rate=16000, channels=1, duration_sec=1.0)
    return path


@pytest.fixture
def valid_wav_stereo(temp_dir: Path) -> Path:
    """Create a valid stereo WAV file (needs channel conversion)."""
    path = temp_dir / "test_stereo.wav"
    _create_wav_file(path, sample_rate=44100, channels=2, duration_sec=0.5)
    return path


@pytest.fixture
def valid_mp3_file(temp_dir: Path) -> Path:
    """Create a valid MP3 file (needs format conversion)."""
    # Generate WAV first, then convert to MP3 using pydub
    wav_path = temp_dir / "test_source.wav"
    _create_wav_file(wav_path, sample_rate=16000, channels=1, duration_sec=0.5)

    from pydub import AudioSegment
    from pydub.utils import which

    if not which("ffmpeg"):
        pytest.skip("ffmpeg not found, skipping MP3 test")

    audio = AudioSegment.from_wav(str(wav_path))
    mp3_path = temp_dir / "test_valid.mp3"
    audio.export(str(mp3_path), format="mp3")
    return mp3_path


@pytest.fixture
def empty_file(temp_dir: Path) -> Path:
    """Create an empty file."""
    path = temp_dir / "empty.wav"
    path.touch()
    return path


@pytest.fixture
def corrupted_file(temp_dir: Path) -> Path:
    """Create a corrupted audio file (random bytes with WAV extension)."""
    path = temp_dir / "corrupted.wav"
    path.write_bytes(b"NOT_A_REAL_WAV_FILE_HEADER_" + os.urandom(200))
    return path


@pytest.fixture
def unsupported_file(temp_dir: Path) -> Path:
    """Create a file with an unsupported extension."""
    path = temp_dir / "test.xyz"
    path.write_text("not audio")
    return path


@pytest.fixture
def too_small_file(temp_dir: Path) -> Path:
    """Create a file that is too small to be valid audio."""
    path = temp_dir / "tiny.wav"
    path.write_bytes(b"tiny")
    return path


# ---------------------------------------------------------------------------
# Mock fixtures for model-dependent tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_whisper_model():
    """Mock faster-whisper model to avoid loading real weights in unit tests."""
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.5
    mock_segment.text = " This is a test transcript."
    mock_segment.avg_logprob = -0.3
    mock_segment.no_speech_prob = 0.01

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.98
    mock_info.duration = 2.5

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

    return mock_model, mock_segment, mock_info


@pytest.fixture
def mock_whisper_hindi():
    """Mock faster-whisper model returning Hindi transcript."""
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 3.0
    mock_segment.text = " मुझे तीन दिन से बुखार है।"
    mock_segment.avg_logprob = -0.4
    mock_segment.no_speech_prob = 0.02

    mock_info = MagicMock()
    mock_info.language = "hi"
    mock_info.language_probability = 0.95
    mock_info.duration = 3.0

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

    return mock_model, mock_segment, mock_info


@pytest.fixture
def mock_translation_model():
    """Mock NLLB translation model."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_tokenizer.convert_tokens_to_ids.return_value = 256047  # eng_Latn token id
    mock_tokenizer.batch_decode.return_value = ["I have had fever for three days."]
    mock_tokenizer.src_lang = None

    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()

    return mock_model, mock_tokenizer


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _create_wav_file(
    path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    duration_sec: float = 1.0,
    bits_per_sample: int = 16,
) -> None:
    """Create a valid WAV file with silence."""
    num_samples = int(sample_rate * duration_sec)
    data_size = num_samples * channels * (bits_per_sample // 8)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))  # PCM format
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * channels * (bits_per_sample // 8)))
        f.write(struct.pack("<H", channels * (bits_per_sample // 8)))
        f.write(struct.pack("<H", bits_per_sample))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)  # silence
