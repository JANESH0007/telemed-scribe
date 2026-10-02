"""
tests/test_stt.py — Speech-to-Text Tests (Phase 11)
=====================================================
Tests for:
  - Transcript returned
  - Language returned
  - Segments returned
  - Normalization
  - Model caching

Uses mocks to avoid loading real Whisper models in unit tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.speech.schemas import TranscriptResult, TranscriptSegment
from ml.speech.stt import (
    clear_model_cache,
    get_language_name,
    join_segments,
    normalize_transcript,
    transcribe,
)


class TestNormalization:
    """Test safe transcript normalization."""

    def test_whitespace_collapse(self):
        """Multiple whitespace should collapse to single space."""
        assert normalize_transcript("hello   world") == "hello world"

    def test_newline_collapse(self):
        """Newlines should collapse to single space."""
        assert normalize_transcript("hello\n\nworld") == "hello world"

    def test_strip(self):
        """Leading/trailing whitespace should be stripped."""
        assert normalize_transcript("  hello  ") == "hello"

    def test_double_period(self):
        """Double periods should collapse to single."""
        assert normalize_transcript("hello.. world") == "hello. world"

    def test_unicode_whitespace(self):
        """Non-breaking spaces and other Unicode whitespace should normalize."""
        assert normalize_transcript("hello\u00a0world") == "hello world"

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_transcript("") == ""

    def test_medical_terms_preserved(self):
        """Medical terminology should NOT be changed."""
        text = "paracetamol 500 mg twice daily"
        assert normalize_transcript(text) == text

    def test_hindi_text_preserved(self):
        """Hindi text should pass through unchanged."""
        text = "मुझे बुखार है"
        assert normalize_transcript(text) == text


class TestJoinSegments:
    """Test segment joining."""

    def test_join_basic(self):
        """Segments should be joined with spaces."""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=1.0, text=" Hello"),
            TranscriptSegment(id=1, start=1.0, end=2.0, text=" world"),
        ]
        assert join_segments(segments) == "Hello world"

    def test_join_empty(self):
        """Empty segment list should return empty string."""
        assert join_segments([]) == ""

    def test_join_skips_empty_segments(self):
        """Segments with empty text should be skipped."""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=1.0, text=" Hello"),
            TranscriptSegment(id=1, start=1.0, end=2.0, text="  "),
            TranscriptSegment(id=2, start=2.0, end=3.0, text=" world"),
        ]
        assert join_segments(segments) == "Hello world"


class TestLanguageNames:
    """Test language name lookup."""

    def test_english(self):
        assert get_language_name("en") == "English"

    def test_hindi(self):
        assert get_language_name("hi") == "Hindi"

    def test_unknown(self):
        """Unknown codes should be capitalized."""
        assert get_language_name("xx") == "Xx"


class TestTranscribe:
    """Test the transcribe function with mocked model."""

    def test_returns_transcript_result(self, valid_wav_file: Path, mock_whisper_model):
        """transcribe should return a TranscriptResult."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = transcribe(valid_wav_file)

        assert isinstance(result, TranscriptResult)

    def test_text_returned(self, valid_wav_file: Path, mock_whisper_model):
        """Result should contain transcribed text."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = transcribe(valid_wav_file)

        assert result.text
        assert "test transcript" in result.text.lower()

    def test_language_returned(self, valid_wav_file: Path, mock_whisper_model):
        """Result should contain detected language."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = transcribe(valid_wav_file)

        assert result.language == "en"
        assert result.language_probability > 0

    def test_segments_returned(self, valid_wav_file: Path, mock_whisper_model):
        """Result should contain segments with timing info."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = transcribe(valid_wav_file)

        assert len(result.segments) > 0
        seg = result.segments[0]
        assert seg.start >= 0
        assert seg.end > seg.start
        assert seg.text

    def test_duration_returned(self, valid_wav_file: Path, mock_whisper_model):
        """Result should contain audio duration."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = transcribe(valid_wav_file)

        assert result.duration > 0

    def test_hindi_transcription(self, valid_wav_file: Path, mock_whisper_hindi):
        """Hindi transcription should return Hindi language code."""
        mock_model, _, _ = mock_whisper_hindi

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = transcribe(valid_wav_file)

        assert result.language == "hi"

    def test_model_size_override(self, valid_wav_file: Path, mock_whisper_model):
        """Model size override should be passed to config."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model) as mock_get:
            result = transcribe(valid_wav_file, model_size="small")

        # The config passed to _get_model should have the overridden size
        call_args = mock_get.call_args
        config = call_args[0][0] if call_args[0] else call_args[1].get("config")
        if config:
            assert config.model_size == "small"

    def test_model_cache_clear(self):
        """Model cache should be clearable."""
        # Just verify the function doesn't raise
        clear_model_cache()


class TestTranscriptSegment:
    """Test TranscriptSegment properties."""

    def test_duration_property(self):
        seg = TranscriptSegment(id=0, start=1.0, end=3.5, text="test")
        assert seg.duration == 2.5

    def test_optional_fields(self):
        seg = TranscriptSegment(id=0, start=0.0, end=1.0, text="test")
        assert seg.avg_logprob is None
        assert seg.no_speech_prob is None
