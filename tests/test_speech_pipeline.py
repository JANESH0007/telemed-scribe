"""
tests/test_speech_pipeline.py — Full Pipeline Tests (Phase 11)
================================================================
Tests the end-to-end pipeline:
    audio → STT → language detection → translation → final result

Uses mocks for both Whisper and NLLB models to keep tests fast.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.speech.pipeline import (
    PipelineError,
    get_consultation_dict,
    get_english_transcript,
    process_consultation_audio,
)
from ml.speech.schemas import ConsultationResult


class TestPipelineEnglish:
    """Test pipeline with English audio (no translation needed)."""

    def test_english_returns_consultation_result(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """Pipeline should return a ConsultationResult."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = process_consultation_audio(valid_wav_file)

        assert isinstance(result, ConsultationResult)

    def test_english_no_translation(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """English audio should skip translation."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = process_consultation_audio(valid_wav_file)

        assert result.was_translated is False
        assert result.language == "English"
        assert result.language_code == "en"
        # For English, original and translated should be the same
        assert result.translated_text == result.original_text

    def test_english_text_available(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """Result should have text for Micro-LM consumption."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = process_consultation_audio(valid_wav_file)

        assert result.translated_text
        assert result.for_micro_lm()  # Convenience method

    def test_english_segments_present(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """Result should contain segments."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = process_consultation_audio(valid_wav_file)

        assert len(result.segments) > 0

    def test_english_duration_present(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """Result should contain duration."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = process_consultation_audio(valid_wav_file)

        assert result.duration > 0


class TestPipelineHindi:
    """Test pipeline with Hindi audio (translation required)."""

    def test_hindi_triggers_translation(
        self, valid_wav_file: Path, mock_whisper_hindi, mock_translation_model
    ):
        """Hindi audio should trigger translation."""
        whisper_model, _, _ = mock_whisper_hindi
        nllb_model, nllb_tokenizer = mock_translation_model

        with (
            patch("ml.speech.stt._get_model", return_value=whisper_model),
            patch("ml.speech.translation._get_translator") as mock_get_trans,
        ):
            mock_get_trans.return_value = {
                "model": nllb_model,
                "tokenizer": nllb_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            result = process_consultation_audio(valid_wav_file)

        assert result.was_translated is True
        assert result.language == "Hindi"
        assert result.language_code == "hi"

    def test_hindi_preserves_original(
        self, valid_wav_file: Path, mock_whisper_hindi, mock_translation_model
    ):
        """Hindi pipeline should preserve the original transcript."""
        whisper_model, _, _ = mock_whisper_hindi
        nllb_model, nllb_tokenizer = mock_translation_model

        with (
            patch("ml.speech.stt._get_model", return_value=whisper_model),
            patch("ml.speech.translation._get_translator") as mock_get_trans,
        ):
            mock_get_trans.return_value = {
                "model": nllb_model,
                "tokenizer": nllb_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            result = process_consultation_audio(valid_wav_file)

        # Original Hindi text should be preserved
        assert "बुखार" in result.original_text
        # Translated text should be English
        assert result.translated_text != result.original_text

    def test_hindi_translated_text_for_microlm(
        self, valid_wav_file: Path, mock_whisper_hindi, mock_translation_model
    ):
        """translated_text should be consumable by Person B's Micro-LM."""
        whisper_model, _, _ = mock_whisper_hindi
        nllb_model, nllb_tokenizer = mock_translation_model

        with (
            patch("ml.speech.stt._get_model", return_value=whisper_model),
            patch("ml.speech.translation._get_translator") as mock_get_trans,
        ):
            mock_get_trans.return_value = {
                "model": nllb_model,
                "tokenizer": nllb_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            result = process_consultation_audio(valid_wav_file)

        micro_lm_input = result.translated_text
        assert isinstance(micro_lm_input, str)
        assert len(micro_lm_input) > 0


class TestPipelineOptions:
    """Test pipeline configuration options."""

    def test_skip_translation_flag(
        self, valid_wav_file: Path, mock_whisper_hindi
    ):
        """skip_translation=True should skip translation even for Hindi."""
        whisper_model, _, _ = mock_whisper_hindi

        with patch("ml.speech.stt._get_model", return_value=whisper_model):
            result = process_consultation_audio(
                valid_wav_file, skip_translation=True
            )

        assert result.was_translated is False
        assert result.language_code == "hi"
        # Without translation, translated_text == original_text
        assert result.translated_text == result.original_text

    def test_model_size_override(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """model_size parameter should be passed through."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = process_consultation_audio(
                valid_wav_file, model_size="small"
            )

        assert isinstance(result, ConsultationResult)


class TestPipelineOutput:
    """Test pipeline output formats."""

    def test_to_dict(self, valid_wav_file: Path, mock_whisper_model):
        """to_dict should return a plain dictionary."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            result = process_consultation_audio(valid_wav_file)

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "original_text" in d
        assert "translated_text" in d
        assert "language" in d
        assert "segments" in d

    def test_convenience_dict_function(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """get_consultation_dict should return a dict directly."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            d = get_consultation_dict(valid_wav_file)

        assert isinstance(d, dict)
        assert "translated_text" in d

    def test_convenience_text_function(
        self, valid_wav_file: Path, mock_whisper_model
    ):
        """get_english_transcript should return a string directly."""
        mock_model, _, _ = mock_whisper_model

        with patch("ml.speech.stt._get_model", return_value=mock_model):
            text = get_english_transcript(valid_wav_file)

        assert isinstance(text, str)
        assert len(text) > 0


class TestPipelineErrors:
    """Test pipeline error handling."""

    def test_invalid_audio_raises(self, temp_dir: Path):
        """Pipeline should raise for invalid audio files."""
        from ml.speech.audio import AudioValidationError

        with pytest.raises(AudioValidationError):
            process_consultation_audio(temp_dir / "nonexistent.wav")

    def test_corrupted_audio_raises(self, corrupted_file: Path):
        """Pipeline should raise for corrupted audio."""
        from ml.speech.audio import AudioValidationError

        with pytest.raises(AudioValidationError):
            process_consultation_audio(corrupted_file)
