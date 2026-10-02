"""
tests/test_translation.py — Translation Tests (Phase 11)
==========================================================
Tests for:
  - English input skips translation
  - Hindi input produces English output
  - Original text is always preserved
  - Code-mixed speech handling
  - Language code mapping

Uses mocks to avoid loading the real NLLB model in unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.speech.schemas import TranslationResult
from ml.speech.translation import (
    WHISPER_TO_NLLB,
    _needs_translation,
    _resolve_nllb_code,
    clear_translation_cache,
    translate_code_mixed,
    translate_to_english,
    translate_with_metadata,
)


class TestNeedsTranslation:
    """Test the translation routing logic."""

    def test_english_skips(self):
        """English should NOT be translated."""
        assert _needs_translation("en") is False

    def test_hindi_needs_translation(self):
        """Hindi should be translated."""
        assert _needs_translation("hi") is True

    def test_bengali_needs_translation(self):
        """Bengali should be translated."""
        assert _needs_translation("bn") is True

    def test_nllb_english_code_skips(self):
        """NLLB's eng_Latn code should also skip."""
        assert _needs_translation("eng_Latn") is False


class TestLanguageCodeMapping:
    """Test Whisper → NLLB code mapping."""

    def test_hindi_mapping(self):
        assert _resolve_nllb_code("hi") == "hin_Deva"

    def test_english_mapping(self):
        assert _resolve_nllb_code("en") == "eng_Latn"

    def test_bengali_mapping(self):
        assert _resolve_nllb_code("bn") == "ben_Beng"

    def test_unknown_falls_back_to_hindi(self):
        """Unknown codes should fall back to Hindi (safe default)."""
        assert _resolve_nllb_code("unknown") == "hin_Deva"

    def test_already_nllb_code(self):
        """NLLB codes should pass through."""
        assert _resolve_nllb_code("hin_Deva") == "hin_Deva"


class TestTranslateToEnglish:
    """Test the main translation function with mocked model."""

    def test_english_input_skips(self):
        """English text should be returned as-is without loading any model."""
        result = translate_to_english("Hello doctor", "en")
        assert result == "Hello doctor"

    def test_empty_text_returns_empty(self):
        """Empty text should return empty."""
        result = translate_to_english("", "hi")
        assert result == ""

    def test_whitespace_only_returns_as_is(self):
        """Whitespace-only text should return as-is."""
        result = translate_to_english("   ", "hi")
        assert result == "   "

    def test_hindi_input_calls_model(self, mock_translation_model):
        """Hindi input should invoke the translation model."""
        mock_model, mock_tokenizer = mock_translation_model

        with patch("ml.speech.translation._get_translator") as mock_get:
            mock_get.return_value = {
                "model": mock_model,
                "tokenizer": mock_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            result = translate_to_english("मुझे बुखार है", "hi")

        assert result == "I have had fever for three days."
        mock_model.generate.assert_called_once()

    def test_hindi_sets_source_language(self, mock_translation_model):
        """Translation should set tokenizer source language."""
        mock_model, mock_tokenizer = mock_translation_model

        with patch("ml.speech.translation._get_translator") as mock_get:
            mock_get.return_value = {
                "model": mock_model,
                "tokenizer": mock_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            translate_to_english("test", "hi")

        assert mock_tokenizer.src_lang == "hin_Deva"


class TestTranslateWithMetadata:
    """Test translation with metadata preservation."""

    def test_english_preserves_original(self):
        """English translation should preserve original text."""
        result = translate_with_metadata("Hello doctor", "en")

        assert isinstance(result, TranslationResult)
        assert result.original_text == "Hello doctor"
        assert result.translated_text == "Hello doctor"
        assert result.was_translated is False
        assert result.source_language == "en"

    def test_hindi_preserves_original(self, mock_translation_model):
        """Hindi translation should preserve original text."""
        mock_model, mock_tokenizer = mock_translation_model

        with patch("ml.speech.translation._get_translator") as mock_get:
            mock_get.return_value = {
                "model": mock_model,
                "tokenizer": mock_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            result = translate_with_metadata("मुझे बुखार है", "hi")

        assert result.original_text == "मुझे बुखार है"
        assert result.translated_text == "I have had fever for three days."
        assert result.was_translated is True


class TestCodeMixed:
    """Test code-mixed speech translation."""

    def test_code_mixed_calls_translation(self, mock_translation_model):
        """Code-mixed text should go through translation pipeline."""
        mock_model, mock_tokenizer = mock_translation_model

        with patch("ml.speech.translation._get_translator") as mock_get:
            mock_get.return_value = {
                "model": mock_model,
                "tokenizer": mock_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            result = translate_code_mixed(
                "Mujhe three days se fever hai",
                detected_language="hi",
            )

        assert isinstance(result, TranslationResult)
        assert result.was_translated is True
        assert result.original_text == "Mujhe three days se fever hai"

    def test_code_mixed_preserves_original(self, mock_translation_model):
        """Code-mixed translation should always preserve original."""
        mock_model, mock_tokenizer = mock_translation_model

        with patch("ml.speech.translation._get_translator") as mock_get:
            mock_get.return_value = {
                "model": mock_model,
                "tokenizer": mock_tokenizer,
                "config": MagicMock(device="cpu", max_length=512),
            }

            result = translate_code_mixed(
                "Doctor sahab, mera blood pressure high hai",
                detected_language="hi",
            )

        assert result.original_text == "Doctor sahab, mera blood pressure high hai"


class TestCacheManagement:
    """Test translation model cache."""

    def test_clear_cache(self):
        """Cache clearing should not raise."""
        clear_translation_cache()
