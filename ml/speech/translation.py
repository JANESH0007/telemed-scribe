"""
ml/speech/translation.py — Translation Module (Phase 4-5)
===========================================================
Integrates NLLB-200 (distilled-600M) for non-English → English translation.

Language routing:
  - English → skip translation, return as-is
  - Hindi → translate to English
  - Other supported Indian languages → translate to English

Code-mixed speech (Phase 5):
  - Hindi-English mixed transcripts are passed through the Hindi→English
    translation pipeline. NLLB handles the mixed input reasonably well
    because it was trained on noisy, real-world text.
  - The original mixed transcript is always preserved.

Public API:
    translated = translate_to_english(text, source_language)
    result = translate_with_metadata(text, source_language)

Model initialization is separated from inference — loaded once and cached.
"""

from __future__ import annotations

import logging
from typing import Optional

from ml.speech.config import TranslationConfig, get_config
from ml.speech.schemas import TranslationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLLB language code mapping
# ---------------------------------------------------------------------------
# NLLB-200 uses BCP-47-like codes with script tags.
# Map Whisper's ISO 639-1 codes → NLLB's flores200 codes.

WHISPER_TO_NLLB = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "as": "asm_Beng",
    "or": "ory_Orya",
}

# Languages that should skip translation (already English)
SKIP_TRANSLATION = {"en", "eng_latn"}

# Target language for all translations
TARGET_LANG = "eng_Latn"


# ---------------------------------------------------------------------------
# Model singleton — loaded once, reused
# ---------------------------------------------------------------------------

_translator_cache: dict[str, dict] = {}


def _get_translator(config: Optional[TranslationConfig] = None) -> dict:
    """
    Load and cache the NLLB-200 model + tokenizer.
    Returns a dict with 'model' and 'tokenizer'.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = config or get_config().translation
    cache_key = config.model_name

    if cache_key not in _translator_cache:
        logger.info("Loading translation model: %s", config.model_name)

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

        # Move to device
        if config.device != "cpu":
            import torch
            if torch.cuda.is_available():
                model = model.to(config.device)

        model.eval()

        _translator_cache[cache_key] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
        }
        logger.info("Translation model loaded successfully.")

    return _translator_cache[cache_key]


def clear_translation_cache() -> None:
    """Release cached translation models."""
    _translator_cache.clear()
    logger.info("Translation model cache cleared.")


# ---------------------------------------------------------------------------
# Translation core
# ---------------------------------------------------------------------------

def _needs_translation(source_language: str) -> bool:
    """Check if the source language requires translation."""
    return source_language.lower() not in SKIP_TRANSLATION


def _resolve_nllb_code(whisper_lang_code: str) -> str:
    """
    Convert a Whisper language code to NLLB's flores200 code.
    Falls back to Hindi if the code is unknown (safe default for
    Indian language detection errors).
    """
    # Already an NLLB code
    if "_" in whisper_lang_code and len(whisper_lang_code) == 8:
        return whisper_lang_code

    code = whisper_lang_code.lower()

    nllb_code = WHISPER_TO_NLLB.get(code)
    if nllb_code is None:
        logger.warning(
            "Unknown language code '%s', falling back to Hindi (hin_Deva)", code
        )
        return "hin_Deva"

    return nllb_code


def translate_to_english(
    text: str,
    source_language: str,
    config: Optional[TranslationConfig] = None,
) -> str:
    """
    Translate text to English using NLLB-200.

    Args:
        text: Input text to translate.
        source_language: Source language code (Whisper-style: 'hi', 'bn', etc.)
        config: Override TranslationConfig.

    Returns:
        Translated English text, or the original text if source is English.

    This function never overwrites the original transcript — callers should
    store both the original and translated versions.
    """
    if not text or not text.strip():
        return text

    if not _needs_translation(source_language):
        logger.info("Source is English — skipping translation.")
        return text

    config = config or get_config().translation
    translator = _get_translator(config)

    model = translator["model"]
    tokenizer = translator["tokenizer"]

    src_code = _resolve_nllb_code(source_language)
    logger.info("Translating %s → %s", src_code, TARGET_LANG)

    # Set source language for tokenizer
    tokenizer.src_lang = src_code

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length,
    )

    # Move to device if needed
    if config.device != "cpu":
        import torch
        if torch.cuda.is_available():
            inputs = {k: v.to(config.device) for k, v in inputs.items()}

    # Generate translation
    import torch
    target_lang_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_new_tokens=config.max_length,
            num_beams=4,
            early_stopping=True,
        )

    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    logger.info("Translation complete: %d chars → %d chars", len(text), len(translated))
    return translated.strip()


def translate_with_metadata(
    text: str,
    source_language: str,
    config: Optional[TranslationConfig] = None,
) -> TranslationResult:
    """
    Translate text and return a full TranslationResult with metadata.
    Always preserves the original text.
    """
    was_translated = _needs_translation(source_language)

    if was_translated:
        translated_text = translate_to_english(text, source_language, config)
    else:
        translated_text = text

    return TranslationResult(
        original_text=text,
        translated_text=translated_text,
        source_language=source_language,
        was_translated=was_translated,
    )


# ---------------------------------------------------------------------------
# Code-mixed speech support (Phase 5)
# ---------------------------------------------------------------------------

def translate_code_mixed(
    text: str,
    detected_language: str = "hi",
    config: Optional[TranslationConfig] = None,
) -> TranslationResult:
    """
    Handle Hindi-English code-mixed speech.

    Strategy:
      - Pass the entire mixed transcript through NLLB's Hindi→English pipeline.
      - NLLB handles English words in Hindi text reasonably well because it was
        trained on diverse multilingual data.
      - The original mixed transcript is always preserved.

    Known limitations (documented):
      - Medical terminology in English within Hindi speech may get
        mistranslated (e.g., "paracetamol" might be transliterated)
      - Dosage numbers are generally preserved well
      - Very short code-mixed segments may confuse the model

    Args:
        text: Code-mixed transcript (e.g., "Mujhe three days se fever hai")
        detected_language: The language Whisper detected (usually "hi" for mixed)
        config: Override TranslationConfig.

    Returns:
        TranslationResult with original and translated text.
    """
    logger.info("Processing code-mixed speech (detected lang: %s)", detected_language)

    # For code-mixed, always translate through the Hindi pipeline
    # even if some words are English — NLLB handles this
    return translate_with_metadata(text, detected_language, config)
