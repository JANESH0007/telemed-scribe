"""
TeleMed-Scribe — Speech, Audio & Multilingual Processing Module
================================================================
Person A's module: audio preprocessing, STT (faster-whisper),
translation (NLLB-200), and evaluation.

Public API:
    from ml.speech import process_consultation_audio
    from ml.speech import process_audio, transcribe, translate_to_english
"""

from ml.speech.pipeline import process_consultation_audio
from ml.speech.audio import process_audio
from ml.speech.stt import transcribe
from ml.speech.translation import translate_to_english

__all__ = [
    "process_consultation_audio",
    "process_audio",
    "transcribe",
    "translate_to_english",
]
