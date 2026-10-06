"""
Heavy ML dependencies behind one object so the API can be tested without models.

Defaults load lazily on first use:
  transcriber -> ml.speech.process_consultation_audio  (Person A)
  extractor   -> ml.extraction.MedicalExtractor        (Person B; weights from LORA_WEIGHTS_PATH, default ./lora-medical-t5)
  med_rag / hist_rag                                   (Person C)
"""
from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional


class BadInput(Exception):
    """Client-side problem (bad/unsupported audio, empty transcript) -> HTTP 422."""


class Services:
    def __init__(self, transcriber: Optional[Callable] = None, extractor: Optional[Callable] = None,
                 med_rag=None, hist_rag=None):
        self._transcriber, self._extractor = transcriber, extractor
        if med_rag is not None:
            self.med_rag = med_rag
        if hist_rag is not None:
            self.hist_rag = hist_rag

    # transcribe(path: str, language: str | None) -> dict   (ConsultationResult.to_dict())
    def transcribe(self, path: str, language: Optional[str] = None) -> dict:
        if self._transcriber:
            return self._transcriber(path, language)
        from ml.speech import process_consultation_audio
        from ml.speech.audio import AudioProcessingError
        try:
            return process_consultation_audio(path, force_language=language).to_dict()
        except AudioProcessingError as e:
            raise BadInput(str(e)) from e

    # extract(text: str) -> dict with symptoms/diagnosis/prescriptions/follow_up/raw_output
    def extract(self, text: str) -> dict:
        if not text.strip():
            raise BadInput("Transcript is empty; nothing to extract.")
        if self._extractor:
            return self._extractor(text)
        return self._default_extractor.extract(text)

    @cached_property
    def _default_extractor(self):
        from ml.extraction.extractor import MedicalExtractor
        weights = os.getenv("LORA_WEIGHTS_PATH", "./lora-medical-t5")
        return MedicalExtractor(lora_weights_path=weights if Path(weights).exists() else None)

    @cached_property
    def med_rag(self):
        from ml.rag import MedicationRAG
        return MedicationRAG.build()

    @cached_property
    def hist_rag(self):
        from ml.rag import PatientHistoryRAG
        return PatientHistoryRAG()
