# TeleMed-Scribe — Speech, Audio & Multilingual Processing (Person A)

Speech-to-Text and Multilingual translation pipeline for telemedicine consultations in India (supporting English, Hindi, and Code-mixed clinical dialogue).

## Architecture Overview

```text
Audio (.wav / .mp3 / .m4a)
       │
       ▼
 Audio Preprocessing (pydub / standard 16kHz mono / silence-aware chunking)
       │
       ▼
 faster-whisper (int8 quantization on CPU)
       │
       ▼
 Language Detection
  ├── English ─────────────► Clean English Transcript
  └── Hindi / Code-Mixed ──► NLLB-200 (facebook/nllb-200-distilled-600M) ──► Clean English Transcript
```

## Downstream Interface (Person B & Person D)

```python
from ml.speech import process_consultation_audio

result = process_consultation_audio("consultation.mp3")

# For Person B (T5 Micro-LM Medical Extractor):
english_text = result.translated_text

# For Person D (FastAPI / Streamlit):
response_json = result.to_dict()
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Run Pipeline Demo
```bash
python demo.py path/to/audio.wav
```

### 4. Run Evaluation (WER & Medical Term Accuracy)
```bash
python -m evaluation_data.run_evaluation
```
