# Evaluation Dataset — TeleMed-Scribe STT
# ==========================================
#
# This directory contains the STT evaluation dataset for Person A's module.
#
# ## Structure
#
# ```
# evaluation_data/
# ├── README.md              ← this file
# ├── references/            ← reference transcripts (JSON)
# │   ├── english.json
# │   ├── hindi.json
# │   └── mixed.json
# └── audio/                 ← audio files (create via TTS or record)
#     ├── en_*.wav
#     ├── hi_*.wav
#     └── mix_*.wav
# ```
#
# ## Reference Transcript Format
#
# Each JSON file contains a list of evaluation samples:
#
# ```json
# [
#   {
#     "audio_file": "audio/en_consultation_01.wav",
#     "reference_transcript": "The patient reports fever for three days...",
#     "language": "english",
#     "speaker_context": "doctor",
#     "notes": "Standard consultation greeting"
#   }
# ]
# ```
#
# ## How to create audio samples
#
# Option 1: Record yourself reading the scripted consultations.
# Option 2: Use a TTS engine (e.g., gTTS) to generate audio:
#
#     python -m evaluation_data.generate_audio
#
# Option 3: Use the generate script in this directory.
#
# ## Important notes
#
# - Reference transcripts are the GROUND TRUTH and must be manually verified.
# - STT model output is NEVER used as the reference — it goes in hypothesis.
# - Keep audio samples short (30-60 seconds) for practical evaluation.
# - Include samples with medical terminology, dosages, and mixed language.
