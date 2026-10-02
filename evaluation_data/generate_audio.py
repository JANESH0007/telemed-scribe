"""
evaluation_data/generate_audio.py — Generate evaluation audio using TTS
=========================================================================
Creates audio files from the reference transcripts using gTTS (Google TTS).

Usage:
    python -m evaluation_data.generate_audio

This generates scripted consultation audio for evaluation. For real-world
testing, record actual speech instead.

Note: gTTS requires internet access. Install with: pip install gTTS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
AUDIO_DIR = EVAL_DIR / "audio"
REFS_DIR = EVAL_DIR / "references"


def generate_audio_samples():
    """Generate audio files from all reference transcript files."""
    try:
        from gtts import gTTS
    except ImportError:
        print("ERROR: gTTS not installed. Run: pip install gTTS")
        sys.exit(1)

    AUDIO_DIR.mkdir(exist_ok=True)

    # Language mapping for gTTS
    lang_map = {
        "english": "en",
        "hindi": "hi",
        "mixed": "hi",  # Use Hindi TTS for code-mixed (best effort)
    }

    total_generated = 0

    for ref_file in sorted(REFS_DIR.glob("*.json")):
        print(f"\nProcessing: {ref_file.name}")

        with open(ref_file) as f:
            samples = json.load(f)

        for sample in samples:
            audio_file = EVAL_DIR / sample["audio_file"]
            language = sample["language"]
            text = sample["reference_transcript"]

            if audio_file.exists():
                print(f"  SKIP (exists): {audio_file.name}")
                continue

            audio_file.parent.mkdir(parents=True, exist_ok=True)

            tts_lang = lang_map.get(language, "en")
            print(f"  Generating: {audio_file.name} (lang={tts_lang})")

            try:
                tts = gTTS(text=text, lang=tts_lang, slow=False)

                # gTTS outputs MP3, so save as MP3 first then convert
                mp3_path = audio_file.with_suffix(".mp3")
                tts.save(str(mp3_path))

                # Convert MP3 → WAV (16kHz mono) using pydub
                from pydub import AudioSegment

                audio = AudioSegment.from_mp3(str(mp3_path))
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(str(audio_file), format="wav")

                # Remove temporary MP3
                mp3_path.unlink()

                total_generated += 1
                print(f"    ✓ Created: {audio_file.name} ({audio.duration_seconds:.1f}s)")

            except Exception as e:
                print(f"    ✗ Failed: {e}")

    print(f"\nDone! Generated {total_generated} audio files in {AUDIO_DIR}")


if __name__ == "__main__":
    generate_audio_samples()
