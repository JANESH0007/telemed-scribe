import sys
from ml.speech import process_consultation_audio

def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_audio>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    print(f"\n🎧 Processing Audio: {audio_path}")
    print("⏳ Running pipeline (Audio Preprocessing -> STT -> Translation)...")
    print("-" * 50)
    
    try:
        result = process_consultation_audio(audio_path)
        
        print("\n✅ PIPELINE COMPLETE!")
        print(f"🌍 Detected Language: {result.language} (Confidence: {result.language_probability:.2f})")
        print(f"⏱️  Duration: {result.duration}s")
        
        print("\n📝 --- RAW TRANSCRIPT (Original Language) ---")
        print(result.original_text)
        
        print("\n🎯 --- FINAL ENGLISH TRANSCRIPT (For Person B) ---")
        print(result.translated_text)
        
        print("\n" + "="*50)
        print("Success! The pipeline correctly converts raw audio to clean English text.")
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")

if __name__ == "__main__":
    main()
