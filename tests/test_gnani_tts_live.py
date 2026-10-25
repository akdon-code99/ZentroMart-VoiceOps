
"""One-request, manually confirmed Gnani TTS smoke test.

Run from the project root. This makes one real API request only after
the user types YES. It does not automatically retry or save audio.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv

from voice_adapter import GnaniTTS, LocalAudioInterface


def main() -> None:
    load_dotenv()

    text = "Hello, your order status is being checked."

    print("Gnani TTS live smoke test")
    print(f"Text to synthesize: {text}")
    print("Language: en (Indian English); voice/model come from voice_adapter.py")
    print("This will make one real TTS API request and play the returned audio.")

    if input('Type YES to continue, or anything else to cancel: ').strip() != "YES":
        print("Cancelled. No API request was made.")
        return

    tts = GnaniTTS()
    audio = list(tts.stream_speech(text, language="en"))

    if not audio:
        raise RuntimeError("Gnani returned no audio chunks.")

    print(f"Received {len(audio)} audio chunk(s). Playing now...")
    LocalAudioInterface().play_wav(audio)
    print("TTS smoke test completed.")


if __name__ == "__main__":
    main()
