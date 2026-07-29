"""
whisper_ollama_translate.py — Whisper (STT) + your FastAPI/Ollama backend (translation).

Two separate, well-established tools instead of Gemma 4's native audio input:
  1. Whisper transcribes the audio to text (runs on CPU by default here, so it
     doesn't compete with Ollama for your 6GB of VRAM).
  2. The transcript is sent to your existing FastAPI /translate endpoint
     (backend/main.py), which is backed by Ollama running gemma4:e4b.

This sidesteps every GPU/audio-merging bug we hit with Gemma 4's native audio
input — Whisper is a dedicated STT model, not a multimodal LLM, so there's no
audio-embedding-into-language-model step to go wrong.

Requirements:
  pip install openai-whisper requests
  ffmpeg on PATH (same requirement as make_wav.py had)

Before running, make sure these are both up:
  Terminal 1: ollama serve                         (if not already running)
  Terminal 2: cd backend && uvicorn main:app --reload --port 8000

Usage:
  python whisper_ollama_translate.py doctor_note.mp3
  python whisper_ollama_translate.py patient_reply.wav --whisper-model small
  python whisper_ollama_translate.py note.m4a --device cuda   # faster, but shares GPU with Ollama
"""

import argparse
import sys
from pathlib import Path

import requests
import whisper


def transcribe(audio_path: str, model_size: str, device: str):
    print(f"Loading Whisper '{model_size}' model on {device}...")
    model = whisper.load_model(model_size, device=device)

    print(f"Transcribing {audio_path}...")
    # Whisper decodes/resamples the audio internally via ffmpeg — any common
    # format works directly, no separate conversion step needed.
    result = model.transcribe(audio_path)

    return result["text"].strip(), result.get("language", "unknown")


def translate(api_base: str, text: str) -> dict:
    url = f"{api_base}/translate"
    response = requests.post(url, json={"text": text}, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Whisper STT + your FastAPI/Ollama translation backend")
    parser.add_argument("audio_path", help="Path to any audio file (mp3, wav, m4a, webm, ...)")
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        help="Whisper model size (default: small — good balance of speed/accuracy for Hindi+English)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper (default: cpu, so it doesn't fight Ollama for your 6GB of VRAM)",
    )
    parser.add_argument("--api-base", default="http://localhost:8000", help="Your FastAPI backend URL")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        sys.exit(f"File not found: {audio_path}")

    transcript, detected_lang = transcribe(str(audio_path), args.whisper_model, args.device)
    if not transcript:
        sys.exit("Whisper returned an empty transcript — check the audio file actually has clear speech in it.")

    print(f"\nTranscript ({detected_lang}): {transcript}")

    print("Sending to /translate...")
    try:
        result = translate(args.api_base, transcript)
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Could not reach {args.api_base}/translate.\n"
            f"Is your FastAPI backend running? Start it with:\n"
            f"  cd backend && uvicorn main:app --reload --port 8000\n"
            f"And make sure Ollama is running with gemma4:e4b pulled (ollama serve)."
        )
    except requests.exceptions.HTTPError as e:
        body = e.response.text if e.response is not None else ""
        sys.exit(f"Backend returned an error: {e}\n{body}")

    print("\n--- Result ---")
    print(f"Original  ({result['source_lang']}): {result['original_text']}")
    print(f"Translated ({result['target_lang']}): {result['translated_text']}")


if __name__ == "__main__":
    main()