"""
Phase 1 — Doctor/Patient Voice Translator backend.

Single responsibility: POST /translate
- Takes raw utterance text (already transcribed by the browser's Web Speech API)
- Sends it to Gemma with a fixed system prompt
- Validates the model's JSON reply against a strict Pydantic schema
- Retries once on malformed output, then returns 502

No DB, no auth, no persistence — fully stateless.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import numpy
import httpx
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, field_validator
load_dotenv()
import re


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Which backend actually hosts the Gemma model. Two options are wired up:
#   "ollama" -> local Ollama server running a gemma model (default, easiest for a demo)
#   "google" -> Google AI Studio (generativelanguage.googleapis.com) hosted Gemma
GEMMA_PROVIDER = os.getenv("GEMMA_PROVIDER", "ollama")

# Ollama (local) config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")

# Google AI Studio config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemma-3-27b-it")

# Whisper (STT) config
# Loaded ONCE, at process startup, and kept resident in memory for the life of
# the server — this is what makes it "stay on" instead of reloading per file
# the way the old whisper_ollama_translate.py CLI script did.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # "cpu" by default so it doesn't fight Ollama for VRAM

# Dev origins for CORS — covers both Vite and CRA default ports.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


SYSTEM_PROMPT = """You are a real-time medical interpreter translating utterances spoken in a doctor-patient conversation. Each input is a single utterance in either Hindi or English.

Do the following, in order:
1. Detect whether the input text is Hindi ("hi") or English ("en"). If mixed, choose the dominant language.
2. Translate it into the OTHER language (English -> Hindi, or Hindi -> English).
3. Preserve the medical meaning precisely. Do not add, omit, guess, or interpret anything not present in the source text. Keep the tone neutral, respectful, and clinically appropriate.
4. Output ONLY one JSON object and nothing else. No markdown, no code fences, no commentary before or after.

SCRIPT REQUIREMENT (critical):
- Hindi text — both detection and translation output — MUST be written in Devanagari script ONLY (e.g. क, ख, ग, अ, आ ... मुझे कुछ परहेज़ करना होगा?).
- NEVER use Urdu / Nastaliq / Perso-Arabic script (e.g. ی, ک, ھ, گ) under any circumstances, even if the input is ambiguous, colloquial, or contains Urdu-origin loanwords. Loanwords common to Hindi and Urdu (e.g. "parhez", "dawa", "tabiyat") must still be rendered in Devanagari: परहेज़, दवा, तबियत.
- If you detect the input script is Perso-Arabic (Urdu) rather than Devanagari (Hindi), still treat "hi" as meaning Devanagari Hindi for output purposes — transliterate the meaning into Devanagari, never pass through or produce Nastaliq script.
- English output must use standard Latin script only.

The JSON object must exactly match this schema:
{
  "source_lang": "hi" or "en",
  "target_lang": "hi" or "en",
  "original_text": "<the exact original input text, unmodified>",
  "translated_text": "<the translated text>"
}

Rules:
- source_lang and target_lang must each be exactly "hi" or "en".
- target_lang must always be the opposite of source_lang.
- original_text must be an exact, unmodified copy of the input text.
- translated_text must contain only the translation, no notes or explanations.
- translated_text must never mix scripts — Hindi output is 100% Devanagari, English output is 100% Latin, with the sole exception of standard medical/numeric notation (e.g. "10mg", "2 बार/दिन").
- Never wrap the JSON in markdown code fences.
- Never output anything other than the single JSON object.

Examples:

Input: "Do I have to restrict anything in my diet?"
Output: {"source_lang": "en", "target_lang": "hi", "original_text": "Do I have to restrict anything in my diet?", "translated_text": "क्या मुझे अपने खाने में कुछ परहेज़ करना होगा?"}

Input: "मुझे पेट में दर्द हो रहा है"
Output: {"source_lang": "hi", "target_lang": "en", "original_text": "मुझे पेट में दर्द हो रहा है", "translated_text": "I am having pain in my stomach"}
"""

app = FastAPI(title="Doctor-Patient Voice Translator — Phase 1")

print(f"[startup] Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} — this stays loaded for as long as the server runs...")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
print("[startup] Whisper model loaded and resident in memory.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str


class TranslationResult(BaseModel):
    source_lang: Literal["hi", "en"]
    target_lang: Literal["hi", "en"]
    original_text: str
    translated_text: str

    @field_validator("target_lang")
    @classmethod
    def target_must_be_opposite(cls, v, info):
        source = info.data.get("source_lang")
        if source is not None and v == source:
            raise ValueError("target_lang must differ from source_lang")
        return v


class ConversationTurnIn(BaseModel):
    speaker: Literal["doctor", "patient"]
    text: str  # English-normalized text for this turn


class SummarizeRequest(BaseModel):
    conversation: list[ConversationTurnIn]


class PatientContext(BaseModel):
    known_conditions: list[str] = []
    current_medications: list[str] = []
    known_allergies: list[str] = []


class ConversationSummaryResult(BaseModel):
    symptoms: list[str] = []
    duration: str = ""
    urgency: Literal["LOW", "MEDIUM", "HIGH"]
    suggested_department: str
    summary: str
    patient_context: PatientContext


SUMMARY_SYSTEM_PROMPT = """You are a clinical documentation assistant. You will receive a transcript of a doctor-patient conversation, already in English, with each line labeled "Doctor:" or "Patient:". Extract only what was explicitly said — never invent, assume, or infer anything beyond the transcript.

Extract:
- symptoms: symptoms explicitly mentioned by the patient. Empty list if none mentioned.
- duration: how long symptoms have lasted, stated as in the transcript (e.g. "3 days"). Empty string if not mentioned.
- urgency: your triage-style assessment based ONLY on what's in the transcript — exactly "LOW", "MEDIUM", or "HIGH". This is a suggestion to help prioritize, not a diagnosis.
- suggested_department: the hospital department best suited to this case (e.g. "Emergency Medicine", "General Medicine", "Cardiology", "Pediatrics"), based on the symptoms mentioned.
- summary: a neutral, 1-3 sentence clinical summary of the conversation.
- patient_context: known_conditions, current_medications, known_allergies — each a list of items explicitly stated by the patient. Empty list if not mentioned. Do not infer a condition just because a related symptom was mentioned.

Output ONLY one JSON object matching this schema, nothing else — no markdown, no commentary:
{
  "symptoms": ["..."],
  "duration": "...",
  "urgency": "LOW" | "MEDIUM" | "HIGH",
  "suggested_department": "...",
  "summary": "...",
  "patient_context": {
    "known_conditions": ["..."],
    "current_medications": ["..."],
    "known_allergies": ["..."]
  }
}

Rules:
- Never fabricate information not present in the transcript.
- If a field isn't mentioned, use an empty list or empty string — never guess.
- urgency must be exactly one of "LOW", "MEDIUM", "HIGH".
- Never wrap the JSON in markdown code fences.
"""


# ---------------------------------------------------------------------------
# Gemma call (provider-agnostic wrapper, configurable system prompt)
# ---------------------------------------------------------------------------
async def _call_ollama(system_prompt: str, user_text: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "format": "json",  # ask Ollama to constrain output to valid JSON
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


async def _call_google(system_prompt: str, user_text: str) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GOOGLE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_gemma(system_prompt: str, user_text: str) -> str:
    if GEMMA_PROVIDER == "google":
        return await _call_google(system_prompt, user_text)
    return await _call_ollama(system_prompt, user_text)


def _strip_code_fences(raw: str) -> str:
    """Gemma sometimes wraps JSON in ```json ... ``` even when told not to. Strip it defensively."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


async def _get_validated_translation(user_text: str) -> TranslationResult:
    raw = await call_gemma(SYSTEM_PROMPT, user_text)
    cleaned = _strip_code_fences(raw)
    parsed = json.loads(cleaned)  # may raise json.JSONDecodeError
    return TranslationResult(**parsed)  # may raise ValidationError


async def _get_validated_summary(transcript_text: str) -> ConversationSummaryResult:
    raw = await call_gemma(SUMMARY_SYSTEM_PROMPT, transcript_text)
    cleaned = _strip_code_fences(raw)
    parsed = json.loads(cleaned)
    return ConversationSummaryResult(**parsed)


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------
async def _translate_with_retry(text: str) -> TranslationResult:
    last_error = None
    for _attempt in range(2):  # initial try + exactly one retry
        try:
            return await _get_validated_translation(text)
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            last_error = f"Model returned invalid JSON: {e}"
        except httpx.HTTPError as e:
            last_error = f"Gemma request failed: {e}"
        except RuntimeError as e:
            last_error = str(e)

    raise HTTPException(status_code=502, detail=last_error or "Translation failed")


async def summarize_conversation(turns: list[ConversationTurnIn]) -> ConversationSummaryResult:
    if not turns:
        raise HTTPException(status_code=400, detail="conversation must not be empty")

    transcript_text = "\n".join(f"{t.speaker.capitalize()}: {t.text}" for t in turns)

    last_error = None
    for _attempt in range(2):  # initial try + exactly one retry
        try:
            return await _get_validated_summary(transcript_text)
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            last_error = f"Model returned invalid JSON: {e}"
        except httpx.HTTPError as e:
            last_error = f"Gemma request failed: {e}"
        except RuntimeError as e:
            last_error = str(e)

    raise HTTPException(status_code=502, detail=last_error or "Summary generation failed")


def _transcribe_audio_file(tmp_path: str) -> str:
    audio = whisper.load_audio(tmp_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)

    # Clamp to hi/en only — ignore Urdu and any other language Whisper might guess
    lang = "hi" if probs.get("hi", 0.0) >= probs.get("en", 0.0) else "en"

    result = whisper_model.transcribe(
        tmp_path,
        task="transcribe",
        language=lang,
        beam_size=1,
        temperature=0.0,
        fp16=True,
        condition_on_previous_text=False,
        without_timestamps=True,
        initial_prompt="Doctor-patient conversation. English and हिंदी."
    )
    return result["text"].strip()


async def _save_upload_to_tempfile(audio: UploadFile) -> str:
    suffix = Path(audio.filename or "").suffix or ".webm"
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio was empty")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/translate", response_model=TranslationResult)
async def translate(req: TranslateRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return await _translate_with_retry(req.text)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Whisper-only: send an audio clip back its transcript. Useful for debugging STT in isolation."""
    tmp_path = await _save_upload_to_tempfile(audio)
    try:
        text = _transcribe_audio_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not text:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    return {"text": text}


@app.post("/listen", response_model=TranslationResult)
async def listen(audio: UploadFile = File(...)):
    """One-shot pipeline for the live app: audio in -> Whisper transcript -> Gemma translation out."""
    tmp_path = await _save_upload_to_tempfile(audio)
    try:
        text = _transcribe_audio_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not text:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    return await _translate_with_retry(text)


@app.post("/summarize", response_model=ConversationSummaryResult)
async def summarize(req: SummarizeRequest):
    return await summarize_conversation(req.conversation)


@app.get("/health")
async def health():
    return {"status": "ok", "provider": GEMMA_PROVIDER, "whisper_model": WHISPER_MODEL_SIZE}
