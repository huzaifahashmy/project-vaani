# Doctor ⇄ Patient Voice Translator — Phase 1

Translation + TTS only. No medical extraction, no graph, no urgency logic.
Goal: mic → speak (EN or HI) → see transcript → see translation → hear it spoken, in
near real time, reliably enough to demo live.

## Stack

- **Frontend**: React (Vite), single page, no router
- **STT**: Browser Web Speech API (`webkitSpeechRecognition`) — no backend STT
- **TTS**: Browser `speechSynthesis` API — no backend TTS
- **Translation**: FastAPI `/translate` endpoint that calls Gemma
- **Backend**: FastAPI, stateless, no DB/auth

⚠️ **Build and demo on Chrome desktop.** Web Speech API STT and Hindi TTS voice
availability are inconsistent across browsers/OSes — Chrome desktop has the most
reliable coverage of both.

---

## 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Point it at a Gemma model

Two providers are wired up out of the box — pick one in `.env`:

**Option A — Ollama (local, recommended for a demo):**

```bash
ollama pull gemma2:9b
ollama serve   # usually already running in the background after install
```

`.env`:

```
GEMMA_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:9b
```

**Option B — Google AI Studio (hosted):**

```
GEMMA_PROVIDER=google
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemma-3-27b-it
```

### Run it

```bash
uvicorn main:app --reload --port 8000
```

Sanity check: `curl -X POST http://localhost:8000/translate -H "Content-Type: application/json" -d '{"text":"Where does it hurt?"}'`

---

## 2. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (default `http://localhost:5173`) in **Chrome desktop**.

If your backend isn't on `http://localhost:8000`, create `frontend/.env`:

```
VITE_API_BASE=http://your-backend-host:8000
```

---

## How it works

1. Tap the mic. A pulsing red ring + "Listening…" label confirms it's live.
2. Speak in English or Hindi. On speech end, the raw transcript renders immediately
   in a chat bubble (aligned left for English, right for Hindi).
3. The transcript is POSTed to `/translate`. While waiting, the bubble shows
   "translating…".
4. The translation renders beneath the original text in the same bubble, with a
   language badge (EN/HI), and is spoken aloud automatically.
5. The app tracks which language it expects to hear next (it flips based on the
   last translation's target language) so the conversation naturally alternates
   doctor ⇄ patient — with a manual EN/HI toggle as an override.
6. If `/translate` fails, the bubble shows an inline **Retry** button instead of
   crashing the conversation.
7. A text input below the mic is a full fallback — same flow, no mic required.
   Use this as a live-demo safety net if STT or the mic misbehaves.

Conversation state is a single `useState` array in `App.jsx` — nothing persists
across a refresh, there's no login, no routing, no local storage.

## Known constraints (by design, Phase 1 scope)

- No language auto-detection at the STT layer — the browser's SpeechRecognition
  API requires a language hint before it starts listening. The app infers the
  _expected_ next language from the previous turn's translation target, with a
  manual EN/HI override for safety.
- No persistence, no auth, no medical logic — that's Phase 2+.
- Hindi TTS quality/availability depends on the OS's installed voices.
