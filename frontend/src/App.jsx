import React, { useCallback, useEffect, useRef, useState } from 'react'

// Backend URL — override with VITE_API_BASE in a .env file if needed.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Whisper (via the backend's /listen endpoint) does the STT now, so we no
// longer depend on the browser's Web Speech API — this works in any browser
// that supports MediaRecorder + getUserMedia (all modern browsers).
const MIC_SUPPORTED =
  typeof window !== 'undefined' &&
  !!navigator.mediaDevices &&
  typeof window.MediaRecorder !== 'undefined'

// -----------------------------------------------------------------------
// Voice-activity tuning
// -----------------------------------------------------------------------
// The mic stream stays open the ENTIRE time the user is "listening" — we
// never re-request getUserMedia per utterance. Instead we watch volume
// continuously and cut a new audio segment every time we detect a pause,
// so the app behaves like "always on" rather than push-to-talk.
const VOLUME_THRESHOLD = 0.02 // RMS level considered "speech"
const SILENCE_MS = 900 // how long a pause has to last before we consider the utterance finished
const MIN_SPEECH_MS = 300 // ignore blips shorter than this (coughs, taps, etc.)

let turnCounter = 0
function nextId() {
  turnCounter += 1
  return turnCounter
}

// -----------------------------------------------------------------------
// TTS helpers
// -----------------------------------------------------------------------
// NOTE: Hindi voice availability depends entirely on the OS/browser's
// installed TTS voices. Chrome on desktop has the most reliable coverage
// for both en-IN and hi-IN voices — build and demo on Chrome desktop.
function pickVoice(voices, targetLang) {
  if (!voices || voices.length === 0) return null

  if (targetLang === 'hi') {
    return (
      voices.find((v) => v.lang === 'hi-IN') ||
      voices.find((v) => v.lang?.toLowerCase().startsWith('hi')) ||
      null
    )
  }

  // target_lang === 'en' — prefer en-IN, then fall back to en-US, then any English voice.
  return (
    voices.find((v) => v.lang === 'en-IN') ||
    voices.find((v) => v.lang === 'en-US') ||
    voices.find((v) => v.lang?.toLowerCase().startsWith('en')) ||
    null
  )
}

function speak(text, targetLang, voices) {
  if (!('speechSynthesis' in window) || !text) return
  window.speechSynthesis.cancel() // don't let utterances pile up/overlap
  const utterance = new SpeechSynthesisUtterance(text)
  const voice = pickVoice(voices, targetLang)
  if (voice) {
    utterance.voice = voice
    utterance.lang = voice.lang
  } else {
    // Fallback if no matching voice was found on this OS/browser.
    utterance.lang = targetLang === 'hi' ? 'hi-IN' : 'en-US'
  }
  utterance.rate = 0.98
  window.speechSynthesis.speak(utterance)
}

// -----------------------------------------------------------------------
// Language badge
// -----------------------------------------------------------------------
function LangBadge({ lang }) {
  const label = lang === 'hi' ? 'HI' : lang === 'en' ? 'EN' : '…'
  return <span className={`badge badge-${lang || 'pending'}`}>{label}</span>
}

// -----------------------------------------------------------------------
// Conversation bubble
// -----------------------------------------------------------------------
function Bubble({ turn, onRetry }) {
  const sideClass = turn.sourceLang === 'hi' ? 'bubble-right' : 'bubble-left'

  return (
    <div className={`bubble ${sideClass}`}>
      {turn.status !== 'processing' && (
        <div className="bubble-original">
          <LangBadge lang={turn.sourceLang} />
          <span className="bubble-text">{turn.originalText}</span>
        </div>
      )}

      {turn.status === 'processing' && (
        <div className="bubble-status">transcribing &amp; translating…</div>
      )}

      {turn.status === 'done' && (
        <div className="bubble-translated">
          <LangBadge lang={turn.targetLang} />
          <span className="bubble-text">{turn.translatedText}</span>
        </div>
      )}

      {turn.status === 'error' && (
        <div className="bubble-error">
          <span>{turn.errorMessage || 'Translation failed.'}</span>
          <button className="retry-btn" onClick={() => onRetry(turn.id)}>
            Retry
          </button>
        </div>
      )}
    </div>
  )
}

// -----------------------------------------------------------------------
// Conversation summary panel (rendered after "End Conversation")
// -----------------------------------------------------------------------
const URGENCY_STYLES = {
  LOW: { color: '#22c55e', label: 'LOW' },
  MEDIUM: { color: '#f59e0b', label: 'MEDIUM' },
  HIGH: { color: '#ef4444', label: 'HIGH' },
}

function SummaryPanel({ summary, onClose }) {
  const urgency = URGENCY_STYLES[summary.urgency] || URGENCY_STYLES.LOW
  const hasContext =
    summary.patient_context.known_conditions.length > 0 ||
    summary.patient_context.current_medications.length > 0 ||
    summary.patient_context.known_allergies.length > 0

  return (
    <div className="summary-overlay">
      <div className="summary-card">
        <button className="summary-close" onClick={onClose} aria-label="Close summary">
          ×
        </button>

        <h2 className="summary-title">🩺 Symptoms Detected</h2>
        {summary.symptoms.length > 0 ? (
          <ul className="summary-checklist">
            {summary.symptoms.map((s) => (
              <li key={s}>✓ {s}</li>
            ))}
          </ul>
        ) : (
          <p className="summary-empty">No specific symptoms mentioned.</p>
        )}

        <hr className="summary-divider" />

        <h2 className="summary-title">⏱ Duration</h2>
        <p>{summary.duration || 'Not mentioned'}</p>

        <hr className="summary-divider" />

        <h2 className="summary-title">🚨 Urgency</h2>
        <p className="summary-urgency" style={{ color: urgency.color }}>
          {urgency.label}
        </p>

        <hr className="summary-divider" />

        <h2 className="summary-title">📋 Suggested Department</h2>
        <p>{summary.suggested_department}</p>

        <hr className="summary-divider" />

        <h2 className="summary-title">🧠 Conversation Summary</h2>
        <p>{summary.summary}</p>

        {hasContext && (
          <>
            <hr className="summary-divider" />
            <h2 className="summary-title">🧠 Patient Context</h2>

            {summary.patient_context.known_conditions.length > 0 && (
              <div className="summary-context-group">
                <h3>Known Conditions</h3>
                <ul>
                  {summary.patient_context.known_conditions.map((c) => (
                    <li key={c}>• {c}</li>
                  ))}
                </ul>
              </div>
            )}

            {summary.patient_context.current_medications.length > 0 && (
              <div className="summary-context-group">
                <h3>Current Medication</h3>
                <ul>
                  {summary.patient_context.current_medications.map((m) => (
                    <li key={m}>• {m}</li>
                  ))}
                </ul>
              </div>
            )}

            {summary.patient_context.known_allergies.length > 0 && (
              <div className="summary-context-group">
                <h3>Known Allergy</h3>
                <ul>
                  {summary.patient_context.known_allergies.map((a) => (
                    <li key={a}>• {a}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        <p className="summary-disclaimer">
          AI-generated from the conversation transcript — a documentation aid, not a diagnosis. Verify with the patient before acting on it clinically.
        </p>
      </div>
    </div>
  )
}

// -----------------------------------------------------------------------
// Main app
// -----------------------------------------------------------------------
export default function App() {
  const [turns, setTurns] = useState([])
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false) // volume currently above threshold
  const [textInput, setTextInput] = useState('')
  const [voices, setVoices] = useState([])
  const [micError, setMicError] = useState('')
  const [summary, setSummary] = useState(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [summaryError, setSummaryError] = useState('')

  const scrollRef = useRef(null)

  // Continuous-listening plumbing. These are refs (not state) because they're
  // read/written from a requestAnimationFrame loop and must not trigger re-renders.
  const isListeningRef = useRef(false)
  const streamRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const speakingRef = useRef(false)
  const speechStartRef = useRef(0)
  const silenceStartRef = useRef(null)
  const rafRef = useRef(null)

  // Load available speech-synthesis voices (can load async in some browsers).
  useEffect(() => {
    function loadVoices() {
      setVoices(window.speechSynthesis.getVoices())
    }
    loadVoices()
    if ('speechSynthesis' in window) {
      window.speechSynthesis.onvoiceschanged = loadVoices
    }
  }, [])

  // Auto-scroll to latest turn.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [turns])

  const updateTurn = useCallback((id, patch) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)))
  }, [])

  // ---------------------------------------------------------------------
  // Send one audio segment to the backend (/listen = Whisper + Gemma in one call)
  // ---------------------------------------------------------------------
  const sendSegment = useCallback(
    async (blob) => {
      const id = nextId()
      setTurns((prev) => [
        ...prev,
        {
          id,
          status: 'processing',
          originalText: '',
          sourceLang: null,
          targetLang: null,
          translatedText: '',
        },
      ])

      try {
        const formData = new FormData()
        formData.append('audio', blob, 'segment.webm')

        const res = await fetch(`${API_BASE}/listen`, {
          method: 'POST',
          body: formData,
        })

        if (!res.ok) {
          const body = await res.json().catch(() => ({}))
          throw new Error(body.detail || `HTTP ${res.status}`)
        }
        const data = await res.json()

        updateTurn(id, {
          status: 'done',
          originalText: data.original_text,
          sourceLang: data.source_lang,
          targetLang: data.target_lang,
          translatedText: data.translated_text,
        })

        speak(data.translated_text, data.target_lang, voices)
      } catch (err) {
        updateTurn(id, { status: 'error', errorMessage: err.message, _blob: blob })
      }
    },
    [updateTurn, voices],
  )

  const retryTurn = useCallback(
    (id) => {
      const turn = turns.find((t) => t.id === id)
      if (turn?._blob) sendSegment(turn._blob)
    },
    [turns, sendSegment],
  )

  // Manual text fallback still available (typed input skips Whisper entirely).
  const submitTypedText = useCallback(
    (text) => {
      const trimmed = text.trim()
      if (!trimmed) return
      const id = nextId()
      setTurns((prev) => [
        ...prev,
        {
          id,
          status: 'processing',
          originalText: trimmed,
          sourceLang: null,
          targetLang: null,
          translatedText: '',
        },
      ])
      fetch(`${API_BASE}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          const data = await res.json()
          updateTurn(id, {
            status: 'done',
            sourceLang: data.source_lang,
            targetLang: data.target_lang,
            translatedText: data.translated_text,
          })
          speak(data.translated_text, data.target_lang, voices)
        })
        .catch((err) => updateTurn(id, { status: 'error', errorMessage: err.message }))
    },
    [updateTurn, voices],
  )

  // ---------------------------------------------------------------------
  // Segment-level MediaRecorder control (the mic STREAM stays open the
  // whole session — only these short-lived recorder instances start/stop).
  // ---------------------------------------------------------------------
  const startSegmentRecorder = useCallback(() => {
    if (!streamRef.current) return
    chunksRef.current = []
    const recorder = new MediaRecorder(streamRef.current, { mimeType: 'audio/webm' })
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.start()
    recorderRef.current = recorder
  }, [])

  const endSegmentAndRestart = useCallback(() => {
    const recorder = recorderRef.current
    if (!recorder) return

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      chunksRef.current = []
      if (blob.size > 0) sendSegment(blob)
      // Immediately open the next segment so the mic never actually stops
      // listening between utterances.
      if (isListeningRef.current) startSegmentRecorder()
    }
    recorder.stop()
  }, [sendSegment, startSegmentRecorder])

  // Volume-monitoring loop — decides when an utterance has ended.
  const monitorVolume = useCallback(() => {
    if (!isListeningRef.current || !analyserRef.current) return

    const analyser = analyserRef.current
    const data = new Uint8Array(analyser.fftSize)
    analyser.getByteTimeDomainData(data)

    let sumSquares = 0
    for (let i = 0; i < data.length; i++) {
      const centered = (data[i] - 128) / 128
      sumSquares += centered * centered
    }
    const rms = Math.sqrt(sumSquares / data.length)
    const now = performance.now()

    if (rms > VOLUME_THRESHOLD) {
      if (!speakingRef.current) {
        speakingRef.current = true
        speechStartRef.current = now
        setIsSpeaking(true)
      }
      silenceStartRef.current = null
    } else if (speakingRef.current) {
      if (silenceStartRef.current === null) {
        silenceStartRef.current = now
      } else if (now - silenceStartRef.current > SILENCE_MS) {
        speakingRef.current = false
        setIsSpeaking(false)
        silenceStartRef.current = null
        const spokeLongEnough = now - speechStartRef.current > MIN_SPEECH_MS
        if (spokeLongEnough) {
          endSegmentAndRestart()
        } else {
          // Too short to be real speech — just restart the segment, discard it.
          chunksRef.current = []
        }
      }
    }

    rafRef.current = requestAnimationFrame(monitorVolume)
  }, [endSegmentAndRestart])

  const startListening = useCallback(async () => {
    if (!MIC_SUPPORTED) {
      setMicError('Microphone capture is not supported in this browser. Use the text box below.')
      return
    }
    setMicError('')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const audioContext = new (window.AudioContext || window.webkitAudioContext)()
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      audioContextRef.current = audioContext
      analyserRef.current = analyser

      isListeningRef.current = true
      setIsListening(true)
      speakingRef.current = false
      silenceStartRef.current = null

      startSegmentRecorder()
      rafRef.current = requestAnimationFrame(monitorVolume)
    } catch (err) {
      setMicError(`Could not access microphone: ${err.message}`)
    }
  }, [monitorVolume, startSegmentRecorder])

  const stopListening = useCallback(() => {
    isListeningRef.current = false
    setIsListening(false)
    setIsSpeaking(false)

    if (rafRef.current) cancelAnimationFrame(rafRef.current)

    // Flush whatever's currently being recorded as a final segment, then tear everything down.
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        chunksRef.current = []
        const spokeLongEnough =
          speakingRef.current && performance.now() - speechStartRef.current > MIN_SPEECH_MS
        if (blob.size > 0 && spokeLongEnough) sendSegment(blob)
      }
      recorder.stop()
    }

    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    audioContextRef.current?.close()
    audioContextRef.current = null
    analyserRef.current = null
  }, [sendSegment])

  // Clean up on unmount.
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      streamRef.current?.getTracks().forEach((t) => t.stop())
      audioContextRef.current?.close()
    }
  }, [])

  const handleMicClick = () => {
    if (isListening) {
      stopListening()
    } else {
      startListening()
    }
  }

  const handleTextSubmit = (e) => {
    e.preventDefault()
    submitTypedText(textInput)
    setTextInput('')
  }

  // ---------------------------------------------------------------------
  // End conversation -> summarize
  // ---------------------------------------------------------------------
  const endConversation = useCallback(async () => {
    // Stop listening first, if still active — don't summarize mid-recording.
    if (isListeningRef.current) {
      stopListening()
    }

    const doneTurns = turns.filter((t) => t.status === 'done')
    if (doneTurns.length === 0) {
      setSummaryError('No completed turns to summarize yet.')
      return
    }

    // Normalize every turn to English for the summarizer, regardless of who
    // spoke which language. sourceLang 'en' -> assume doctor, 'hi' -> patient.
    // (Adjust this mapping if your conversation convention differs.)
    const conversation = doneTurns.map((t) => {
      const isDoctor = t.sourceLang === 'en'
      const englishText = t.sourceLang === 'en' ? t.originalText : t.translatedText
      return { speaker: isDoctor ? 'doctor' : 'patient', text: englishText }
    })

    setSummary(null)
    setSummaryError('')
    setSummaryLoading(true)
    try {
      const res = await fetch(`${API_BASE}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation }),
      })
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setSummary(data)
    } catch (err) {
      setSummaryError(err.message || 'Failed to generate summary.')
    } finally {
      setSummaryLoading(false)
    }
  }, [turns, stopListening])

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-row">
          <div>
            <h1>Doctor ⇄ Patient Voice Translator</h1>
            <p className="subtitle">Phase 1 — live speech translation (Whisper + Gemma), English ⇄ Hindi</p>
          </div>
          <button
            className="end-conversation-btn"
            onClick={endConversation}
            disabled={summaryLoading || turns.filter((t) => t.status === 'done').length === 0}
          >
            {summaryLoading ? 'Summarizing…' : 'End Conversation'}
          </button>
        </div>
        {summaryError && <div className="summary-error-banner">{summaryError}</div>}
      </header>

      <main className="conversation" ref={scrollRef}>
        {turns.length === 0 && (
          <div className="empty-state">
            Tap the mic and start talking — it stays on and auto-detects each time you pause.
          </div>
        )}
        {turns.map((turn) => (
          <Bubble key={turn.id} turn={turn} onRetry={retryTurn} />
        ))}
      </main>

      <footer className="controls">
        {micError && <div className="mic-error">{micError}</div>}

        <div className="mic-row">
          <button
            className={`mic-btn ${isListening ? 'mic-btn-listening' : ''} ${isSpeaking ? 'mic-btn-speaking' : ''}`}
            onClick={handleMicClick}
            aria-pressed={isListening}
          >
            {isListening && <span className="mic-pulse" />}
            <span className="mic-icon">{isListening ? '■' : '🎙'}</span>
          </button>
          <div className="mic-status">
            {isListening ? (
              <span className="listening-label">
                {isSpeaking ? 'Hearing you…' : 'Listening — always on, speak whenever'}
              </span>
            ) : (
              <span>Tap to start — the mic stays on until you tap it again</span>
            )}
          </div>
        </div>

        <form className="text-fallback" onSubmit={handleTextSubmit}>
          <input
            type="text"
            placeholder="Type instead (demo safety net) — press Enter to send"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
          />
          <button type="submit">Send</button>
        </form>
      </footer>

      {summary && <SummaryPanel summary={summary} onClose={() => setSummary(null)} />}
    </div>
  )
}
