# AI Dubbing Artist 🎙️

A lightweight, powerful, multi-stage AI Dubbing Pipeline. It automates the extraction, transcription, diarization, translation (transcreation), voice cloning, and audio-video synchronization into a seamless workflow. This tool provides studio-quality dubbing.

## ✨ Features

- **Phase 1: Audio Extraction**
  Uses `audio-separator` (MDX-Net) to automatically isolate vocals from background music and sound effects.
- **Phase 2: STT & Diarization**
  Supports local, private transcription via Whisper or lightning-fast cloud transcription via Groq (Whisper). Uses `pyannote.audio` to detect multiple speakers.
- **Phase 3: Transcreation (Dubbing Director)**
  Employs powerful LLMs (Google Gemini, Groq) to translate dialogue (English to Hindi) while focusing on isochronous syllable matching. Includes an interactive AI Proofreader and automatic scene context analysis.
- **Phase 4: Voice Generation & Mixing**
  Supports the Kokoro TTS engine combined with Seed-VC for zero-shot voice cloning. Automatically syncs the generated audio to the original timestamps using `pyrubberband` and mixes it back with the original background music.
- **Studio Alignment Mode**
  Upload your own professional English script to skip STT and map dialogue accurately to audio timestamps.

## 📦 Installation

1. Clone the repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
