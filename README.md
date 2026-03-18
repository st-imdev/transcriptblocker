# TranscriptBlocker

Defeat unauthorized AI transcription of your calls. Injects adversarial audio perturbations into your microphone signal that are **imperceptible to humans** but cause Whisper-based ASR (Granola, Otter.ai, etc.) to produce garbage.

Other call participants hear you normally. Only the machine transcription breaks.

## How it works

1. **Pre-compute** a Universal Adversarial Perturbation (UAP) optimized against Whisper using PGD
2. **Mix** the perturbation into your live mic audio at ~30–35 dB SNR (inaudible — 1/1600th the power of speech)
3. **Route** the mixed audio through BlackHole (virtual audio device) to your video call app
4. Whisper-based transcription tools capture the mixed signal and produce garbled output

```
Real Mic ──▶ TranscriptBlocker ──▶ BlackHole 2ch ──▶ Zoom / Teams / Meet
                    ▲                                        │
             UAP mixed in                              Granola / Otter
             (inaudible)                               captures this ──▶ garbage
```

## Setup

```bash
# 1. Install BlackHole virtual audio device (requires reboot)
brew install blackhole-2ch

# 2. Clone and install
git clone https://github.com/yourusername/transcriptblocker.git
cd transcriptblocker
uv venv && uv pip install -e .

# 3. Generate a perturbation (one-time, ~5 min on Apple Silicon)
uv run transcriptblocker generate

# 4. Configure your video call app's microphone to "BlackHole 2ch"
```

## Usage

### CLI

```bash
# Start blocking
uv run transcriptblocker start

# List audio devices
uv run transcriptblocker devices

# Test Whisper transcription on a file
uv run transcriptblocker test recording.wav

# Adjust strength (lower dB = more aggressive)
uv run transcriptblocker start --snr 25
```

### Menu bar app (macOS)

```bash
uv run transcriptblocker-gui
```

Adds a **TB** icon to your menu bar with:
- Start / Stop toggle
- Strength presets (Light 40 dB / Normal 32 dB / Aggressive 25 dB)
- Audio device selection
- One-click effectiveness test

### End-to-end test

```bash
# Records from your mic, mixes in the UAP, transcribes both, and compares
uv run python scripts/test_e2e.py
```

## UAP generation options

```bash
uv run transcriptblocker generate \
  --model base \     # Whisper model to target (tiny/base/small/medium)
  --snr 32 \         # Target SNR in dB — higher = more subtle
  --steps 200 \      # PGD optimization steps — more = better quality
  --duration 5.0     # Perturbation length in seconds (loops during playback)
```

## How effective is it?

At 30–35 dB SNR, universal adversarial perturbations achieve **80–100% word error rate** against Whisper while remaining imperceptible to human listeners.

### Research basis

- Neekhara et al. (Interspeech 2019) — Universal adversarial perturbations for speech recognition
- AdvPulse (ACM CCS 2020) — Subsecond adversarial perturbations for streaming ASR
- Schönherr et al. (2018) — Psychoacoustic hiding of adversarial audio
- CommanderUAP (2024) — High success rates against multiple ASR model families

## Requirements

- macOS (uses BlackHole + Core Audio)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [BlackHole 2ch](https://existential.audio/blackhole/) (free, open-source virtual audio driver)

## License

MIT
