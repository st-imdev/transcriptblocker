#!/usr/bin/env python3
"""
End-to-end test: record from mic, mix in UAP perturbation, transcribe both,
and compare results to measure perturbation effectiveness.

Usage:
    python scripts/test_e2e.py
    python scripts/test_e2e.py --duration 10 --snr 28 --model small
"""

import argparse
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

from transcriptblocker import config


def record_audio(duration: float, sample_rate: int = config.SAMPLE_RATE, device: int | None = None) -> np.ndarray:
    """Record audio from the specified (or default) microphone."""
    # Use the device's native sample rate if it differs, then resample
    if device is not None:
        dev_info = sd.query_devices(device)
        native_sr = int(dev_info["default_samplerate"])
    else:
        dev_info = sd.query_devices(kind="input")
        native_sr = int(dev_info["default_samplerate"])

    print(f"Recording {duration}s from '{dev_info['name']}' at {native_sr} Hz...")
    print("  Speak now!")
    audio = sd.rec(
        int(duration * native_sr),
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("  Recording complete.")

    audio = audio.flatten()

    # Resample to target sample rate if needed
    if native_sr != sample_rate:
        from scipy.signal import resample
        new_len = int(len(audio) * sample_rate / native_sr)
        audio = resample(audio, new_len).astype(np.float32)
        print(f"  Resampled {native_sr} Hz -> {sample_rate} Hz")

    return audio


def load_uap(uap_path: Path, target_length: int, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load UAP perturbation and tile/trim to match target length."""
    if not uap_path.exists():
        print(f"ERROR: UAP file not found: {uap_path}")
        print("Run 'transcriptblocker generate' first to create one.")
        sys.exit(1)

    sr, uap_data = wavfile.read(str(uap_path))

    # Convert to float32 normalized to [-1, 1]
    if uap_data.dtype == np.int16:
        uap = uap_data.astype(np.float32) / 32768.0
    elif uap_data.dtype == np.float32:
        uap = uap_data
    else:
        uap = uap_data.astype(np.float32) / np.iinfo(uap_data.dtype).max

    # Resample if needed
    if sr != sample_rate:
        from scipy.signal import resample
        new_len = int(len(uap) * sample_rate / sr)
        uap = resample(uap, new_len).astype(np.float32)

    # Tile to cover the full recording, then trim
    if len(uap) < target_length:
        repeats = (target_length // len(uap)) + 1
        uap = np.tile(uap, repeats)
    uap = uap[:target_length]

    return uap


def mix_at_snr(clean: np.ndarray, perturbation: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    """
    Mix perturbation into clean audio at the specified SNR (dB).

    Returns the mixed audio and the gain applied to the perturbation.
    """
    speech_power = np.mean(clean ** 2)
    pert_power = np.mean(perturbation ** 2)

    if speech_power < 1e-10:
        print("WARNING: Recorded audio is nearly silent. Speak louder next time.")
        gain = 0.001
    elif pert_power < 1e-12:
        print("WARNING: UAP perturbation has near-zero energy.")
        gain = 1.0
    else:
        target_noise_power = speech_power / (10 ** (snr_db / 10))
        gain = np.sqrt(target_noise_power / pert_power)

    mixed = clean + perturbation * gain
    mixed = np.clip(mixed, -1.0, 1.0)
    return mixed, gain


def rms(signal: np.ndarray) -> float:
    """Compute RMS level of a signal."""
    return float(np.sqrt(np.mean(signal ** 2)))


def rms_db(signal: np.ndarray) -> float:
    """Compute RMS level in dBFS."""
    level = rms(signal)
    if level < 1e-10:
        return -100.0
    return 20 * np.log10(level)


def transcribe(audio_path: str, model_name: str):
    """Transcribe an audio file using Whisper."""
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"].strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute word error rate as Levenshtein distance on word lists / max length.

    This is a simplified WER: edit distance between word sequences divided by
    the length of the longer sequence. Returns a value between 0.0 and 1.0+.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words or not hyp_words:
        return 1.0

    # Levenshtein distance on word lists
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    distance = dp[n][m]
    max_len = max(n, m)
    return distance / max_len


def print_separator(char: str = "-", width: int = 60):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test: record, mix UAP, transcribe, and compare."
    )
    parser.add_argument(
        "--duration", type=float, default=5,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--snr", type=float, default=18,
        help="SNR in dB for mixing perturbation (default: 32)",
    )
    parser.add_argument(
        "--model", type=str, default="base",
        help="Whisper model name (default: base)",
    )
    parser.add_argument(
        "--uap-path", type=str, default=None,
        help=f"Path to UAP .wav file (default: {config.DEFAULT_UAP_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/tmp/transcriptblocker_test",
        help="Directory for output wav files (default: /tmp/transcriptblocker_test)",
    )
    parser.add_argument(
        "--input-device", type=int, default=None,
        help="Input device index (default: system default mic)",
    )
    args = parser.parse_args()

    uap_path = Path(args.uap_path) if args.uap_path else config.DEFAULT_UAP_PATH
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "clean.wav"
    mixed_path = output_dir / "mixed.wav"

    print_separator("=")
    print("TranscriptBlocker End-to-End Test")
    print_separator("=")
    print(f"  Duration:    {args.duration}s")
    print(f"  SNR:         {args.snr} dB")
    print(f"  Model:       {args.model}")
    print(f"  UAP path:    {uap_path}")
    print(f"  Output dir:  {output_dir}")
    print_separator("=")
    print()

    # Step 1: Record audio
    clean_audio = record_audio(args.duration, device=args.input_device)
    wavfile.write(str(clean_path), config.SAMPLE_RATE, (clean_audio * 32767).astype(np.int16))
    print(f"  Saved clean audio to {clean_path}")
    print()

    # Step 2: Load UAP and mix
    print("Loading UAP perturbation...")
    uap = load_uap(uap_path, len(clean_audio))
    mixed_audio, gain = mix_at_snr(clean_audio, uap, args.snr)
    wavfile.write(str(mixed_path), config.SAMPLE_RATE, (mixed_audio * 32767).astype(np.int16))
    print(f"  Saved mixed audio to {mixed_path}")
    print()

    # Step 3: Print RMS levels
    perturbation_contribution = uap * gain
    print_separator()
    print("Audio Levels")
    print_separator()
    print(f"  Clean audio RMS:         {rms(clean_audio):.6f}  ({rms_db(clean_audio):+.1f} dBFS)")
    print(f"  Perturbation RMS:        {rms(perturbation_contribution):.6f}  ({rms_db(perturbation_contribution):+.1f} dBFS)")
    print(f"  Mixed audio RMS:         {rms(mixed_audio):.6f}  ({rms_db(mixed_audio):+.1f} dBFS)")
    print(f"  Perturbation gain:       {gain:.6f}")
    print(f"  Actual SNR:              {args.snr:.1f} dB")
    print()

    # Step 4: Transcribe both
    print_separator()
    print(f"Transcribing with Whisper ({args.model})...")
    print_separator()
    print()

    print("  Transcribing clean audio...")
    clean_text = transcribe(str(clean_path), args.model)

    print("  Transcribing mixed audio...")
    mixed_text = transcribe(str(mixed_path), args.model)
    print()

    # Step 5: Print results side by side
    print_separator("=")
    print("Results")
    print_separator("=")
    print()
    print(f"  CLEAN:  {clean_text if clean_text else '(empty transcription)'}")
    print()
    print(f"  MIXED:  {mixed_text if mixed_text else '(empty transcription)'}")
    print()

    # Step 6: Compute WER
    wer = word_error_rate(clean_text, mixed_text)
    print_separator()
    print("Metrics")
    print_separator()
    print(f"  Clean word count:        {len(clean_text.split()) if clean_text else 0}")
    print(f"  Mixed word count:        {len(mixed_text.split()) if mixed_text else 0}")
    print(f"  Word Error Rate (WER):   {wer:.1%}")
    print()

    # Step 7: Verdict
    print_separator("=")
    effective = wer > 0.5
    if effective:
        print(f"  VERDICT: EFFECTIVE  (WER {wer:.1%} > 50% threshold)")
    else:
        print(f"  VERDICT: NOT EFFECTIVE  (WER {wer:.1%} <= 50% threshold)")
    print_separator("=")

    if not clean_text:
        print()
        print("  NOTE: Clean transcription was empty. This usually means the")
        print("  microphone didn't pick up speech. Try speaking louder or")
        print("  increasing --duration.")

    return 0 if effective else 1


if __name__ == "__main__":
    sys.exit(main())
