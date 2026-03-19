"""
Offline Universal Adversarial Perturbation (UAP) generation against Whisper.

Implements PGD-based optimization to find a single perturbation signal that,
when added to any speech at ~30-35dB SNR, causes Whisper to produce garbage output.

Based on:
- Neekhara et al. (Interspeech 2019): Universal adversarial perturbations for ASR
- AdvPulse (ACM CCS 2020): Synchronization-free perturbations
"""

import torch
import torch.nn.functional as F
import numpy as np
import whisper
from whisper.tokenizer import get_tokenizer
from pathlib import Path
import scipy.io.wavfile as wavfile

from . import config


def _load_whisper_model(model_name: str, device: str) -> whisper.Whisper:
    """Load a Whisper model for adversarial optimization."""
    model = whisper.load_model(model_name, device=device)
    model.eval()
    return model


def _load_training_samples(num_samples: int = 20, duration_sec: float = 5.0, verbose: bool = True) -> list[torch.Tensor]:
    """
    Load real speech samples from LibriSpeech for UAP optimization.

    Downloads the test-clean subset (small) via torchaudio on first run.
    Falls back to the user's previously recorded test audio if available.
    """
    import torchaudio

    sr = config.SAMPLE_RATE
    target_len = int(sr * duration_sec)
    samples = []

    # Try loading LibriSpeech test-clean
    cache_dir = config.PROJECT_ROOT / ".cache"
    cache_dir.mkdir(exist_ok=True)

    try:
        if verbose:
            print(f"  Loading LibriSpeech test-clean ({num_samples} samples)...")
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(cache_dir),
            url="test-clean",
            download=True,
        )

        count = 0
        for waveform, sample_rate, *_ in dataset:
            if count >= num_samples:
                break

            # Convert to mono float32 at target sample rate
            audio = waveform[0]  # first channel
            if sample_rate != sr:
                audio = torchaudio.functional.resample(audio, sample_rate, sr)

            # Pad or trim to target duration
            if len(audio) < target_len:
                audio = F.pad(audio, (0, target_len - len(audio)))
            else:
                # Take a random offset for variety
                max_start = len(audio) - target_len
                start = np.random.randint(0, max(1, max_start))
                audio = audio[start:start + target_len]

            # Normalize
            audio = audio / (torch.abs(audio).max() + 1e-8) * 0.8
            samples.append(audio)
            count += 1

        if verbose:
            print(f"  Loaded {len(samples)} real speech samples.")

    except Exception as e:
        if verbose:
            print(f"  LibriSpeech download failed: {e}")
            print("  Falling back to synthetic samples...")

    # Also add user's recorded audio if available
    user_recording = Path("/tmp/transcriptblocker_test/clean.wav")
    if user_recording.exists():
        try:
            import scipy.io.wavfile as wavfile_reader
            rec_sr, rec_data = wavfile_reader.read(str(user_recording))
            if rec_data.dtype == np.int16:
                rec_audio = torch.from_numpy(rec_data.astype(np.float32) / 32768.0)
            else:
                rec_audio = torch.from_numpy(rec_data.astype(np.float32))
            if rec_sr != sr:
                rec_audio = torchaudio.functional.resample(rec_audio, rec_sr, sr)
            # Trim/pad
            if len(rec_audio) >= target_len:
                rec_audio = rec_audio[:target_len]
            else:
                rec_audio = F.pad(rec_audio, (0, target_len - len(rec_audio)))
            rec_audio = rec_audio / (torch.abs(rec_audio).max() + 1e-8) * 0.8
            samples.append(rec_audio)
            if verbose:
                print("  Added user's recorded audio as training sample.")
        except Exception:
            pass

    if not samples:
        raise RuntimeError(
            "No training samples available. Check your internet connection "
            "for LibriSpeech download, or record a test first with:\n"
            "  uv run python scripts/test_e2e.py"
        )

    return samples


def compute_snr_scale(signal: torch.Tensor, perturbation: torch.Tensor, target_snr_db: float) -> float:
    """Compute the scaling factor for perturbation to achieve target SNR."""
    signal_power = torch.mean(signal ** 2).item()
    perturbation_power = torch.mean(perturbation ** 2).item()

    if perturbation_power < 1e-12:
        return 1.0

    # SNR = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / 10^(SNR/10)
    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / perturbation_power)
    return scale


def generate_uap(
    model_name: str = config.DEFAULT_WHISPER_MODEL,
    output_path: Path | str = config.DEFAULT_UAP_PATH,
    snr_db: float = config.DEFAULT_SNR_DB,
    num_steps: int = config.PGD_STEPS,
    step_size: float = config.PGD_STEP_SIZE,
    duration_sec: float = config.UAP_DURATION_SEC,
    device: str | None = None,
    num_samples: int = 20,
    verbose: bool = True,
) -> Path:
    """
    Generate a Universal Adversarial Perturbation targeting Whisper.

    Uses PGD (Projected Gradient Descent) to maximize Whisper's transcription loss,
    producing a single audio signal that disrupts ASR on arbitrary speech.

    Args:
        model_name: Whisper model size (tiny, base, small, medium, large)
        output_path: Where to save the resulting .wav file
        snr_db: Target SNR in dB (higher = more imperceptible, less effective)
        num_steps: Number of PGD optimization steps
        step_size: Gradient step size
        duration_sec: Duration of the perturbation signal in seconds
        device: torch device (auto-detected if None)
        num_samples: Number of training speech samples
        verbose: Print progress

    Returns:
        Path to the saved perturbation .wav file
    """
    # Force CPU — adversarial perturbations don't transfer between MPS and CPU
    # due to floating-point precision differences, and Whisper inference typically
    # runs on CPU. Optimizing on CPU ensures the perturbation works at test time.
    device = "cpu"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading Whisper '{model_name}' on {device}...")
    model = _load_whisper_model(model_name, device)

    if verbose:
        print(f"Loading training samples...")
    speech_samples = _load_training_samples(num_samples, duration_sec, verbose=verbose)

    # Initialize perturbation as small random noise
    num_perturbation_samples = int(config.SAMPLE_RATE * duration_sec)
    perturbation = torch.randn(num_perturbation_samples) * 0.001
    perturbation.requires_grad_(True)

    # Get tokenizer and decoder prompt tokens
    tokenizer = get_tokenizer(model.is_multilingual)
    sot_sequence = torch.tensor(
        [[tokenizer.sot, tokenizer.sot + 1, tokenizer.transcribe, tokenizer.no_timestamps]],
        dtype=torch.long,
    )

    if verbose:
        print(f"Running PGD optimization ({num_steps} steps)...")
        print(f"  L-inf bound: {config.MAX_PERTURBATION_NORM}")

    for step in range(num_steps):
        total_loss = torch.tensor(0.0)

        # Mini-batch: use a subset each step for efficiency
        batch_indices = np.random.choice(len(speech_samples), size=min(4, len(speech_samples)), replace=False)

        for idx in batch_indices:
            speech = speech_samples[idx]

            # Add perturbation at FULL L-inf strength (no SNR scaling).
            # The audio_mixer handles SNR scaling at runtime.
            mixed = speech + perturbation
            mixed = torch.clamp(mixed, -1.0, 1.0)

            # Compute clean and perturbed encoder outputs
            with torch.no_grad():
                clean_padded = whisper.pad_or_trim(speech)
                clean_mel = whisper.log_mel_spectrogram(clean_padded, n_mels=model.dims.n_mels).unsqueeze(0)
                clean_enc = model.encoder(clean_mel)

            mixed_padded = whisper.pad_or_trim(mixed)
            mel = whisper.log_mel_spectrogram(mixed_padded, n_mels=model.dims.n_mels).unsqueeze(0)
            adv_enc = model.encoder(mel)

            # ENCODER COSINE LOSS: maximize cosine distance between clean
            # and perturbed encoder outputs. This is the winning approach
            # from H100 experiments — outperforms decoder-based losses.
            cos_sim = F.cosine_similarity(
                adv_enc.flatten().unsqueeze(0),
                clean_enc.flatten().unsqueeze(0),
            )
            loss = cos_sim.mean()  # minimize similarity = maximize disruption

            total_loss += loss

        total_loss /= len(batch_indices)

        # Backward pass
        total_loss.backward()

        # PGD step with sign-of-gradient (FGSM-style)
        with torch.no_grad():
            perturbation.data -= step_size * perturbation.grad.sign()
            perturbation.data.clamp_(-config.MAX_PERTURBATION_NORM, config.MAX_PERTURBATION_NORM)

        perturbation.grad.zero_()

        if verbose and (step + 1) % 50 == 0:
            pert_rms = torch.sqrt(torch.mean(perturbation.data ** 2)).item()
            print(f"  Step {step + 1}/{num_steps}, loss: {total_loss.item():.4f}, pert RMS: {pert_rms:.4f}")

    # Save perturbation as WAV (normalized to 16-bit range)
    pert_np = perturbation.detach().numpy()
    pert_np = pert_np / (np.abs(pert_np).max() + 1e-8)
    wavfile.write(str(output_path), config.SAMPLE_RATE, (pert_np * 32767).astype(np.int16))

    if verbose:
        print(f"UAP saved to {output_path}")
        print(f"  Duration: {duration_sec:.1f}s, Sample rate: {config.SAMPLE_RATE} Hz")

    return output_path
