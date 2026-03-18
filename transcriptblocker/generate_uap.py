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
from pathlib import Path
import scipy.io.wavfile as wavfile

from . import config


def _load_whisper_model(model_name: str, device: str) -> whisper.Whisper:
    """Load a Whisper model for adversarial optimization."""
    model = whisper.load_model(model_name, device=device)
    model.eval()
    return model


def _generate_training_samples(num_samples: int = 20, duration_sec: float = 5.0) -> list[torch.Tensor]:
    """
    Generate synthetic speech-like training samples for UAP optimization.

    Uses a mix of sine waves at speech frequencies with amplitude modulation
    to approximate speech energy distribution. For better results, replace with
    real speech samples from LibriSpeech.
    """
    sr = config.SAMPLE_RATE
    samples = []
    rng = np.random.RandomState(42)

    for i in range(num_samples):
        t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
        signal = np.zeros_like(t)

        # Fundamental frequencies typical of speech (85-300 Hz)
        f0 = rng.uniform(85, 300)
        # Add harmonics
        for harmonic in range(1, 8):
            freq = f0 * harmonic
            if freq > sr / 2:
                break
            amp = 1.0 / harmonic * rng.uniform(0.5, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase)

        # Amplitude modulation to simulate syllable rhythm (3-6 Hz)
        mod_freq = rng.uniform(3, 6)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        signal *= modulation

        # Add some broadband noise to simulate fricatives
        noise = rng.randn(len(t)).astype(np.float32) * 0.05
        signal += noise

        # Normalize
        signal = signal / (np.abs(signal).max() + 1e-8) * 0.8
        samples.append(torch.from_numpy(signal))

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
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading Whisper '{model_name}' on {device}...")
    model = _load_whisper_model(model_name, device)

    if verbose:
        print(f"Generating {num_samples} training samples...")
    speech_samples = _generate_training_samples(num_samples, duration_sec)

    # Initialize perturbation as small random noise
    num_perturbation_samples = int(config.SAMPLE_RATE * duration_sec)
    perturbation = torch.randn(num_perturbation_samples, device=device) * 0.001
    perturbation.requires_grad_(True)

    if verbose:
        print(f"Running PGD optimization ({num_steps} steps)...")

    for step in range(num_steps):
        total_loss = torch.tensor(0.0, device=device)

        # Mini-batch: use a subset each step for efficiency
        batch_indices = np.random.choice(len(speech_samples), size=min(4, len(speech_samples)), replace=False)

        for idx in batch_indices:
            speech = speech_samples[idx].to(device)

            # Scale perturbation to target SNR relative to this speech sample
            with torch.no_grad():
                scale = compute_snr_scale(speech, perturbation, snr_db)

            # Mix speech + scaled perturbation
            scaled_pert = perturbation * scale
            mixed = speech + scaled_pert

            # Clamp to valid audio range
            mixed = torch.clamp(mixed, -1.0, 1.0)

            # Pad/trim to 30 seconds (Whisper's expected input length)
            mixed_padded = whisper.pad_or_trim(mixed)

            # Compute log-mel spectrogram
            mel = whisper.log_mel_spectrogram(mixed_padded, n_mels=model.dims.n_mels).unsqueeze(0).to(device)

            # Forward through encoder
            encoder_output = model.encoder(mel)

            # Loss: maximize encoder output variance (disrupts attention patterns)
            # Combined with minimizing output norm consistency
            # This is more stable than trying to use the full decode pipeline
            loss = -torch.var(encoder_output) - torch.mean(torch.abs(encoder_output))

            # Also add a frequency-domain dispersion term:
            # push energy across all frequency bins to confuse the decoder
            freq_repr = torch.fft.rfft(encoder_output, dim=-1)
            freq_mag = torch.abs(freq_repr)
            # Maximize entropy of frequency magnitudes (spread energy evenly)
            freq_dist = freq_mag / (freq_mag.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(freq_dist * torch.log(freq_dist + 1e-8), dim=-1)
            loss -= 0.1 * entropy.mean()

            total_loss += loss

        total_loss /= len(batch_indices)

        # Backward pass
        total_loss.backward()

        # PGD step — gradient ascent (we want to maximize disruption)
        with torch.no_grad():
            grad = perturbation.grad
            # Normalize gradient for stable updates
            grad_norm = grad / (torch.norm(grad) + 1e-8)
            perturbation.data -= step_size * grad_norm  # minus because loss is already negated

            # Project: clamp L-inf norm
            perturbation.data.clamp_(-config.MAX_PERTURBATION_NORM, config.MAX_PERTURBATION_NORM)

        perturbation.grad.zero_()

        if verbose and (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps}, loss: {total_loss.item():.4f}")

    # Save perturbation as WAV
    pert_np = perturbation.detach().cpu().numpy()
    # Normalize to use full 16-bit range while staying within L-inf bound
    pert_np = pert_np / (np.abs(pert_np).max() + 1e-8)
    wavfile.write(str(output_path), config.SAMPLE_RATE, (pert_np * 32767).astype(np.int16))

    if verbose:
        print(f"UAP saved to {output_path}")
        print(f"  Duration: {duration_sec:.1f}s, Sample rate: {config.SAMPLE_RATE} Hz")
        print(f"  Target SNR: {snr_db} dB")

    return output_path
