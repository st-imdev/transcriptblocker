"""
Real-time audio mixer: reads mic input, mixes in UAP perturbation, writes to virtual output.

Uses sounddevice for low-latency audio I/O with callback-based streaming.
"""

import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from pathlib import Path

from . import config
from .audio_devices import find_blackhole, find_default_mic, get_device_name, validate_devices


class AudioMixer:
    """
    Real-time audio mixer that overlays adversarial perturbation onto mic input.

    Reads from a real microphone, mixes in a pre-generated UAP signal at a
    controlled SNR, and outputs to a virtual audio device (e.g. BlackHole).
    """

    def __init__(
        self,
        uap_path: Path | str = config.DEFAULT_UAP_PATH,
        snr_db: float = config.DEFAULT_SNR_DB,
        input_device: int | None = None,
        output_device: int | None = None,
        sample_rate: int = config.SAMPLE_RATE,
        block_size: int = config.BLOCK_SIZE,
    ):
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._running = False
        self._stream = None
        self._lock = threading.Lock()

        # Resolve devices
        self.input_device = input_device if input_device is not None else find_default_mic()
        self.output_device = output_device if output_device is not None else find_blackhole()

        if self.input_device is None:
            raise RuntimeError("No input microphone found")
        if self.output_device is None:
            raise RuntimeError(
                "BlackHole virtual audio device not found. "
                "Install it with: brew install blackhole-2ch"
            )

        validate_devices(self.input_device, self.output_device)

        # Load UAP perturbation
        uap_path = Path(uap_path)
        if not uap_path.exists():
            raise FileNotFoundError(
                f"UAP file not found: {uap_path}\n"
                "Run 'transcriptblocker generate' first to create one."
            )

        sr, uap_data = wavfile.read(str(uap_path))
        # Convert to float32 normalized to [-1, 1]
        if uap_data.dtype == np.int16:
            self._uap_buffer = uap_data.astype(np.float32) / 32768.0
        elif uap_data.dtype == np.float32:
            self._uap_buffer = uap_data
        else:
            self._uap_buffer = uap_data.astype(np.float32) / np.iinfo(uap_data.dtype).max

        # Resample if needed
        if sr != self.sample_rate:
            from scipy.signal import resample
            new_len = int(len(self._uap_buffer) * self.sample_rate / sr)
            self._uap_buffer = resample(self._uap_buffer, new_len).astype(np.float32)

        self._uap_pos = 0  # Current position in the looping UAP buffer

    def _compute_gain(self, speech_chunk: np.ndarray) -> float:
        """Compute UAP gain to achieve target SNR relative to speech level."""
        speech_power = np.mean(speech_chunk ** 2)
        if speech_power < 1e-10:
            # Silence — use a very small fixed gain to avoid amplifying noise
            return 0.001

        uap_power = np.mean(self._uap_buffer ** 2)
        if uap_power < 1e-12:
            return 1.0

        target_noise_power = speech_power / (10 ** (self.snr_db / 10))
        return np.sqrt(target_noise_power / uap_power)

    def _get_uap_chunk(self, num_samples: int) -> np.ndarray:
        """Get next chunk from the looping UAP buffer."""
        uap_len = len(self._uap_buffer)
        chunk = np.empty(num_samples, dtype=np.float32)

        remaining = num_samples
        write_pos = 0
        while remaining > 0:
            available = min(remaining, uap_len - self._uap_pos)
            chunk[write_pos:write_pos + available] = self._uap_buffer[self._uap_pos:self._uap_pos + available]
            self._uap_pos = (self._uap_pos + available) % uap_len
            write_pos += available
            remaining -= available

        return chunk

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """sounddevice stream callback — runs in audio thread."""
        if status:
            # Log but don't crash on xruns
            pass

        # indata shape: (frames, channels) — take first channel
        mic_mono = indata[:, 0]

        # Get corresponding UAP chunk
        uap_chunk = self._get_uap_chunk(frames)

        # Compute gain for target SNR
        gain = self._compute_gain(mic_mono)

        # Mix
        mixed = mic_mono + uap_chunk * gain

        # Clip to prevent distortion
        mixed = np.clip(mixed, -1.0, 1.0)

        # Write to all output channels
        for ch in range(outdata.shape[1]):
            outdata[:, ch] = mixed

    def start(self) -> None:
        """Start the audio mixing stream."""
        with self._lock:
            if self._running:
                return

            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=config.CHANNELS,
                dtype=config.DTYPE,
                callback=self._audio_callback,
                latency="low",
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        """Stop the audio mixing stream."""
        with self._lock:
            if not self._running:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def set_snr(self, snr_db: float) -> None:
        """Adjust the perturbation strength (SNR in dB). Higher = more subtle."""
        self.snr_db = snr_db

    def get_status(self) -> dict:
        """Return current mixer status."""
        return {
            "running": self._running,
            "snr_db": self.snr_db,
            "input_device": get_device_name(self.input_device),
            "output_device": get_device_name(self.output_device),
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "latency_ms": self.block_size / self.sample_rate * 1000,
        }
