"""Unit tests for the audio mixer."""

import numpy as np
import pytest
import tempfile
import scipy.io.wavfile as wavfile
from pathlib import Path

from transcriptblocker.audio_mixer import AudioMixer
from transcriptblocker import config


@pytest.fixture
def dummy_uap(tmp_path):
    """Create a dummy UAP .wav file for testing."""
    duration = 2.0
    sr = config.SAMPLE_RATE
    samples = int(sr * duration)
    # Simple sine wave perturbation
    t = np.linspace(0, duration, samples, dtype=np.float32)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.01).astype(np.float32)
    path = tmp_path / "test_uap.wav"
    wavfile.write(str(path), sr, (signal * 32767).astype(np.int16))
    return path


class TestAudioMixerUnit:
    """Unit tests that don't require real audio devices."""

    def test_uap_loading(self, dummy_uap):
        """Test that UAP file is loaded and converted correctly."""
        sr, data = wavfile.read(str(dummy_uap))
        assert sr == config.SAMPLE_RATE
        assert len(data) == int(config.SAMPLE_RATE * 2.0)

    def test_uap_file_not_found(self):
        """Test error when UAP file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="UAP file not found"):
            AudioMixer(uap_path="/nonexistent/path.wav")

    def test_compute_gain(self, dummy_uap, monkeypatch):
        """Test SNR gain computation."""
        # Monkey-patch device detection to avoid needing real devices
        monkeypatch.setattr("transcriptblocker.audio_mixer.find_default_mic", lambda: 0)
        monkeypatch.setattr("transcriptblocker.audio_mixer.find_blackhole", lambda: 1)
        monkeypatch.setattr("transcriptblocker.audio_mixer.validate_devices", lambda a, b: None)

        mixer = AudioMixer(uap_path=dummy_uap, input_device=0, output_device=1)

        # Test with a known speech signal
        speech = np.sin(np.linspace(0, 2 * np.pi * 200, 1024)).astype(np.float32)
        gain = mixer._compute_gain(speech)

        # At 32dB SNR, gain should be small
        assert 0 < gain < 1.0

        # Silence should produce tiny gain
        silence = np.zeros(1024, dtype=np.float32)
        gain_silence = mixer._compute_gain(silence)
        assert gain_silence == pytest.approx(0.001)

    def test_uap_chunk_looping(self, dummy_uap, monkeypatch):
        """Test that UAP buffer loops correctly."""
        monkeypatch.setattr("transcriptblocker.audio_mixer.find_default_mic", lambda: 0)
        monkeypatch.setattr("transcriptblocker.audio_mixer.find_blackhole", lambda: 1)
        monkeypatch.setattr("transcriptblocker.audio_mixer.validate_devices", lambda a, b: None)

        mixer = AudioMixer(uap_path=dummy_uap, input_device=0, output_device=1)

        uap_len = len(mixer._uap_buffer)

        # Request more samples than the buffer length — should loop
        chunk = mixer._get_uap_chunk(uap_len + 100)
        assert len(chunk) == uap_len + 100

        # The wrapped portion should match the beginning of the buffer
        np.testing.assert_array_almost_equal(
            chunk[uap_len:], mixer._uap_buffer[:100]
        )

    def test_snr_adjustment(self, dummy_uap, monkeypatch):
        """Test that SNR can be adjusted dynamically."""
        monkeypatch.setattr("transcriptblocker.audio_mixer.find_default_mic", lambda: 0)
        monkeypatch.setattr("transcriptblocker.audio_mixer.find_blackhole", lambda: 1)
        monkeypatch.setattr("transcriptblocker.audio_mixer.validate_devices", lambda a, b: None)

        mixer = AudioMixer(uap_path=dummy_uap, input_device=0, output_device=1)
        assert mixer.snr_db == config.DEFAULT_SNR_DB

        mixer.set_snr(20.0)
        assert mixer.snr_db == 20.0
