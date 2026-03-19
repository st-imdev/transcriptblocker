"""Configuration defaults for TranscriptBlocker."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERTURBATIONS_DIR = PROJECT_ROOT / "perturbations"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024  # ~64ms at 16kHz — good balance of latency vs stability
DTYPE = "float32"

# UAP generation defaults
DEFAULT_WHISPER_MODEL = "base"
UAP_DURATION_SEC = 1.0  # Short perturbations concentrate adversarial energy better
PGD_STEPS = 500
PGD_STEP_SIZE = 0.002
DEFAULT_SNR_DB = 20.0  # dB below speech — faint background hiss, complete ASR muting
MAX_PERTURBATION_NORM = 0.2  # L-inf clamp for the raw perturbation

# File paths
DEFAULT_UAP_PATH = PERTURBATIONS_DIR / "uap_whisper_base.wav"
