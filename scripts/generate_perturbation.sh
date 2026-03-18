#!/usr/bin/env bash
# Convenience wrapper for UAP generation.

set -euo pipefail

MODEL="${1:-base}"
SNR="${2:-32}"
STEPS="${3:-200}"

echo "Generating UAP perturbation..."
echo "  Model: whisper-${MODEL}"
echo "  Target SNR: ${SNR} dB"
echo "  PGD steps: ${STEPS}"
echo ""

python -m transcriptblocker.cli generate \
    --model "$MODEL" \
    --snr "$SNR" \
    --steps "$STEPS"
