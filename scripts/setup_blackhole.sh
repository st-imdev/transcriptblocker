#!/usr/bin/env bash
# Install and verify BlackHole 2ch virtual audio device on macOS.

set -euo pipefail

echo "=== BlackHole 2ch Setup ==="

# Check if already installed
if brew list blackhole-2ch &>/dev/null; then
    echo "BlackHole 2ch is already installed."
else
    echo "Installing BlackHole 2ch via Homebrew..."
    brew install blackhole-2ch
    echo "BlackHole 2ch installed."
fi

echo ""
echo "=== Setup Instructions ==="
echo ""
echo "1. Open your video call app (Zoom, Teams, Meet, etc.)"
echo "2. Go to audio settings"
echo "3. Set the MICROPHONE to 'BlackHole 2ch'"
echo "   (TranscriptBlocker will send your mic audio + perturbation here)"
echo ""
echo "4. Your SPEAKER should remain set to your normal output device"
echo "   (headphones, speakers, etc.)"
echo ""
echo "5. Run: transcriptblocker devices"
echo "   to verify BlackHole appears in the device list"
echo ""
echo "=== How It Works ==="
echo ""
echo "  Real Mic → TranscriptBlocker → BlackHole 2ch → Zoom/Teams/Meet"
echo "                    ↑"
echo "           UAP perturbation mixed in"
echo ""
echo "  Other participants hear you normally."
echo "  ASR (Granola, Otter, etc.) gets garbled transcription."
