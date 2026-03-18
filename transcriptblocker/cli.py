"""CLI interface for TranscriptBlocker."""

import signal
import sys
import time
from pathlib import Path

import click

from . import config


@click.group()
@click.version_option(version="0.1.0")
def main():
    """TranscriptBlocker — defeat ASR transcription with adversarial audio."""
    pass


@main.command()
@click.option("--model", default=config.DEFAULT_WHISPER_MODEL, help="Whisper model to target (tiny/base/small/medium)")
@click.option("--output", default=str(config.DEFAULT_UAP_PATH), help="Output .wav path")
@click.option("--snr", default=config.DEFAULT_SNR_DB, type=float, help="Target SNR in dB")
@click.option("--steps", default=config.PGD_STEPS, type=int, help="PGD optimization steps")
@click.option("--duration", default=config.UAP_DURATION_SEC, type=float, help="Perturbation duration in seconds")
@click.option("--samples", default=20, type=int, help="Number of training speech samples")
def generate(model, output, snr, steps, duration, samples):
    """Generate a Universal Adversarial Perturbation targeting Whisper."""
    from .generate_uap import generate_uap

    click.echo(f"Generating UAP targeting Whisper-{model}...")
    click.echo(f"  SNR: {snr} dB, Steps: {steps}, Duration: {duration}s")

    path = generate_uap(
        model_name=model,
        output_path=output,
        snr_db=snr,
        num_steps=steps,
        duration_sec=duration,
        num_samples=samples,
        verbose=True,
    )
    click.echo(f"\nDone! UAP saved to: {path}")


@main.command()
@click.option("--uap", default=str(config.DEFAULT_UAP_PATH), help="Path to UAP .wav file")
@click.option("--snr", default=config.DEFAULT_SNR_DB, type=float, help="Perturbation strength (SNR in dB)")
@click.option("--input-device", default=None, type=int, help="Input device index (default: system mic)")
@click.option("--output-device", default=None, type=int, help="Output device index (default: BlackHole)")
def start(uap, snr, input_device, output_device):
    """Start the real-time audio mixer."""
    from .audio_mixer import AudioMixer

    try:
        mixer = AudioMixer(
            uap_path=uap,
            snr_db=snr,
            input_device=input_device,
            output_device=output_device,
        )
    except (RuntimeError, FileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    mixer.start()
    status = mixer.get_status()

    click.echo("TranscriptBlocker active!")
    click.echo(f"  Input:  {status['input_device']}")
    click.echo(f"  Output: {status['output_device']}")
    click.echo(f"  SNR:    {status['snr_db']} dB")
    click.echo(f"  Latency: ~{status['latency_ms']:.0f}ms")
    click.echo("\nPress Ctrl+C to stop.")

    # Handle graceful shutdown
    def shutdown(signum, frame):
        click.echo("\nStopping...")
        mixer.stop()
        click.echo("Stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep main thread alive
    while mixer.is_running:
        time.sleep(0.5)


@main.command()
def devices():
    """List available audio devices."""
    from .audio_devices import print_devices, find_blackhole

    print_devices()

    bh = find_blackhole()
    if bh is None:
        click.echo("\nBlackHole not detected! Install with:")
        click.echo("  brew install blackhole-2ch")


@main.command("set-strength")
@click.argument("snr_db", type=float)
def set_strength(snr_db):
    """Set perturbation strength (SNR in dB). Higher = more subtle, lower = more aggressive."""
    if snr_db < 10:
        click.echo("Warning: SNR below 10 dB will likely be audible to humans.", err=True)
    elif snr_db > 50:
        click.echo("Warning: SNR above 50 dB may not be effective against ASR.", err=True)

    click.echo(f"SNR set to {snr_db} dB")
    click.echo("Note: This only takes effect when starting a new session.")
    click.echo(f"  Run: transcriptblocker start --snr {snr_db}")


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model", default=config.DEFAULT_WHISPER_MODEL, help="Whisper model for transcription")
def test(audio_file, model):
    """Test transcription of an audio file (to verify effectiveness)."""
    import whisper
    click.echo(f"Transcribing {audio_file} with Whisper-{model}...")

    whisper_model = whisper.load_model(model)
    result = whisper_model.transcribe(audio_file)

    click.echo(f"\nTranscription:")
    click.echo(result["text"])


if __name__ == "__main__":
    main()
