"""macOS audio device discovery and selection."""

import sounddevice as sd


def list_devices() -> list[dict]:
    """Return all audio devices with their index, name, and channel counts."""
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        result.append({
            "index": i,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        })
    return result


def find_blackhole() -> int | None:
    """Find the BlackHole 2ch virtual audio device index."""
    for dev in list_devices():
        if "blackhole" in dev["name"].lower():
            return dev["index"]
    return None


def find_default_mic() -> int | None:
    """Find the default input device index."""
    try:
        info = sd.query_devices(kind="input")
        # query_devices with kind= returns the default device dict
        # We need to find its index
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["name"] == info["name"] and dev["max_input_channels"] > 0:
                return i
        return None
    except sd.PortAudioError:
        return None


def get_device_name(index: int) -> str:
    """Get device name by index."""
    return sd.query_devices(index)["name"]


def validate_devices(input_device: int, output_device: int) -> None:
    """Validate that input and output devices exist and have the right channel counts."""
    devices = sd.query_devices()

    if input_device >= len(devices):
        raise ValueError(f"Input device index {input_device} does not exist")
    if output_device >= len(devices):
        raise ValueError(f"Output device index {output_device} does not exist")

    inp = devices[input_device]
    out = devices[output_device]

    if inp["max_input_channels"] < 1:
        raise ValueError(f"Device '{inp['name']}' has no input channels")
    if out["max_output_channels"] < 1:
        raise ValueError(f"Device '{out['name']}' has no output channels")


def print_devices() -> None:
    """Print a formatted table of audio devices."""
    devices = list_devices()
    blackhole_idx = find_blackhole()
    default_mic = find_default_mic()

    print(f"{'Idx':>3}  {'Name':<40}  {'In':>3}  {'Out':>3}  {'Rate':>7}  Notes")
    print("-" * 80)
    for dev in devices:
        notes = []
        if dev["index"] == default_mic:
            notes.append("default mic")
        if dev["index"] == blackhole_idx:
            notes.append("BlackHole")
        note_str = ", ".join(notes)
        print(
            f"{dev['index']:>3}  {dev['name']:<40}  "
            f"{dev['max_input_channels']:>3}  {dev['max_output_channels']:>3}  "
            f"{dev['default_samplerate']:>7.0f}  {note_str}"
        )
