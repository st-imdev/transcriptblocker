"""macOS menu bar application for TranscriptBlocker using rumps."""

import threading
import tempfile
from pathlib import Path

import rumps
import numpy as np
import sounddevice as sd

from . import config
from .audio_devices import list_devices, find_blackhole, find_default_mic, get_device_name
from .audio_mixer import AudioMixer


STRENGTH_PRESETS = {
    "Light (25dB)": 25.0,
    "Normal (20dB)": 20.0,
    "Aggressive (15dB)": 15.0,
}

TITLE_ACTIVE = "\U0001f507 TB"  # muted speaker + TB
TITLE_INACTIVE = "TB"


class TranscriptBlockerApp(rumps.App):
    def __init__(self):
        super().__init__(TITLE_INACTIVE, quit_button=None)

        self._mixer = None
        self._snr_db = config.DEFAULT_SNR_DB
        self._input_device = None
        self._output_device = None

        # --- Build menu ---
        self._toggle_item = rumps.MenuItem("Start Blocking", callback=self._toggle_blocking)

        # Strength submenu
        self._strength_menu = rumps.MenuItem("Strength")
        self._strength_items = {}
        for label, snr in STRENGTH_PRESETS.items():
            item = rumps.MenuItem(label, callback=self._set_strength)
            if snr == self._snr_db:
                item.state = True
            self._strength_items[label] = item
            self._strength_menu.add(item)

        # Devices submenu
        self._devices_menu = rumps.MenuItem("Devices")
        self._build_devices_menu()

        test_item = rumps.MenuItem("Test Effectiveness", callback=self._test_effectiveness)
        quit_item = rumps.MenuItem("Quit", callback=self._quit)

        self.menu = [
            self._toggle_item,
            None,  # separator
            self._strength_menu,
            self._devices_menu,
            None,
            test_item,
            None,
            quit_item,
        ]

    # ---- Devices submenu ----

    def _build_devices_menu(self):
        """Populate the Devices submenu with input and output devices."""
        self._devices_menu.clear()
        self._input_items = {}
        self._output_items = {}

        devices = list_devices()
        default_mic = find_default_mic()
        blackhole_idx = find_blackhole()

        # Auto-select defaults
        if self._input_device is None:
            self._input_device = default_mic
        if self._output_device is None:
            self._output_device = blackhole_idx

        # Input header
        self._devices_menu.add(rumps.MenuItem("-- Input Devices --"))
        for dev in devices:
            if dev["max_input_channels"] > 0:
                label = dev["name"]
                item = rumps.MenuItem(label, callback=self._select_input_device)
                item._device_index = dev["index"]
                if dev["index"] == self._input_device:
                    item.state = True
                self._input_items[dev["index"]] = item
                self._devices_menu.add(item)

        self._devices_menu.add(rumps.MenuItem(""))  # visual gap

        # Output header
        self._devices_menu.add(rumps.MenuItem("-- Output Devices --"))
        for dev in devices:
            if dev["max_output_channels"] > 0:
                label = dev["name"]
                item = rumps.MenuItem(label, callback=self._select_output_device)
                item._device_index = dev["index"]
                if dev["index"] == self._output_device:
                    item.state = True
                self._output_items[dev["index"]] = item
                self._devices_menu.add(item)

    def _select_input_device(self, sender):
        for item in self._input_items.values():
            item.state = False
        sender.state = True
        self._input_device = sender._device_index
        if self._mixer and self._mixer.is_running:
            rumps.notification(
                "TranscriptBlocker",
                "Device Changed",
                "Restart blocking for the new input device to take effect.",
            )

    def _select_output_device(self, sender):
        for item in self._output_items.values():
            item.state = False
        sender.state = True
        self._output_device = sender._device_index
        if self._mixer and self._mixer.is_running:
            rumps.notification(
                "TranscriptBlocker",
                "Device Changed",
                "Restart blocking for the new output device to take effect.",
            )

    # ---- Toggle blocking ----

    def _toggle_blocking(self, sender):
        if self._mixer and self._mixer.is_running:
            self._stop_blocking()
        else:
            self._start_blocking()

    def _start_blocking(self):
        try:
            self._mixer = AudioMixer(
                uap_path=config.DEFAULT_UAP_PATH,
                snr_db=self._snr_db,
                input_device=self._input_device,
                output_device=self._output_device,
            )
            self._mixer.start()
        except FileNotFoundError as e:
            rumps.notification(
                "TranscriptBlocker", "Missing UAP File", str(e),
            )
            return
        except RuntimeError as e:
            rumps.notification(
                "TranscriptBlocker", "Device Error", str(e),
            )
            return
        except Exception as e:
            rumps.notification(
                "TranscriptBlocker", "Error", str(e),
            )
            return

        self.title = TITLE_ACTIVE
        self._toggle_item.title = "Stop Blocking"
        status = self._mixer.get_status()
        rumps.notification(
            "TranscriptBlocker",
            "Blocking Active",
            f"Input: {status['input_device']}\n"
            f"Output: {status['output_device']}\n"
            f"SNR: {status['snr_db']} dB",
        )

    def _stop_blocking(self):
        if self._mixer:
            self._mixer.stop()
        self.title = TITLE_INACTIVE
        self._toggle_item.title = "Start Blocking"
        rumps.notification("TranscriptBlocker", "Stopped", "Blocking is now inactive.")

    # ---- Strength ----

    def _set_strength(self, sender):
        for item in self._strength_items.values():
            item.state = False
        sender.state = True

        self._snr_db = STRENGTH_PRESETS[sender.title]
        if self._mixer and self._mixer.is_running:
            self._mixer.set_snr(self._snr_db)

    # ---- Test effectiveness ----

    def _test_effectiveness(self, _):
        """Record 5 seconds from the output device, transcribe with Whisper, show result."""
        rumps.notification(
            "TranscriptBlocker", "Test Started", "Recording 5 seconds of audio...",
        )

        def _run_test():
            try:
                duration = 5
                sr = config.SAMPLE_RATE
                recording = sd.rec(
                    int(duration * sr),
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    device=self._input_device,
                )
                sd.wait()

                # Save to temp file
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                import scipy.io.wavfile as wavfile

                audio_int16 = (recording[:, 0] * 32767).astype(np.int16)
                wavfile.write(tmp.name, sr, audio_int16)

                # Transcribe
                import whisper

                model = whisper.load_model(config.DEFAULT_WHISPER_MODEL)
                result = model.transcribe(tmp.name)
                text = result["text"].strip()

                if text:
                    rumps.notification(
                        "TranscriptBlocker",
                        "Test Result",
                        f"Whisper heard: \"{text}\"",
                    )
                else:
                    rumps.notification(
                        "TranscriptBlocker",
                        "Test Result",
                        "Whisper transcribed nothing -- blocking is effective!",
                    )

                # Clean up
                Path(tmp.name).unlink(missing_ok=True)

            except Exception as e:
                rumps.notification(
                    "TranscriptBlocker", "Test Failed", str(e),
                )

        # Run in background thread so the UI stays responsive
        thread = threading.Thread(target=_run_test, daemon=True)
        thread.start()

    # ---- Quit ----

    def _quit(self, _):
        if self._mixer and self._mixer.is_running:
            self._mixer.stop()
        rumps.quit_application()


def run():
    """Entry point for the menu bar app."""
    TranscriptBlockerApp().run()


if __name__ == "__main__":
    run()
