from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class AudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


class AudioBackendError(RuntimeError):
    pass


def _sd() -> Any:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise AudioBackendError(
            "sounddevice is required for meeting bridge audio I/O. Install with: pip install sounddevice"
        ) from exc
    return sd


def list_audio_devices() -> List[AudioDeviceInfo]:
    sd = _sd()
    devices = sd.query_devices()
    result: List[AudioDeviceInfo] = []
    for idx, dev in enumerate(devices):
        result.append(
            AudioDeviceInfo(
                index=idx,
                name=str(dev.get("name", "")),
                max_input_channels=int(dev.get("max_input_channels", 0)),
                max_output_channels=int(dev.get("max_output_channels", 0)),
                default_samplerate=float(dev.get("default_samplerate", 0.0)),
            )
        )
    return result


def open_output_stream(sample_rate: int, device: str | int | None = None):
    sd = _sd()
    return sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=0,
    )


def open_input_stream(sample_rate: int, blocksize: int, callback, device: str | int | None = None):
    sd = _sd()
    return sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=blocksize,
        callback=callback,
    )
