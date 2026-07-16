from __future__ import annotations

import re
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


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_device_identifier(device: str | int | None, kind: str) -> str | int | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device

    s = str(device).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)

    sd = _sd()
    raw_devices = sd.query_devices()
    if kind == "output":
        candidates = [(idx, d) for idx, d in enumerate(raw_devices) if int(d.get("max_output_channels", 0)) > 0]
    else:
        candidates = [(idx, d) for idx, d in enumerate(raw_devices) if int(d.get("max_input_channels", 0)) > 0]

    needle = s.lower()
    # 1) exact case-insensitive match
    for idx, dev in candidates:
        name = str(dev.get("name", ""))
        if name.lower() == needle:
            return idx

    # 2) substring match
    for idx, dev in candidates:
        name = str(dev.get("name", ""))
        lname = name.lower()
        if needle in lname or lname in needle:
            return idx

    # 3) normalized fuzzy match (ignore punctuation/spaces)
    normalized_needle = _normalize_name(s)
    for idx, dev in candidates:
        name = str(dev.get("name", ""))
        nname = _normalize_name(name)
        if normalized_needle in nname or nname in normalized_needle:
            return idx

    raise ValueError(
        f"No {kind} device matching '{device}'. Run 'python bridge_main.py --mode list-devices' "
        f"and use the exact name or index."
    )


def open_output_stream(sample_rate: int, device: str | int | None = None, fallback_to_default: bool = False):
    sd = _sd()
    resolved = None
    try:
        resolved = _resolve_device_identifier(device, kind="output")
    except ValueError as exc:
        if not fallback_to_default:
            raise
        print(f"[audio] warning: {exc} Falling back to system default output device.")
        resolved = None

    return sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=resolved,
        blocksize=0,
    )


def open_input_stream(sample_rate: int, blocksize: int, callback, device: str | int | None = None):
    sd = _sd()
    resolved = _resolve_device_identifier(device, kind="input")
    return sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=resolved,
        blocksize=blocksize,
        callback=callback,
    )
