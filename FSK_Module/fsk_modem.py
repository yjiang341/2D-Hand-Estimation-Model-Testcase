from __future__ import annotations

from dataclasses import dataclass
import math
import wave
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class FSKConfig:
    """Configuration for deterministic BFSK modulation."""

    sample_rate: int = 48_000
    symbol_rate: int = 1_200
    freq0_hz: float = 1_200.0
    freq1_hz: float = 2_200.0
    amplitude: float = 0.8
    preamble: bytes = b"\x55\x55\x55\xD5"
    inter_frame_silence_ms: int = 3

    @property
    def samples_per_symbol(self) -> int:
        if self.sample_rate % self.symbol_rate != 0:
            raise ValueError(
                "sample_rate must be divisible by symbol_rate for deterministic framing"
            )
        return self.sample_rate // self.symbol_rate


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to MSB-first bit array of uint8 values {0,1}."""
    if not data:
        return np.zeros(0, dtype=np.uint8)

    out = np.empty(len(data) * 8, dtype=np.uint8)
    write_idx = 0
    for byte in data:
        for bit_shift in range(7, -1, -1):
            out[write_idx] = (byte >> bit_shift) & 1
            write_idx += 1
    return out


def frame_packet_bytes(packet_bytes: bytes, config: FSKConfig) -> bytes:
    """Prefix payload with fixed preamble for future receiver synchronization."""
    return config.preamble + packet_bytes


def frame_packet_stream(packets: Iterable[bytes], config: FSKConfig) -> List[bytes]:
    return [frame_packet_bytes(packet, config) for packet in packets]


def _silence_samples(config: FSKConfig) -> np.ndarray:
    n = int(config.sample_rate * (config.inter_frame_silence_ms / 1000.0))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.zeros(n, dtype=np.float32)


def modulate_bits_fsk(bits: np.ndarray, config: FSKConfig) -> np.ndarray:
    """Generate continuous-phase BFSK waveform from bit array."""
    if bits.dtype != np.uint8:
        bits = bits.astype(np.uint8, copy=False)

    sps = config.samples_per_symbol
    if bits.size == 0:
        return np.zeros(0, dtype=np.float32)

    total_samples = bits.size * sps
    waveform = np.empty(total_samples, dtype=np.float32)

    phase = 0.0
    dt = 1.0 / config.sample_rate
    write_idx = 0

    for bit in bits:
        freq = config.freq1_hz if bit else config.freq0_hz
        phase_step = 2.0 * math.pi * freq * dt
        for _ in range(sps):
            waveform[write_idx] = config.amplitude * math.sin(phase)
            phase += phase_step
            if phase >= 2.0 * math.pi:
                phase -= 2.0 * math.pi
            write_idx += 1

    return waveform


def modulate_packet_stream(packets: Iterable[bytes], config: FSKConfig) -> np.ndarray:
    """
    Modulate packets to one mono waveform:
    [preamble + packet][silence][preamble + packet][silence]...
    """
    framed_packets = frame_packet_stream(packets, config)
    if not framed_packets:
        return np.zeros(0, dtype=np.float32)

    chunks: List[np.ndarray] = []
    silence = _silence_samples(config)

    for packet in framed_packets:
        bits = bytes_to_bits(packet)
        chunks.append(modulate_bits_fsk(bits, config))
        if silence.size:
            chunks.append(silence)

    return np.concatenate(chunks).astype(np.float32, copy=False)


def float_wave_to_pcm16(waveform: np.ndarray) -> np.ndarray:
    """Convert float32 [-1,1] waveform to signed int16."""
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def write_wav_pcm16(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    pcm = float_wave_to_pcm16(waveform)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
