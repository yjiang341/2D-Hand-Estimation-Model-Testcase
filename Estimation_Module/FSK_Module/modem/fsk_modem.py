from __future__ import annotations

from dataclasses import dataclass
import math
import wave
from typing import Dict, Iterable, List, Tuple

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


def read_wav_pcm16(path: str) -> Tuple[np.ndarray, int]:
    """Read mono/stereo PCM16 WAV and return float waveform in [-1, 1] plus sample rate."""
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sampwidth != 2:
        raise ValueError(f"Only PCM16 WAV is supported, got sampwidth={sampwidth} bytes")

    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif channels != 1:
        raise ValueError(f"Only mono/stereo WAV is supported, got channels={channels}")

    waveform = (pcm.astype(np.float32) / 32767.0).clip(-1.0, 1.0)
    return waveform, sample_rate


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert MSB-first bit array to bytes; trailing incomplete bits are dropped."""
    if bits.size == 0:
        return b""

    usable_bits = (bits.size // 8) * 8
    if usable_bits == 0:
        return b""

    out = bytearray(usable_bits // 8)
    read_idx = 0
    write_idx = 0
    while read_idx < usable_bits:
        byte_val = 0
        for _ in range(8):
            byte_val = (byte_val << 1) | int(bits[read_idx])
            read_idx += 1
        out[write_idx] = byte_val
        write_idx += 1
    return bytes(out)


def _goertzel_power(samples: np.ndarray, target_freq: float, sample_rate: int) -> float:
    """Estimate tone power at target_freq using Goertzel algorithm."""
    n = len(samples)
    if n == 0:
        return 0.0

    omega = 2.0 * math.pi * target_freq / sample_rate
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0

    for sample in samples:
        s = float(sample) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return max(power, 0.0)


def demodulate_bits_fsk(
    waveform: np.ndarray,
    config: FSKConfig,
    sample_offset: int = 0,
) -> np.ndarray:
    """
    Demodulate BFSK waveform into MSB-style bit stream.

    Decision rule: compare per-symbol Goertzel power at freq0 and freq1.
    """
    sps = config.samples_per_symbol
    if sample_offset < 0:
        raise ValueError("sample_offset must be >= 0")
    if sample_offset >= waveform.size:
        return np.zeros(0, dtype=np.uint8)

    usable = (waveform.size - sample_offset) // sps
    if usable <= 0:
        return np.zeros(0, dtype=np.uint8)

    bits = np.empty(usable, dtype=np.uint8)
    read_idx = sample_offset

    for i in range(usable):
        symbol = waveform[read_idx : read_idx + sps]
        p0 = _goertzel_power(symbol, config.freq0_hz, config.sample_rate)
        p1 = _goertzel_power(symbol, config.freq1_hz, config.sample_rate)
        bits[i] = 1 if p1 >= p0 else 0
        read_idx += sps

    return bits


def _count_preamble_hits(bits: np.ndarray, preamble_bits: np.ndarray) -> int:
    if bits.size < preamble_bits.size or preamble_bits.size == 0:
        return 0

    hits = 0
    max_start = bits.size - preamble_bits.size
    for i in range(max_start + 1):
        if np.array_equal(bits[i : i + preamble_bits.size], preamble_bits):
            hits += 1
    return hits


def find_best_symbol_offset(
    waveform: np.ndarray,
    config: FSKConfig,
    probe_bits: int = 12_000,
) -> Tuple[int, int]:
    """
    Find symbol alignment offset by maximizing preamble hits in probe region.

    Returns (best_offset, preamble_hit_count).
    """
    sps = config.samples_per_symbol
    preamble_bits = bytes_to_bits(config.preamble)
    best_offset = 0
    best_hits = -1

    for offset in range(sps):
        bits = demodulate_bits_fsk(waveform, config, sample_offset=offset)
        if bits.size > probe_bits:
            bits = bits[:probe_bits]
        hits = _count_preamble_hits(bits, preamble_bits)
        if hits > best_hits:
            best_hits = hits
            best_offset = offset

    return best_offset, max(best_hits, 0)


def extract_packets_from_demod_bits(
    bits: np.ndarray,
    config: FSKConfig,
    packet_size: int,
) -> List[bytes]:
    """
    Detect preamble boundaries in bitstream and extract following packet bytes.
    """
    preamble_bits = bytes_to_bits(config.preamble)
    packet_bits_len = packet_size * 8

    packets: List[bytes] = []
    i = 0
    limit = bits.size - preamble_bits.size

    while i <= limit:
        if np.array_equal(bits[i : i + preamble_bits.size], preamble_bits):
            start = i + preamble_bits.size
            end = start + packet_bits_len
            if end > bits.size:
                break

            packet = bits_to_bytes(bits[start:end])
            if len(packet) == packet_size:
                packets.append(packet)

            i = end
            continue

        i += 1

    return packets


def demodulate_packet_stream(
    waveform: np.ndarray,
    config: FSKConfig,
    packet_size: int,
    auto_align: bool = True,
) -> Tuple[List[bytes], Dict[str, int]]:
    """
    Full offline receiver path:
    1) optional symbol alignment search
    2) BFSK demod to bitstream
    3) preamble-based packet extraction
    """
    if auto_align:
        best_offset, preamble_hits = find_best_symbol_offset(waveform, config)
    else:
        best_offset, preamble_hits = 0, 0

    bits = demodulate_bits_fsk(waveform, config, sample_offset=best_offset)
    packets = extract_packets_from_demod_bits(bits, config, packet_size)

    stats = {
        "sample_offset": best_offset,
        "preamble_hits_probe": preamble_hits,
        "demod_bits": int(bits.size),
        "extracted_packets": len(packets),
    }
    return packets, stats
