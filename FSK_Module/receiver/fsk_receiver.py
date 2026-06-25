from __future__ import annotations

from dataclasses import dataclass
import wave
from typing import List

import numpy as np

from FSK_Module.fsk_modem import FSKConfig, bytes_to_bits, modulate_bits_fsk
from Pose_PacketUp.pose_packet import PACKET_SIZE, PacketDecodeError, PosePacket, decode_packet


@dataclass
class ReceiverReport:
    preamble_candidates: int
    attempted_frames: int
    valid_frames: int
    rejected_frames: int
    valid_packets: List[PosePacket]
    valid_packet_bytes: List[bytes]


def read_wav_pcm16_mono(path: str) -> tuple[np.ndarray, int]:
    """Read mono PCM16 WAV and return normalized float32 waveform in [-1,1]."""
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels == 1:
        return samples, sample_rate
    if channels == 2:
        # Mix stereo down to mono.
        samples = samples.reshape(-1, 2).mean(axis=1)
        return samples.astype(np.float32, copy=False), sample_rate

    raise ValueError(f"Unsupported channel count: {channels}")


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        raise ValueError(f"Bit length must be a multiple of 8, got {bits.size}")

    out = bytearray(bits.size // 8)
    write_idx = 0
    for i in range(0, bits.size, 8):
        byte_val = 0
        for bit in bits[i : i + 8]:
            byte_val = (byte_val << 1) | int(bit)
        out[write_idx] = byte_val
        write_idx += 1
    return bytes(out)


def _symbol_refs(config: FSKConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sps = config.samples_per_symbol
    n = np.arange(sps, dtype=np.float32)
    w0 = 2.0 * np.pi * config.freq0_hz * n / config.sample_rate
    w1 = 2.0 * np.pi * config.freq1_hz * n / config.sample_rate
    return np.cos(w0), np.sin(w0), np.cos(w1), np.sin(w1)


def _demod_packet_bytes(packet_wave: np.ndarray, config: FSKConfig) -> bytes:
    packet_bits = PACKET_SIZE * 8
    sps = config.samples_per_symbol
    needed = packet_bits * sps
    if packet_wave.size < needed:
        raise ValueError("Insufficient samples to demodulate one full packet")

    cos0, sin0, cos1, sin1 = _symbol_refs(config)

    bits = np.zeros(packet_bits, dtype=np.uint8)
    for i in range(packet_bits):
        start = i * sps
        end = start + sps
        chunk = packet_wave[start:end]

        i0 = float(np.dot(chunk, cos0))
        q0 = float(np.dot(chunk, sin0))
        e0 = i0 * i0 + q0 * q0

        i1 = float(np.dot(chunk, cos1))
        q1 = float(np.dot(chunk, sin1))
        e1 = i1 * i1 + q1 * q1

        bits[i] = 1 if e1 > e0 else 0

    return _bits_to_bytes(bits)


def _detect_preamble_positions(
    waveform: np.ndarray,
    config: FSKConfig,
    detection_threshold: float,
) -> List[int]:
    """
    Detect likely preamble start indices using matched filtering.

    Returns sample indices sorted in ascending order.
    """
    preamble_bits = bytes_to_bits(config.preamble)
    preamble_wave = modulate_bits_fsk(preamble_bits, config)

    if waveform.size < preamble_wave.size:
        return []

    corr = np.correlate(waveform, preamble_wave, mode="valid")
    if corr.size == 0:
        return []

    max_corr = float(np.max(corr))
    if max_corr <= 0.0:
        return []

    threshold = max_corr * detection_threshold

    packet_samples = PACKET_SIZE * 8 * config.samples_per_symbol
    silence_samples = int(config.sample_rate * (config.inter_frame_silence_ms / 1000.0))
    frame_span = preamble_wave.size + packet_samples + silence_samples
    min_distance = max(1, int(frame_span * 0.7))

    candidate_indices = np.where(corr >= threshold)[0]
    if candidate_indices.size == 0:
        return []

    selected: List[int] = []
    # Greedy non-max suppression on sorted candidates by correlation strength.
    for idx in candidate_indices[np.argsort(corr[candidate_indices])[::-1]]:
        idx_int = int(idx)
        if all(abs(idx_int - kept) >= min_distance for kept in selected):
            selected.append(idx_int)

    selected.sort()
    return selected


def recover_packets_from_waveform(
    waveform: np.ndarray,
    config: FSKConfig,
    detection_threshold: float = 0.55,
) -> ReceiverReport:
    """
    Recover Pose_PacketUp packets from BFSK waveform.

    Steps:
    1) Detect preamble positions.
    2) For each candidate frame, demodulate exactly 104 bytes.
    3) Validate with pose packet decoder (CRC + structure).
    4) Keep valid packets, drop corrupted ones.
    """
    if not (0.0 < detection_threshold <= 1.0):
        raise ValueError("detection_threshold must be in (0, 1]")

    preamble_positions = _detect_preamble_positions(
        waveform=waveform,
        config=config,
        detection_threshold=detection_threshold,
    )

    preamble_samples = len(config.preamble) * 8 * config.samples_per_symbol
    packet_samples = PACKET_SIZE * 8 * config.samples_per_symbol

    attempted = 0
    valid = 0
    rejected = 0
    valid_packets: List[PosePacket] = []
    valid_packet_bytes: List[bytes] = []

    for preamble_start in preamble_positions:
        packet_start = preamble_start + preamble_samples
        packet_end = packet_start + packet_samples
        if packet_end > waveform.size:
            continue

        attempted += 1
        raw_packet = _demod_packet_bytes(waveform[packet_start:packet_end], config)

        try:
            decoded = decode_packet(raw_packet)
        except PacketDecodeError:
            rejected += 1
            continue

        valid += 1
        valid_packets.append(decoded)
        valid_packet_bytes.append(raw_packet)

    return ReceiverReport(
        preamble_candidates=len(preamble_positions),
        attempted_frames=attempted,
        valid_frames=valid,
        rejected_frames=rejected,
        valid_packets=valid_packets,
        valid_packet_bytes=valid_packet_bytes,
    )


def recover_packets_from_wav(
    wav_path: str,
    config: FSKConfig,
    detection_threshold: float = 0.55,
) -> ReceiverReport:
    waveform, sample_rate = read_wav_pcm16_mono(wav_path)
    if sample_rate != config.sample_rate:
        raise ValueError(
            f"WAV sample rate {sample_rate} does not match config.sample_rate {config.sample_rate}"
        )
    return recover_packets_from_waveform(waveform, config, detection_threshold)
