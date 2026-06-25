from __future__ import annotations

import os
import tempfile

import numpy as np

from FSK_Module.fsk_modem import FSKConfig, bytes_to_bits, modulate_packet_stream, write_wav_pcm16
from Pose_PacketUp.pose_packet import PACKET_SIZE, encode_packet


def _assert(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)


def test_bits() -> None:
    bits = bytes_to_bits(bytes([0xA5]))
    expected = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    _assert(np.array_equal(bits, expected), "bytes_to_bits MSB-first mapping")


def test_waveform_length() -> None:
    cfg = FSKConfig(sample_rate=48_000, symbol_rate=1_200, inter_frame_silence_ms=5)
    packet = encode_packet(frame_id=1, timestamp_ms=1000, hands=[])

    waveform = modulate_packet_stream([packet], cfg)

    frame_bytes = len(cfg.preamble) + PACKET_SIZE
    bits_per_frame = frame_bytes * 8
    expected = bits_per_frame * cfg.samples_per_symbol + int(cfg.sample_rate * 0.005)

    _assert(len(waveform) == expected, "modulated waveform sample length")


def test_wav_write() -> None:
    cfg = FSKConfig()
    packet = encode_packet(frame_id=2, timestamp_ms=2000, hands=[])
    waveform = modulate_packet_stream([packet], cfg)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sender.wav")
        write_wav_pcm16(path, waveform, cfg.sample_rate)
        _assert(os.path.exists(path), "wav file created")
        _assert(os.path.getsize(path) > 44, "wav file has PCM payload")


if __name__ == "__main__":
    test_bits()
    test_waveform_length()
    test_wav_write()
    print("All FSK sender tests passed.")
