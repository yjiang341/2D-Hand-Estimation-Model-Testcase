from __future__ import annotations

import os
import tempfile

import numpy as np

from FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream, write_wav_pcm16
from FSK_Module.fsk_receiver import recover_packets_from_wav, recover_packets_from_waveform
from FSK_Module.fsk_sender_main import generate_demo_packets


def _assert(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)


def test_receiver_roundtrip_waveform() -> None:
    cfg = FSKConfig(sample_rate=48_000, symbol_rate=1_200, freq0_hz=1_200.0, freq1_hz=2_200.0)
    packets = generate_demo_packets(frame_count=10, fps=15)
    waveform = modulate_packet_stream(packets, cfg)

    report = recover_packets_from_waveform(waveform, cfg, detection_threshold=0.5)

    _assert(report.valid_frames == len(packets), "all packets recovered from clean waveform")
    _assert(len(report.valid_packet_bytes) == len(packets), "valid packet byte count matches")
    _assert(report.valid_packet_bytes == packets, "recovered packets match transmitted packets")


def test_receiver_roundtrip_wav_file() -> None:
    cfg = FSKConfig()
    packets = generate_demo_packets(frame_count=6, fps=15)
    waveform = modulate_packet_stream(packets, cfg)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tx.wav")
        write_wav_pcm16(wav_path, waveform, cfg.sample_rate)

        report = recover_packets_from_wav(wav_path, cfg, detection_threshold=0.5)

        _assert(report.valid_frames == len(packets), "all packets recovered from wav file")
        _assert(report.valid_packet_bytes == packets, "wav decode matches transmitted packets")


def test_receiver_corruption_drop() -> None:
    cfg = FSKConfig()
    packets = generate_demo_packets(frame_count=8, fps=15)
    waveform = modulate_packet_stream(packets, cfg).copy()

    # Corrupt a middle region strongly; receiver should keep running and drop bad packets.
    start = len(waveform) // 3
    end = min(len(waveform), start + cfg.sample_rate // 2)
    rng = np.random.default_rng(123)
    waveform[start:end] += rng.normal(0.0, 1.5, end - start).astype(np.float32)

    report = recover_packets_from_waveform(waveform, cfg, detection_threshold=0.5)

    _assert(report.attempted_frames >= report.valid_frames, "attempted >= valid")
    _assert(report.rejected_frames >= 0, "rejected count non-negative")
    _assert(report.valid_frames > 0, "some frames still recovered after corruption")


if __name__ == "__main__":
    test_receiver_roundtrip_waveform()
    test_receiver_roundtrip_wav_file()
    test_receiver_corruption_drop()
    print("All Phase 2.2 FSK receiver tests passed.")
