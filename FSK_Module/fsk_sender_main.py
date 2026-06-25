from __future__ import annotations

import argparse
import os
import time
from typing import List

from FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream, write_wav_pcm16
from Pose_PacketUp.pose_codec import QuantizedHand
from Pose_PacketUp.pose_packet import PACKET_SIZE, encode_packet


def _build_demo_hand(fill_byte: int) -> QuantizedHand:
    return QuantizedHand(bytes([fill_byte] * 42))


def generate_demo_packets(frame_count: int, fps: int) -> List[bytes]:
    """Create deterministic packet vectors for offline FSK sender validation."""
    packets: List[bytes] = []
    frame_period_ms = int(1000 / fps)
    timestamp_ms = int(time.time() * 1000)

    for frame_id in range(frame_count):
        # Alternate hand patterns so resulting bitstream has known variability.
        if frame_id % 3 == 0:
            hands = [_build_demo_hand(0x00)]
        elif frame_id % 3 == 1:
            hands = [_build_demo_hand(0x7F), _build_demo_hand(0x33)]
        else:
            hands = []

        packets.append(
            encode_packet(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms + frame_id * frame_period_ms,
                hands=hands,
            )
        )

    return packets


def load_packets_from_bin(path: str) -> List[bytes]:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) % PACKET_SIZE != 0:
        raise ValueError(
            f"Input packet file size ({len(data)} bytes) is not a multiple of PACKET_SIZE={PACKET_SIZE}."
        )

    packets: List[bytes] = []
    for offset in range(0, len(data), PACKET_SIZE):
        packets.append(data[offset : offset + PACKET_SIZE])
    return packets


def save_packet_stream(path: str, packets: List[bytes]) -> None:
    with open(path, "wb") as f:
        for packet in packets:
            f.write(packet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.1 offline FSK sender: packet stream -> WAV modulation"
    )
    parser.add_argument("--packet-bin", type=str, default="", help="Optional binary file of concatenated 104-byte packets")
    parser.add_argument("--frames", type=int, default=45, help="Number of demo frames if --packet-bin is not provided")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate used for demo timestamps")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--symbol-rate", type=int, default=1_200)
    parser.add_argument("--freq0", type=float, default=1_200.0, help="FSK frequency for bit 0")
    parser.add_argument("--freq1", type=float, default=2_200.0, help="FSK frequency for bit 1")
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--silence-ms", type=int, default=3, help="Silence between framed packets")
    parser.add_argument("--out-wav", type=str, default="logs/phase2_1_sender.wav")
    parser.add_argument("--out-packets", type=str, default="logs/phase2_1_packets.bin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_wav) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_packets) or ".", exist_ok=True)

    if args.packet_bin:
        packets = load_packets_from_bin(args.packet_bin)
        packet_source = f"Loaded from {args.packet_bin}"
    else:
        packets = generate_demo_packets(frame_count=args.frames, fps=args.fps)
        packet_source = f"Generated demo packets ({args.frames} frames)"

    config = FSKConfig(
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        freq0_hz=args.freq0,
        freq1_hz=args.freq1,
        amplitude=args.amplitude,
        inter_frame_silence_ms=args.silence_ms,
    )

    waveform = modulate_packet_stream(packets, config)
    write_wav_pcm16(args.out_wav, waveform, config.sample_rate)
    save_packet_stream(args.out_packets, packets)

    frame_bytes = len(packets[0]) if packets else 0
    print("=== Phase 2.1 FSK Sender Output ===")
    print(f"Packet source       : {packet_source}")
    print(f"Packets count       : {len(packets)}")
    print(f"Bytes per packet    : {frame_bytes}")
    print(f"Total packet bytes  : {len(packets) * frame_bytes}")
    print(f"Sample rate         : {config.sample_rate}")
    print(f"Symbol rate         : {config.symbol_rate}")
    print(f"Samples per symbol  : {config.samples_per_symbol}")
    print(f"FSK freq0/freq1     : {config.freq0_hz}/{config.freq1_hz} Hz")
    print(f"WAV samples         : {len(waveform)}")
    print(f"WAV duration        : {len(waveform) / config.sample_rate:.3f} s")
    print(f"WAV output          : {args.out_wav}")
    print(f"Packet output       : {args.out_packets}")


if __name__ == "__main__":
    main()
