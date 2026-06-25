from __future__ import annotations

import argparse
import os

from FSK_Module.fsk_modem import FSKConfig
from FSK_Module.fsk_receiver import recover_packets_from_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline FSK receiver: WAV -> packet stream"
    )
    parser.add_argument("--in-wav", type=str, default="logs/fsk_sender.wav")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--symbol-rate", type=int, default=1_200)
    parser.add_argument("--freq0", type=float, default=1_200.0)
    parser.add_argument("--freq1", type=float, default=2_200.0)
    parser.add_argument("--silence-ms", type=int, default=3)
    parser.add_argument("--detect-threshold", type=float, default=0.55)
    parser.add_argument("--out-recovered", type=str, default="logs/recovered_packets.bin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_recovered) or ".", exist_ok=True)

    config = FSKConfig(
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        freq0_hz=args.freq0,
        freq1_hz=args.freq1,
        inter_frame_silence_ms=args.silence_ms,
    )

    report = recover_packets_from_wav(
        wav_path=args.in_wav,
        config=config,
        detection_threshold=args.detect_threshold,
    )

    with open(args.out_recovered, "wb") as f:
        for packet in report.valid_packet_bytes:
            f.write(packet)

    print("=== FSK Receiver Output ===")
    print(f"Input WAV            : {args.in_wav}")
    print(f"Preamble candidates  : {report.preamble_candidates}")
    print(f"Attempted frames     : {report.attempted_frames}")
    print(f"Valid frames         : {report.valid_frames}")
    print(f"Rejected frames      : {report.rejected_frames}")
    print(f"Recovered packet bin : {args.out_recovered}")

    if report.valid_packets:
        first = report.valid_packets[0].frame_id
        last = report.valid_packets[-1].frame_id
        print(f"Recovered frame_id   : {first} .. {last}")


if __name__ == "__main__":
    main()
