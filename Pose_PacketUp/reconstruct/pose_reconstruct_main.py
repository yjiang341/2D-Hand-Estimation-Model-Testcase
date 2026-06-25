from __future__ import annotations

import argparse
import os

import numpy as np

from Pose_PacketUp.pose_reconstruct import (
    decode_packet_bytes_stream,
    ema_smooth_frames,
    enforce_sanity_constraints,
    frames_to_numpy,
    load_packet_bin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose reconstruction: recovered packet bin -> normalized, constrained, smoothed keypoints"
    )
    parser.add_argument("--in-packets", type=str, default="logs/recovered_packets.bin")
    parser.add_argument("--out-npz", type=str, default="logs/pose_stream.npz")
    parser.add_argument("--alpha", type=float, default=0.65, help="EMA alpha in (0,1]")
    parser.add_argument("--max-step", type=float, default=0.20, help="Per-joint max normalized movement per frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)

    raw_data = load_packet_bin(args.in_packets)
    decoded_frames = decode_packet_bytes_stream(raw_data)

    constrained = enforce_sanity_constraints(decoded_frames, max_step=args.max_step)
    smoothed = ema_smooth_frames(constrained, alpha=args.alpha)

    frame_ids, timestamps_ms, hand_present, hand_xy = frames_to_numpy(smoothed)

    np.savez_compressed(
        args.out_npz,
        frame_ids=frame_ids,
        timestamps_ms=timestamps_ms,
        hand_present=hand_present,
        hand_xy=hand_xy,
        alpha=np.array([args.alpha], dtype=np.float32),
        max_step=np.array([args.max_step], dtype=np.float32),
    )

    print("=== Pose Reconstruction Output ===")
    print(f"Input packet bin      : {args.in_packets}")
    print(f"Decoded frames        : {len(decoded_frames)}")
    print(f"EMA alpha             : {args.alpha}")
    print(f"Sanity max_step       : {args.max_step}")
    print(f"Output NPZ            : {args.out_npz}")
    if len(decoded_frames) > 0:
        print(f"Frame id range        : {frame_ids[0]} .. {frame_ids[-1]}")
        print(f"Hands present samples : {int(hand_present.sum())} hand-slots across stream")


if __name__ == "__main__":
    main()
