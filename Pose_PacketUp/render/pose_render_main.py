from __future__ import annotations

import argparse
import os

from Pose_PacketUp.pose_render import RenderConfig, preview_npz_realtime, render_mp4_from_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skeleton rendering from reconstructed pose stream"
    )
    parser.add_argument("--in-npz", type=str, default="logs/pose_stream.npz")
    parser.add_argument("--mode", type=str, choices=["mp4", "preview", "both"], default="mp4")
    parser.add_argument("--out-mp4", type=str, default="logs/skeleton.mp4")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--loop", action="store_true", help="Loop indefinitely in preview mode until q/esc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)

    config = RenderConfig(width=args.width, height=args.height, fps=args.fps)

    rendered = 0
    if args.mode in ("mp4", "both"):
        rendered = render_mp4_from_npz(args.in_npz, args.out_mp4, config)
        print("=== MP4 Render Output ===")
        print(f"Input NPZ            : {args.in_npz}")
        print(f"Output MP4           : {args.out_mp4}")
        print(f"Rendered frames      : {rendered}")
        print(f"Canvas/FPS           : {args.width}x{args.height} @ {args.fps}")

    if args.mode in ("preview", "both"):
        shown = preview_npz_realtime(args.in_npz, config, loop=args.loop)
        print("=== Realtime Preview Output ===")
        print(f"Input NPZ            : {args.in_npz}")
        print(f"Shown frames         : {shown}")
        print("Exit key             : q or esc")


if __name__ == "__main__":
    main()
