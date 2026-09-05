from __future__ import annotations

import argparse
import math
import os
from glob import glob

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a contact sheet from a FoundHand iphone_video sequence."
    )
    p.add_argument("--sequence-dir", required=True)
    p.add_argument("--out", default="reference_contact_sheet.jpg")
    p.add_argument("--max-frames", type=int, default=36)
    p.add_argument("--thumb-size", type=int, default=180)
    p.add_argument("--cols", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob(os.path.join(args.sequence_dir, ext)))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No image frames found in {args.sequence_dir}")

    if len(paths) > args.max_frames:
        idxs = np.linspace(0, len(paths) - 1, args.max_frames).round().astype(int)
        paths = [paths[i] for i in idxs]

    cols = max(1, args.cols)
    rows = math.ceil(len(paths) / cols)
    tile_w = tile_h = args.thumb_size
    label_h = 28

    sheet = np.full(
        (rows * (tile_h + label_h), cols * tile_w, 3),
        245,
        dtype=np.uint8,
    )

    for n, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(tile_w / w, tile_h / h)
        resized = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        rh, rw = resized.shape[:2]
        y0 = (tile_h - rh) // 2
        x0 = (tile_w - rw) // 2
        canvas[y0:y0 + rh, x0:x0 + rw] = resized

        r = n // cols
        c = n % cols
        sy = r * (tile_h + label_h)
        sx = c * tile_w
        sheet[sy:sy + tile_h, sx:sx + tile_w] = canvas

        frame_name = os.path.splitext(os.path.basename(path))[0]
        cv2.putText(
            sheet,
            frame_name,
            (sx + 6, sy + tile_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not cv2.imwrite(args.out, sheet):
        raise RuntimeError(f"Failed to write {args.out}")

    print(f"Sequence frames found : {len(glob(os.path.join(args.sequence_dir, '*')))}")
    print(f"Frames shown          : {len(paths)}")
    print(f"Saved contact sheet   : {args.out}")


if __name__ == "__main__":
    main()
