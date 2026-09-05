from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


HAND_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [0, 17, 18, 19, 20],
    [5, 9, 13, 17],
]


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1280
    height: int = 720
    fps: float = 15.0
    point_radius: int = 4
    line_thickness: int = 2
    left_color_bgr: Tuple[int, int, int] = (0, 255, 255)
    right_color_bgr: Tuple[int, int, int] = (255, 255, 0)
    text_color_bgr: Tuple[int, int, int] = (200, 200, 200)
    background_bgr: Tuple[int, int, int] = (0, 0, 0)


def load_pose_stream_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    required = ["frame_ids", "timestamps_ms", "hand_present", "hand_xy"]
    for key in required:
        if key not in data:
            raise ValueError(f"NPZ is missing required key: {key}")

    frame_ids = data["frame_ids"]
    timestamps_ms = data["timestamps_ms"]
    hand_present = data["hand_present"]
    hand_xy = data["hand_xy"]

    if frame_ids.ndim != 1:
        raise ValueError("frame_ids must be 1D")
    if timestamps_ms.ndim != 1:
        raise ValueError("timestamps_ms must be 1D")
    if hand_present.ndim != 2 or hand_present.shape[1] != 2:
        raise ValueError("hand_present must have shape (F, 2)")
    if hand_xy.ndim != 4 or hand_xy.shape[1:] != (2, 21, 2):
        raise ValueError("hand_xy must have shape (F, 2, 21, 2)")
    if not (len(frame_ids) == len(timestamps_ms) == hand_present.shape[0] == hand_xy.shape[0]):
        raise ValueError("frame count mismatch among frame_ids, timestamps_ms, hand_present, hand_xy")

    return frame_ids, timestamps_ms, hand_present, hand_xy


def _to_pixel_points(hand_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    # hand_xy in [0,1], shape (21,2)
    x = np.clip(hand_xy[:, 0], 0.0, 1.0) * (width - 1)
    y = np.clip(hand_xy[:, 1], 0.0, 1.0) * (height - 1)
    points = np.stack([x, y], axis=1).astype(np.int32)
    return points


def render_skeleton_frame(
    hand_present_row: np.ndarray,
    hand_xy_row: np.ndarray,
    frame_id: int,
    timestamp_ms: int,
    config: RenderConfig,
) -> np.ndarray:
    canvas = np.full((config.height, config.width, 3), config.background_bgr, dtype=np.uint8)

    for slot in range(2):
        if int(hand_present_row[slot]) == 0:
            continue

        points = _to_pixel_points(hand_xy_row[slot], config.width, config.height)
        color = config.left_color_bgr if slot == 0 else config.right_color_bgr

        for path in HAND_CONNECTIONS:
            for i in range(len(path) - 1):
                p0 = tuple(points[path[i]])
                p1 = tuple(points[path[i + 1]])
                cv2.line(canvas, p0, p1, color, config.line_thickness)

        for p in points:
            cv2.circle(canvas, tuple(p), config.point_radius, color, cv2.FILLED)

    cv2.putText(
        canvas,
        f"Frame: {frame_id}  Ts(ms): {timestamp_ms}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        config.text_color_bgr,
        2,
    )
    return canvas


def iter_rendered_frames(npz_path: str, config: RenderConfig) -> Iterator[np.ndarray]:
    frame_ids, timestamps_ms, hand_present, hand_xy = load_pose_stream_npz(npz_path)
    for i in range(len(frame_ids)):
        yield render_skeleton_frame(
            hand_present_row=hand_present[i],
            hand_xy_row=hand_xy[i],
            frame_id=int(frame_ids[i]),
            timestamp_ms=int(timestamps_ms[i]),
            config=config,
        )


def render_mp4_from_npz(npz_path: str, out_mp4: str, config: RenderConfig) -> int:
    frame_ids, timestamps_ms, hand_present, hand_xy = load_pose_stream_npz(npz_path)
    frame_count = len(frame_ids)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, float(config.fps), (config.width, config.height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_mp4}")

    try:
        for i in range(frame_count):
            frame = render_skeleton_frame(
                hand_present_row=hand_present[i],
                hand_xy_row=hand_xy[i],
                frame_id=int(frame_ids[i]),
                timestamp_ms=int(timestamps_ms[i]),
                config=config,
            )
            writer.write(frame)
    finally:
        writer.release()

    return frame_count


def preview_npz_realtime(npz_path: str, config: RenderConfig, loop: bool = False) -> int:
    frame_ids, timestamps_ms, hand_present, hand_xy = load_pose_stream_npz(npz_path)
    frame_count = len(frame_ids)
    if frame_count == 0:
        return 0

    dt_target = 1.0 / max(config.fps, 1e-6)
    shown = 0

    while True:
        for i in range(frame_count):
            start = time.perf_counter()

            frame = render_skeleton_frame(
                hand_present_row=hand_present[i],
                hand_xy_row=hand_xy[i],
                frame_id=int(frame_ids[i]),
                timestamp_ms=int(timestamps_ms[i]),
                config=config,
            )
            cv2.imshow("Skeleton Preview", frame)
            shown += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return shown

            elapsed = time.perf_counter() - start
            remaining = dt_target - elapsed
            if remaining > 0:
                # waitKey takes milliseconds; keep window responsive.
                cv2.waitKey(max(1, int(remaining * 1000)))

        if not loop:
            break

    cv2.destroyAllWindows()
    return shown
