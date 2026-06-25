from __future__ import annotations

import os
import tempfile

import numpy as np

from Pose_PacketUp.pose_render import RenderConfig, load_pose_stream_npz, render_mp4_from_npz, render_skeleton_frame


def _assert(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)


def _write_dummy_npz(path: str, frames: int = 4) -> None:
    frame_ids = np.arange(frames, dtype=np.int64)
    timestamps_ms = (1000 + np.arange(frames) * 66).astype(np.int64)
    hand_present = np.zeros((frames, 2), dtype=np.uint8)
    hand_xy = np.zeros((frames, 2, 21, 2), dtype=np.float32)

    # Fill one hand with simple diagonal motion.
    for i in range(frames):
        hand_present[i, 0] = 1
        val = min(1.0, i / max(1, frames - 1))
        hand_xy[i, 0, :, 0] = val
        hand_xy[i, 0, :, 1] = val

    np.savez_compressed(
        path,
        frame_ids=frame_ids,
        timestamps_ms=timestamps_ms,
        hand_present=hand_present,
        hand_xy=hand_xy,
    )


def test_load_and_render_frame() -> None:
    with tempfile.TemporaryDirectory() as td:
        npz_path = os.path.join(td, "stream.npz")
        _write_dummy_npz(npz_path)

        frame_ids, timestamps_ms, hand_present, hand_xy = load_pose_stream_npz(npz_path)
        cfg = RenderConfig(width=640, height=360, fps=15.0)
        img = render_skeleton_frame(
            hand_present_row=hand_present[0],
            hand_xy_row=hand_xy[0],
            frame_id=int(frame_ids[0]),
            timestamp_ms=int(timestamps_ms[0]),
            config=cfg,
        )

        _assert(img.shape == (360, 640, 3), "rendered frame shape")
        _assert(img.dtype == np.uint8, "rendered frame dtype")
        _assert(int(img.sum()) > 0, "rendered frame has non-black drawings")


def test_render_mp4() -> None:
    with tempfile.TemporaryDirectory() as td:
        npz_path = os.path.join(td, "stream.npz")
        mp4_path = os.path.join(td, "out.mp4")
        _write_dummy_npz(npz_path, frames=6)

        cfg = RenderConfig(width=640, height=360, fps=15.0)
        rendered = render_mp4_from_npz(npz_path, mp4_path, cfg)

        _assert(rendered == 6, "rendered frame count")
        _assert(os.path.exists(mp4_path), "mp4 file created")
        _assert(os.path.getsize(mp4_path) > 0, "mp4 file non-empty")


if __name__ == "__main__":
    test_load_and_render_frame()
    test_render_mp4()
    print("All renderer tests passed.")
