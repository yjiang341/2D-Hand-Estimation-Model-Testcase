from __future__ import annotations

import os
import tempfile

import numpy as np

from Pose_PacketUp.pose_packet import encode_packet
from Pose_PacketUp.pose_codec import QuantizedHand
from Pose_PacketUp.pose_reconstruct import (
    decode_packet_bytes_stream,
    ema_smooth_frames,
    enforce_sanity_constraints,
    frames_to_numpy,
)


def _assert(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)


def _hand(fill: int) -> QuantizedHand:
    return QuantizedHand(bytes([fill] * 42))


def test_decode_stream() -> None:
    packets = [
        encode_packet(frame_id=0, timestamp_ms=1000, hands=[_hand(0)]),
        encode_packet(frame_id=1, timestamp_ms=1066, hands=[_hand(255)]),
    ]
    frames = decode_packet_bytes_stream(b"".join(packets))

    _assert(len(frames) == 2, "decoded frame count")
    _assert(frames[0].frame_id == 0, "frame 0 id")
    _assert(frames[1].frame_id == 1, "frame 1 id")
    _assert(frames[0].hands[0] is not None, "hand exists in frame 0")
    _assert(frames[1].hands[0] is not None, "hand exists in frame 1")
    _assert(np.isclose(frames[0].hands[0][0, 0], 0.0), "dequantized min")
    _assert(np.isclose(frames[1].hands[0][0, 0], 1.0), "dequantized max")


def test_sanity_max_step() -> None:
    packets = [
        encode_packet(frame_id=0, timestamp_ms=1000, hands=[_hand(0)]),
        encode_packet(frame_id=1, timestamp_ms=1066, hands=[_hand(255)]),
    ]
    frames = decode_packet_bytes_stream(b"".join(packets))
    constrained = enforce_sanity_constraints(frames, max_step=0.10)

    prev = constrained[0].hands[0]
    cur = constrained[1].hands[0]
    step = np.linalg.norm(cur - prev, axis=1)
    _assert(float(step.max()) <= 0.100001, "max_step constraint applied")


def test_ema_behavior() -> None:
    packets = [
        encode_packet(frame_id=0, timestamp_ms=1000, hands=[_hand(0)]),
        encode_packet(frame_id=1, timestamp_ms=1066, hands=[_hand(255)]),
    ]
    frames = decode_packet_bytes_stream(b"".join(packets))
    # Use a very large max_step so EMA test measures smoothing behavior only.
    constrained = enforce_sanity_constraints(frames, max_step=10.0)
    smoothed = ema_smooth_frames(constrained, alpha=0.5)

    first = smoothed[0].hands[0][0, 0]
    second = smoothed[1].hands[0][0, 0]
    _assert(np.isclose(first, 0.0), "first frame unchanged with no history")
    _assert(0.49 <= second <= 0.51, "second frame EMA midpoint")


def test_frames_to_numpy() -> None:
    packets = [
        encode_packet(frame_id=0, timestamp_ms=1000, hands=[_hand(20)]),
        encode_packet(frame_id=1, timestamp_ms=1066, hands=[]),
    ]
    frames = decode_packet_bytes_stream(b"".join(packets))
    constrained = enforce_sanity_constraints(frames, max_step=1.0)
    smoothed = ema_smooth_frames(constrained, alpha=1.0)

    frame_ids, timestamps_ms, hand_present, hand_xy = frames_to_numpy(smoothed)

    _assert(frame_ids.shape == (2,), "frame_ids shape")
    _assert(timestamps_ms.shape == (2,), "timestamps shape")
    _assert(hand_present.shape == (2, 2), "hand_present shape")
    _assert(hand_xy.shape == (2, 2, 21, 2), "hand_xy shape")
    _assert(hand_present[0, 0] == 1, "first frame first hand present")
    _assert(hand_present[1, 0] == 0, "second frame first hand absent")


if __name__ == "__main__":
    test_decode_stream()
    test_sanity_max_step()
    test_ema_behavior()
    test_frames_to_numpy()
    print("All pose reconstruction tests passed.")
