from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Pose_PacketUp.pose_codec import LANDMARK_COUNT, quantized_hand_to_xy_pairs
from Pose_PacketUp.pose_packet import HAND_SLOT_COUNT, PACKET_SIZE, PacketDecodeError, PosePacket, decode_packet


@dataclass
class ReconstructedPoseFrame:
    frame_id: int
    timestamp_ms: int
    hands: List[Optional[np.ndarray]]  # length == HAND_SLOT_COUNT; each shape (21,2)


def _decode_packet_to_frame(packet: PosePacket) -> ReconstructedPoseFrame:
    hands: List[Optional[np.ndarray]] = []
    for hand in packet.hands:
        if hand is None:
            hands.append(None)
            continue
        xy = np.array(quantized_hand_to_xy_pairs(hand), dtype=np.float32)
        hands.append(xy)

    return ReconstructedPoseFrame(
        frame_id=packet.frame_id,
        timestamp_ms=packet.timestamp_ms,
        hands=hands,
    )


def decode_packet_bytes_stream(data: bytes) -> List[ReconstructedPoseFrame]:
    """
    Decode concatenated 104-byte packet stream into normalized hand keypoints.

    Corrupted packets are skipped (same behavior as packet decode stage).
    """
    frames: List[ReconstructedPoseFrame] = []
    offset = 0
    while offset + PACKET_SIZE <= len(data):
        chunk = data[offset : offset + PACKET_SIZE]
        try:
            packet = decode_packet(chunk)
        except PacketDecodeError:
            offset += PACKET_SIZE
            continue

        frames.append(_decode_packet_to_frame(packet))
        offset += PACKET_SIZE

    return frames


def load_packet_bin(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def enforce_sanity_constraints(
    frames: List[ReconstructedPoseFrame],
    max_step: float = 0.20,
) -> List[ReconstructedPoseFrame]:
    """
    Enforce simple pose sanity constraints in normalized space.

    1) Clamp all keypoints to [0, 1].
    2) Limit per-joint motion between consecutive valid frames to max_step.
    """
    if max_step <= 0.0:
        raise ValueError("max_step must be > 0")

    out: List[ReconstructedPoseFrame] = []
    prev_hands: List[Optional[np.ndarray]] = [None] * HAND_SLOT_COUNT

    for frame in frames:
        constrained_hands: List[Optional[np.ndarray]] = []

        for slot in range(HAND_SLOT_COUNT):
            hand = frame.hands[slot]
            if hand is None:
                constrained_hands.append(None)
                prev_hands[slot] = None
                continue

            cur = np.clip(hand.astype(np.float32, copy=True), 0.0, 1.0)
            prev = prev_hands[slot]
            if prev is not None:
                delta = cur - prev
                dist = np.linalg.norm(delta, axis=1, keepdims=True)
                scale = np.minimum(1.0, max_step / np.maximum(dist, 1e-8))
                cur = prev + delta * scale

            cur = np.clip(cur, 0.0, 1.0)
            constrained_hands.append(cur)
            prev_hands[slot] = cur.copy()

        out.append(
            ReconstructedPoseFrame(
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                hands=constrained_hands,
            )
        )

    return out


def ema_smooth_frames(
    frames: List[ReconstructedPoseFrame],
    alpha: float = 0.65,
) -> List[ReconstructedPoseFrame]:
    """
    Apply per-joint EMA smoothing to normalized keypoints.

    smoothed_t = alpha * current + (1 - alpha) * smoothed_{t-1}
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    out: List[ReconstructedPoseFrame] = []
    prev_smoothed: List[Optional[np.ndarray]] = [None] * HAND_SLOT_COUNT

    for frame in frames:
        smoothed_hands: List[Optional[np.ndarray]] = []
        for slot in range(HAND_SLOT_COUNT):
            hand = frame.hands[slot]
            if hand is None:
                smoothed_hands.append(None)
                prev_smoothed[slot] = None
                continue

            cur = hand.astype(np.float32, copy=False)
            prev = prev_smoothed[slot]
            if prev is None:
                smoothed = cur.copy()
            else:
                smoothed = alpha * cur + (1.0 - alpha) * prev

            smoothed = np.clip(smoothed, 0.0, 1.0)
            smoothed_hands.append(smoothed)
            prev_smoothed[slot] = smoothed.copy()

        out.append(
            ReconstructedPoseFrame(
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                hands=smoothed_hands,
            )
        )

    return out


def frames_to_numpy(
    frames: List[ReconstructedPoseFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert frame list into numpy tensors for downstream stages.

    Returns:
      frame_ids:      (F,) int64
      timestamps_ms:  (F,) int64
      hand_present:   (F, 2) uint8
      hand_xy:        (F, 2, 21, 2) float32
    """
    f = len(frames)
    frame_ids = np.zeros((f,), dtype=np.int64)
    timestamps_ms = np.zeros((f,), dtype=np.int64)
    hand_present = np.zeros((f, HAND_SLOT_COUNT), dtype=np.uint8)
    hand_xy = np.zeros((f, HAND_SLOT_COUNT, LANDMARK_COUNT, 2), dtype=np.float32)

    for i, frame in enumerate(frames):
        frame_ids[i] = frame.frame_id
        timestamps_ms[i] = frame.timestamp_ms
        for slot in range(HAND_SLOT_COUNT):
            hand = frame.hands[slot]
            if hand is None:
                continue
            hand_present[i, slot] = 1
            hand_xy[i, slot, :, :] = hand

    return frame_ids, timestamps_ms, hand_present, hand_xy
