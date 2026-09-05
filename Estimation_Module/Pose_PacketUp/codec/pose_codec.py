from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


LANDMARK_COUNT = 21
COORD_CHANNELS = 2
BYTES_PER_HAND = LANDMARK_COUNT * COORD_CHANNELS


@dataclass(frozen=True)
class QuantizedHand:
    """Compact hand payload with 21 (x, y) uint8 pairs."""

    points_u8: bytes

    def __post_init__(self) -> None:
        if len(self.points_u8) != BYTES_PER_HAND:
            raise ValueError(
                f"Quantized hand payload must be {BYTES_PER_HAND} bytes, "
                f"got {len(self.points_u8)}"
            )


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def quantize_coord_u8(value: float) -> int:
    """Map normalized float [0, 1] to uint8 [0, 255] with clamping."""
    return int(round(clamp01(value) * 255.0))


def dequantize_coord_u8(value: int) -> float:
    """Map uint8 [0, 255] back to normalized float [0, 1]."""
    return max(0, min(255, int(value))) / 255.0


def quantize_hand_landmarks_xy(hand_landmarks: Sequence[object]) -> QuantizedHand:
    """
    Convert MediaPipe hand landmarks (x, y, z) into compact uint8 x/y bytes.

    Input is expected to contain 21 landmarks with `.x` and `.y` attributes.
    """
    if len(hand_landmarks) != LANDMARK_COUNT:
        raise ValueError(f"Expected {LANDMARK_COUNT} landmarks, got {len(hand_landmarks)}")

    payload = bytearray(BYTES_PER_HAND)
    write_idx = 0

    for landmark in hand_landmarks:
        payload[write_idx] = quantize_coord_u8(float(landmark.x))
        payload[write_idx + 1] = quantize_coord_u8(float(landmark.y))
        write_idx += 2

    return QuantizedHand(bytes(payload))


def quantized_hand_to_xy_pairs(hand: QuantizedHand) -> List[Tuple[float, float]]:
    """Decode compact uint8 bytes back to normalized x/y tuples for debugging."""
    points: List[Tuple[float, float]] = []
    for idx in range(0, BYTES_PER_HAND, 2):
        x = dequantize_coord_u8(hand.points_u8[idx])
        y = dequantize_coord_u8(hand.points_u8[idx + 1])
        points.append((x, y))
    return points


def quantize_all_hands(hand_landmarks_list: Iterable[Sequence[object]]) -> List[QuantizedHand]:
    return [quantize_hand_landmarks_xy(hand_landmarks) for hand_landmarks in hand_landmarks_list]
