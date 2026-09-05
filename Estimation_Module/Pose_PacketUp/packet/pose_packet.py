from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass, field
from typing import List, Optional

from Estimation_Module.Pose_PacketUp.pose_codec import BYTES_PER_HAND, QuantizedHand, quantized_hand_to_xy_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────────────
#
# Wire format (all little-endian, fixed 104 bytes per packet):
#
#  Offset  Size  Type    Field
#  ──────  ────  ──────  ───────────────────────────────────────────────────
#    0      2    uint8×2  MAGIC        = 0xA5, 0x4C
#    2      1    uint8    VERSION      = 0x03
#    3      4    uint32   FRAME_ID     monotonically increasing frame counter
#    7      8    uint64   TIMESTAMP_MS wall-clock timestamp in milliseconds
#   15      1    uint8    STATUS_FLAGS
#                                       bit 0 = RIGHT hand present
#                                       bit 1 = LEFT hand present
#                                       bit 2 = RIGHT orientation value (0=PALM, 1=BACK)
#                                       bit 3 = RIGHT orientation valid
#                                       bit 4 = LEFT orientation value  (0=PALM, 1=BACK)
#                                       bit 5 = LEFT orientation valid
#                                       bits 6..7 reserved, must be 0
#   16      2    uint16   PAYLOAD_LEN  always PAYLOAD_SIZE (84); receiver sanity check
#   18      2    uint16   CHECKSUM     CRC-16/CCITT over bytes [0..17] + [20..103]
#                                       (i.e. full packet with checksum bytes zeroed)
#   20     42    uint8×42 HAND_0       slot 0 – 21 (x,y) uint8 pairs; zeros = absent
#   62     42    uint8×42 HAND_1       slot 1 – 21 (x,y) uint8 pairs; zeros = absent
#
# Total: 20 header bytes + 84 payload bytes = 104 bytes per packet.
# ──────────────────────────────────────────────────────────────────────────────

MAGIC: bytes = b"\xa5\x4c"
VERSION: int = 0x03

HAND_SLOT_COUNT: int = 2
PAYLOAD_SIZE: int = HAND_SLOT_COUNT * BYTES_PER_HAND  # 84

# struct format: 2s B I Q B H H  →  2+1+4+8+1+2+2 = 20 bytes
_HEADER_FMT: str = "<2sBIQBHH"
HEADER_SIZE: int = struct.calcsize(_HEADER_FMT)          # 20
PACKET_SIZE: int = HEADER_SIZE + PAYLOAD_SIZE            # 104

# Byte slice of the checksum field within the header
_CHECKSUM_OFFSET: int = 18
_CHECKSUM_SIZE: int = 2

_EMPTY_HAND_SLOT: bytes = bytes(BYTES_PER_HAND)  # 42 zero bytes

# Orientation values exposed by the decoded PosePacket.
# These intentionally match the debug detector's stable labels for
# UNKNOWN/ABSENT, PALM, and BACK.
ORIENTATION_UNKNOWN: int = 0
ORIENTATION_PALM: int = 1
ORIENTATION_BACK: int = 2

ORIENTATION_NAMES = {
    ORIENTATION_UNKNOWN: "UNKNOWN",
    ORIENTATION_PALM: "PALM",
    ORIENTATION_BACK: "BACK",
}

# Byte 15 bit layout.
_HAND_PRESENT_BITS = (0, 1)          # RIGHT, LEFT
_ORIENTATION_VALUE_BITS = (2, 4)    # 0=PALM, 1=BACK
_ORIENTATION_VALID_BITS = (3, 5)    # 1=orientation is valid
_ALLOWED_STATUS_MASK: int = 0b00111111


# ──────────────────────────────────────────────────────────────────────────────
# CRC-16 helper
# ──────────────────────────────────────────────────────────────────────────────

def _crc16(data: bytes) -> int:
    """CRC-16/CCITT (initial value 0xFFFF)."""
    return binascii.crc_hqx(data, 0xFFFF)


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PosePacket:
    """Decoded contents of a single pose packet."""

    frame_id: int
    timestamp_ms: int
    hands: List[Optional[QuantizedHand]]   # length == HAND_SLOT_COUNT; None = absent
    orientations: List[int] = field(
        default_factory=lambda: [ORIENTATION_UNKNOWN] * HAND_SLOT_COUNT
    )


# ──────────────────────────────────────────────────────────────────────────────
# Encode
# ──────────────────────────────────────────────────────────────────────────────

def encode_packet(
    frame_id: int,
    timestamp_ms: int,
    hands: List[Optional[QuantizedHand]],
    orientations: Optional[List[Optional[int]]] = None,
) -> bytes:
    """
    Encode a pose frame into a fixed 104-byte packet.

    Slot convention:
        hands[0] -> RIGHT hand
        hands[1] -> LEFT hand

    hands must contain up to 2 entries.
    None means that hand is absent.

    orientations uses the same slot convention:
        orientations[0] -> RIGHT
        orientations[1] -> LEFT

    Accepted orientation values:
        ORIENTATION_UNKNOWN (0) or None -> orientation not valid on wire
        ORIENTATION_PALM    (1)         -> valid PALM
        ORIENTATION_BACK    (2)         -> valid BACK

    The argument is optional so existing callers remain source-compatible
    while the sender is being upgraded to protocol v3.
    """

    if not (0 <= frame_id <= 0xFFFFFFFF):
        raise ValueError(
            f"frame_id {frame_id} out of uint32 range"
        )

    if not (0 <= timestamp_ms <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError(
            f"timestamp_ms {timestamp_ms} out of uint64 range"
        )

    # Normalize to exactly two slots:
    #
    # slot 0 = RIGHT
    # slot 1 = LEFT
    slots: List[Optional[QuantizedHand]] = [None, None]

    for i, hand in enumerate(hands[:HAND_SLOT_COUNT]):
        slots[i] = hand

    # Normalize orientation metadata to exactly two slots.
    orientation_slots: List[int] = [
        ORIENTATION_UNKNOWN,
        ORIENTATION_UNKNOWN,
    ]

    if orientations is not None:
        for i, orientation in enumerate(orientations[:HAND_SLOT_COUNT]):
            if orientation is None:
                orientation_slots[i] = ORIENTATION_UNKNOWN
                continue

            orientation = int(orientation)

            if orientation not in (
                ORIENTATION_UNKNOWN,
                ORIENTATION_PALM,
                ORIENTATION_BACK,
            ):
                raise ValueError(
                    f"Invalid orientation for slot {i}: {orientation}. "
                    f"Expected UNKNOWN={ORIENTATION_UNKNOWN}, "
                    f"PALM={ORIENTATION_PALM}, or BACK={ORIENTATION_BACK}."
                )

            orientation_slots[i] = orientation

    payload = bytearray(PAYLOAD_SIZE)
    status_flags = 0

    for slot, hand in enumerate(slots):

        if hand is None:
            # An absent hand never carries orientation metadata.
            continue

        # Mark this hand slot as present.
        status_flags |= (1 << _HAND_PRESENT_BITS[slot])

        offset = slot * BYTES_PER_HAND

        payload[
            offset : offset + BYTES_PER_HAND
        ] = hand.points_u8

        orientation = orientation_slots[slot]

        if orientation in (ORIENTATION_PALM, ORIENTATION_BACK):
            # Orientation is known and stable.
            status_flags |= (1 << _ORIENTATION_VALID_BITS[slot])

            if orientation == ORIENTATION_BACK:
                status_flags |= (1 << _ORIENTATION_VALUE_BITS[slot])

    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        VERSION,
        frame_id,
        timestamp_ms,
        status_flags,
        PAYLOAD_SIZE,
        0,
    )

    raw = header + bytes(payload)

    crc = _crc16(raw)

    packet = bytearray(raw)

    struct.pack_into(
        "<H",
        packet,
        _CHECKSUM_OFFSET,
        crc,
    )

    return bytes(packet)


# ──────────────────────────────────────────────────────────────────────────────
# Decode
# ──────────────────────────────────────────────────────────────────────────────

class PacketDecodeError(Exception):
    """Raised when a packet cannot be decoded due to structural or integrity issues."""


def decode_packet(data: bytes) -> PosePacket:
    """
    Decode and validate a 104-byte wire packet.

    Slot convention:
        slot 0 -> RIGHT hand
        slot 1 -> LEFT hand
    """

    if len(data) != PACKET_SIZE:
        raise PacketDecodeError(
            f"Expected {PACKET_SIZE} bytes, got {len(data)}"
        )

    (
        magic,
        version,
        frame_id,
        timestamp_ms,
        status_flags,
        payload_len,
        received_crc,
    ) = struct.unpack_from(
        _HEADER_FMT,
        data,
        0,
    )

    if magic != MAGIC:
        raise PacketDecodeError(
            f"Bad magic: expected {MAGIC.hex()}, "
            f"got {magic.hex()}"
        )

    if version != VERSION:
        raise PacketDecodeError(
            f"Unsupported version: {version:#04x}"
        )

    if payload_len != PAYLOAD_SIZE:
        raise PacketDecodeError(
            f"payload_len mismatch: "
            f"expected {PAYLOAD_SIZE}, got {payload_len}"
        )

    # Bits 6 and 7 are reserved in protocol v3.
    if status_flags & ~_ALLOWED_STATUS_MASK:
        raise PacketDecodeError(
            f"Invalid status flags: {status_flags:#04x}"
        )

    # CRC validation
    verify_buf = bytearray(data)

    struct.pack_into(
        "<H",
        verify_buf,
        _CHECKSUM_OFFSET,
        0,
    )

    expected_crc = _crc16(
        bytes(verify_buf)
    )

    if received_crc != expected_crc:
        raise PacketDecodeError(
            f"CRC mismatch: "
            f"expected {expected_crc:#06x}, "
            f"got {received_crc:#06x}"
        )

    payload = data[HEADER_SIZE:]

    hands: List[Optional[QuantizedHand]] = []
    orientations: List[int] = []

    for slot in range(HAND_SLOT_COUNT):

        present = bool(
            status_flags & (1 << _HAND_PRESENT_BITS[slot])
        )

        orientation_valid = bool(
            status_flags & (1 << _ORIENTATION_VALID_BITS[slot])
        )

        orientation_is_back = bool(
            status_flags & (1 << _ORIENTATION_VALUE_BITS[slot])
        )

        # Canonical/structural validation.
        if not present and (orientation_valid or orientation_is_back):
            raise PacketDecodeError(
                f"Orientation metadata set for absent hand slot {slot}"
            )

        if orientation_is_back and not orientation_valid:
            raise PacketDecodeError(
                f"Orientation value bit set without valid bit for slot {slot}"
            )

        if present:

            offset = slot * BYTES_PER_HAND

            slot_bytes = payload[
                offset : offset + BYTES_PER_HAND
            ]

            hands.append(
                QuantizedHand(slot_bytes)
            )

            if orientation_valid:
                orientations.append(
                    ORIENTATION_BACK
                    if orientation_is_back
                    else ORIENTATION_PALM
                )
            else:
                orientations.append(ORIENTATION_UNKNOWN)

        else:
            hands.append(None)
            orientations.append(ORIENTATION_UNKNOWN)

    return PosePacket(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        hands=hands,
        orientations=orientations,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: serialise a stream of packets to/from bytes
# ──────────────────────────────────────────────────────────────────────────────

def encode_packet_stream(
    packets: List[PosePacket],
    source_hands_list: List[
        List[Optional[QuantizedHand]]
    ],
) -> bytes:
    """Concatenate multiple encoded packets into a single byte string."""
    return b"".join(
        encode_packet(
            p.frame_id,
            p.timestamp_ms,
            hands,
            orientations=p.orientations,
        )
        for p, hands in zip(packets, source_hands_list)
    )


def iter_decode_stream(data: bytes):
    """
    Iterate over a concatenated byte stream and yield decoded PosePackets.

    Silently skips any packet whose CRC or structural validation fails,
    allowing the FSK decoder to keep processing remaining frames.
    """
    offset = 0
    while offset + PACKET_SIZE <= len(data):
        chunk = data[offset : offset + PACKET_SIZE]
        try:
            yield decode_packet(chunk)
        except PacketDecodeError:
            pass
        offset += PACKET_SIZE
