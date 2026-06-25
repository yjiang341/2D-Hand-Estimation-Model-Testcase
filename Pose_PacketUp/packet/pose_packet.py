from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass
from typing import List, Optional

from Pose_PacketUp.pose_codec import BYTES_PER_HAND, QuantizedHand, quantized_hand_to_xy_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────────────
#
# Wire format (all little-endian, fixed 104 bytes per packet):
#
#  Offset  Size  Type    Field
#  ──────  ────  ──────  ───────────────────────────────────────────────────
#    0      2    uint8×2  MAGIC        = 0xA5, 0x4C
#    2      1    uint8    VERSION      = 0x01
#    3      4    uint32   FRAME_ID     monotonically increasing frame counter
#    7      8    uint64   TIMESTAMP_MS wall-clock timestamp in milliseconds
#   15      1    uint8    HAND_COUNT   number of valid hands in this frame (0-2)
#   16      2    uint16   PAYLOAD_LEN  always PAYLOAD_SIZE (84); receiver sanity check
#   18      2    uint16   CHECKSUM     CRC-16/CCITT over bytes [0..17] + [20..103]
#                                       (i.e. full packet with checksum bytes zeroed)
#   20     42    uint8×42 HAND_0       slot 0 – 21 (x,y) uint8 pairs; zeros = absent
#   62     42    uint8×42 HAND_1       slot 1 – 21 (x,y) uint8 pairs; zeros = absent
#
# Total: 20 header bytes + 84 payload bytes = 104 bytes per packet.
# ──────────────────────────────────────────────────────────────────────────────

MAGIC: bytes = b"\xa5\x4c"
VERSION: int = 0x01

HAND_SLOT_COUNT: int = 2
PAYLOAD_SIZE: int = HAND_SLOT_COUNT * BYTES_PER_HAND  # 84

# struct format: 2s B I Q B H H  →  2+1+4+8+1+2+2 = 20 bytes
_HEADER_FMT: str = "<2sBIQBHH"
HEADER_SIZE: int = struct.calcsize(_HEADER_FMT)          # 20
PACKET_SIZE: int = HEADER_SIZE + PAYLOAD_SIZE            # 104

# Byte slice of the checksum field within the header
_CHECKSUM_OFFSET: int = 18
_CHECKSUM_SIZE: int = 2

_EMPTY_HAND_SLOT: bytes = bytes(BYTES_PER_HAND)          # 42 zero bytes


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


# ──────────────────────────────────────────────────────────────────────────────
# Encode
# ──────────────────────────────────────────────────────────────────────────────

def encode_packet(
    frame_id: int,
    timestamp_ms: int,
    hands: List[QuantizedHand],
) -> bytes:
    """
    Encode a pose frame into a fixed 104-byte wire packet.

    Args:
        frame_id:     Monotonically increasing frame counter (uint32, wraps at 2^32).
        timestamp_ms: Wall-clock timestamp in milliseconds (uint64).
        hands:        0, 1, or 2 QuantizedHand objects for this frame.
                      Excess entries beyond HAND_SLOT_COUNT are silently ignored.

    Returns:
        104 bytes ready for bit-packing and FSK modulation.

    Raises:
        ValueError: If any argument exceeds its representable range.
    """
    if not (0 <= frame_id <= 0xFFFFFFFF):
        raise ValueError(f"frame_id {frame_id} out of uint32 range")
    if not (0 <= timestamp_ms <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError(f"timestamp_ms {timestamp_ms} out of uint64 range")
    if len(hands) > HAND_SLOT_COUNT:
        hands = hands[:HAND_SLOT_COUNT]

    hand_count = len(hands)

    # Build fixed-size payload (two slots; absent slot = 42 zero bytes)
    payload = bytearray(PAYLOAD_SIZE)
    for slot, hand in enumerate(hands):
        offset = slot * BYTES_PER_HAND
        payload[offset : offset + BYTES_PER_HAND] = hand.points_u8

    # Pack header with checksum = 0 initially
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        VERSION,
        frame_id,
        timestamp_ms,
        hand_count,
        PAYLOAD_SIZE,
        0,  # checksum placeholder
    )

    # Compute CRC over (header with zeroed checksum) + payload
    raw = header + bytes(payload)
    crc = _crc16(raw)

    # Splice CRC into the correct bytes
    packet = bytearray(raw)
    struct.pack_into("<H", packet, _CHECKSUM_OFFSET, crc)

    return bytes(packet)


# ──────────────────────────────────────────────────────────────────────────────
# Decode
# ──────────────────────────────────────────────────────────────────────────────

class PacketDecodeError(Exception):
    """Raised when a packet cannot be decoded due to structural or integrity issues."""


def decode_packet(data: bytes) -> PosePacket:
    """
    Decode and validate a 104-byte wire packet.

    Raises:
        PacketDecodeError: On wrong size, bad magic, version mismatch,
                           payload_len mismatch, or CRC failure.
    """
    if len(data) != PACKET_SIZE:
        raise PacketDecodeError(
            f"Expected {PACKET_SIZE} bytes, got {len(data)}"
        )

    # Unpack header
    (
        magic,
        version,
        frame_id,
        timestamp_ms,
        hand_count,
        payload_len,
        received_crc,
    ) = struct.unpack_from(_HEADER_FMT, data, 0)

    if magic != MAGIC:
        raise PacketDecodeError(
            f"Bad magic: expected {MAGIC.hex()}, got {magic.hex()}"
        )
    if version != VERSION:
        raise PacketDecodeError(
            f"Unsupported version: {version:#04x}"
        )
    if payload_len != PAYLOAD_SIZE:
        raise PacketDecodeError(
            f"payload_len mismatch: expected {PAYLOAD_SIZE}, got {payload_len}"
        )
    if not (0 <= hand_count <= HAND_SLOT_COUNT):
        raise PacketDecodeError(
            f"hand_count {hand_count} out of range [0, {HAND_SLOT_COUNT}]"
        )

    # Verify CRC: zero out checksum field and recompute
    verify_buf = bytearray(data)
    struct.pack_into("<H", verify_buf, _CHECKSUM_OFFSET, 0)
    expected_crc = _crc16(bytes(verify_buf))
    if received_crc != expected_crc:
        raise PacketDecodeError(
            f"CRC mismatch: expected {expected_crc:#06x}, got {received_crc:#06x}"
        )

    # Extract hand slots
    payload = data[HEADER_SIZE:]
    hands: List[Optional[QuantizedHand]] = []
    for slot in range(HAND_SLOT_COUNT):
        offset = slot * BYTES_PER_HAND
        slot_bytes = payload[offset : offset + BYTES_PER_HAND]
        if slot < hand_count:
            hands.append(QuantizedHand(slot_bytes))
        else:
            hands.append(None)

    return PosePacket(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        hands=hands,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: serialise a stream of packets to/from bytes
# ──────────────────────────────────────────────────────────────────────────────

def encode_packet_stream(packets: List[PosePacket], source_hands_list: List[List[QuantizedHand]]) -> bytes:
    """Concatenate multiple encoded packets into a single byte string."""
    return b"".join(
        encode_packet(p.frame_id, p.timestamp_ms, hands)
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
