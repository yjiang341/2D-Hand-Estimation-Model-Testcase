"""
Golden-vector tests for pose_packet.py.

Run with:
    python test_pose_packet.py

All assertions produce a clear PASS / FAIL line with the specific value that
failed, so they are easy to diagnose without a test framework.
"""
from __future__ import annotations

import struct

from pose_codec import BYTES_PER_HAND, QuantizedHand
from pose_packet import (
    HAND_SLOT_COUNT,
    HEADER_SIZE,
    MAGIC,
    PACKET_SIZE,
    PAYLOAD_SIZE,
    VERSION,
    PacketDecodeError,
    _CHECKSUM_OFFSET,
    _crc16,
    decode_packet,
    encode_packet,
    iter_decode_stream,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_hand(fill: int) -> QuantizedHand:
    """Return a QuantizedHand with all 42 bytes set to `fill`."""
    return QuantizedHand(bytes([fill] * BYTES_PER_HAND))


def _assert(condition: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    if not condition:
        raise AssertionError(label)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: layout constants
# ──────────────────────────────────────────────────────────────────────────────

def test_layout_constants() -> None:
    print("Test 1 – layout constants")
    _assert(BYTES_PER_HAND == 42,    "BYTES_PER_HAND == 42",  str(BYTES_PER_HAND))
    _assert(HAND_SLOT_COUNT == 2,    "HAND_SLOT_COUNT == 2",  str(HAND_SLOT_COUNT))
    _assert(PAYLOAD_SIZE == 84,      "PAYLOAD_SIZE == 84",    str(PAYLOAD_SIZE))
    _assert(HEADER_SIZE == 20,       "HEADER_SIZE == 20",     str(HEADER_SIZE))
    _assert(PACKET_SIZE == 104,      "PACKET_SIZE == 104",    str(PACKET_SIZE))
    _assert(_CHECKSUM_OFFSET == 18,  "checksum at offset 18", str(_CHECKSUM_OFFSET))


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: encode produces 104-byte packet
# ──────────────────────────────────────────────────────────────────────────────

def test_encode_size() -> None:
    print("Test 2 – encode produces exactly 104 bytes")
    for n_hands, label in [(0, "0 hands"), (1, "1 hand"), (2, "2 hands")]:
        hands = [_make_hand(0xAB)] * n_hands
        pkt = encode_packet(frame_id=1, timestamp_ms=1000, hands=hands)
        _assert(len(pkt) == PACKET_SIZE, f"len(packet) == 104 ({label})", str(len(pkt)))


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: header fields are written correctly
# ──────────────────────────────────────────────────────────────────────────────

def test_header_fields() -> None:
    print("Test 3 – header field values")
    pkt = encode_packet(
        frame_id=0x0000_1234,
        timestamp_ms=9_876_543_210,
        hands=[_make_hand(0x10), _make_hand(0x20)],
    )
    _assert(pkt[0:2] == MAGIC,       "magic bytes correct",       pkt[0:2].hex())
    _assert(pkt[2] == VERSION,       "version correct",           str(pkt[2]))
    frame_id_le = struct.unpack_from("<I", pkt, 3)[0]
    _assert(frame_id_le == 0x1234,   "frame_id correct",          hex(frame_id_le))
    ts_le = struct.unpack_from("<Q", pkt, 7)[0]
    _assert(ts_le == 9_876_543_210,  "timestamp_ms correct",      str(ts_le))
    _assert(pkt[15] == 2,            "hand_count == 2",           str(pkt[15]))
    plen = struct.unpack_from("<H", pkt, 16)[0]
    _assert(plen == PAYLOAD_SIZE,    "payload_len == 84",         str(plen))


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: hand slot payloads are placed correctly
# ──────────────────────────────────────────────────────────────────────────────

def test_payload_slots() -> None:
    print("Test 4 – payload slot placement")
    hand0 = _make_hand(0xAA)
    hand1 = _make_hand(0xBB)
    pkt = encode_packet(frame_id=0, timestamp_ms=0, hands=[hand0, hand1])
    slot0 = pkt[HEADER_SIZE : HEADER_SIZE + BYTES_PER_HAND]
    slot1 = pkt[HEADER_SIZE + BYTES_PER_HAND : HEADER_SIZE + 2 * BYTES_PER_HAND]
    _assert(slot0 == bytes([0xAA] * BYTES_PER_HAND), "slot 0 data correct")
    _assert(slot1 == bytes([0xBB] * BYTES_PER_HAND), "slot 1 data correct")

    # Absent hand slot must be zeroed
    pkt1 = encode_packet(frame_id=0, timestamp_ms=0, hands=[hand0])
    slot1_absent = pkt1[HEADER_SIZE + BYTES_PER_HAND : HEADER_SIZE + 2 * BYTES_PER_HAND]
    _assert(slot1_absent == bytes(BYTES_PER_HAND), "absent slot 1 is zeroed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: round-trip encode → decode
# ──────────────────────────────────────────────────────────────────────────────

def test_round_trip() -> None:
    print("Test 5 – encode → decode round trip")
    hand0 = _make_hand(0x7F)
    hand1 = _make_hand(0x3C)

    for n_hands in (0, 1, 2):
        hands = [hand0, hand1][:n_hands]
        pkt = encode_packet(frame_id=n_hands, timestamp_ms=n_hands * 1000, hands=hands)
        decoded = decode_packet(pkt)

        _assert(decoded.frame_id == n_hands,          f"frame_id ({n_hands} hands)")
        _assert(decoded.timestamp_ms == n_hands * 1000, f"timestamp_ms ({n_hands} hands)")

        for slot in range(HAND_SLOT_COUNT):
            if slot < n_hands:
                h = decoded.hands[slot]
                _assert(h is not None, f"slot {slot} not None ({n_hands} hands)")
                _assert(h.points_u8 == hands[slot].points_u8,
                        f"slot {slot} payload matches ({n_hands} hands)")
            else:
                _assert(decoded.hands[slot] is None,
                        f"absent slot {slot} is None ({n_hands} hands)")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: boundary quantization values (x=0.0 → 0, x=1.0 → 255)
# ──────────────────────────────────────────────────────────────────────────────

def test_quantization_boundaries() -> None:
    print("Test 6 – boundary quantization values in payload")
    # Build a hand where first point is (0,0) and last point is (255,255)
    buf = bytearray(BYTES_PER_HAND)
    buf[0], buf[1] = 0, 0          # point 0: x=0, y=0
    buf[40], buf[41] = 255, 255    # point 20: x=255, y=255
    hand = QuantizedHand(bytes(buf))

    pkt = encode_packet(frame_id=99, timestamp_ms=0, hands=[hand])
    decoded = decode_packet(pkt)
    recovered = decoded.hands[0].points_u8

    _assert(recovered[0] == 0,   "min x boundary == 0",   str(recovered[0]))
    _assert(recovered[1] == 0,   "min y boundary == 0",   str(recovered[1]))
    _assert(recovered[40] == 255, "max x boundary == 255", str(recovered[40]))
    _assert(recovered[41] == 255, "max y boundary == 255", str(recovered[41]))


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: CRC protects every byte position
# ──────────────────────────────────────────────────────────────────────────────

def test_crc_corruption() -> None:
    print("Test 7 – single-byte corruptions are caught by CRC")
    hand = _make_hand(0x55)
    good_pkt = encode_packet(frame_id=7, timestamp_ms=7000, hands=[hand])

    rejected = 0
    for pos in range(PACKET_SIZE):
        if pos in (_CHECKSUM_OFFSET, _CHECKSUM_OFFSET + 1):
            continue  # CRC field itself — always differs, always caught

        corrupt = bytearray(good_pkt)
        corrupt[pos] ^= 0xFF  # flip all bits at this position
        try:
            decode_packet(bytes(corrupt))
        except PacketDecodeError:
            rejected += 1

    # All PACKET_SIZE - 2 non-checksum byte positions must be caught
    expected_rejections = PACKET_SIZE - 2
    _assert(
        rejected == expected_rejections,
        f"CRC catches all {expected_rejections} single-byte corruptions",
        f"caught {rejected}/{expected_rejections}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: structural rejection (wrong magic, bad version, size mismatch)
# ──────────────────────────────────────────────────────────────────────────────

def test_structural_rejection() -> None:
    print("Test 8 – structural validation rejects malformed packets")
    hand = _make_hand(0x12)
    good = encode_packet(frame_id=8, timestamp_ms=8000, hands=[hand])

    # Wrong size
    try:
        decode_packet(good[:-1])
        _assert(False, "wrong size rejected")
    except PacketDecodeError:
        _assert(True, "wrong size rejected")

    # Bad magic
    bad_magic = bytearray(good)
    bad_magic[0] = 0x00
    try:
        decode_packet(bytes(bad_magic))
        _assert(False, "bad magic rejected")
    except PacketDecodeError:
        _assert(True, "bad magic rejected")

    # Bad version
    bad_ver = bytearray(good)
    bad_ver[2] = 0x99
    try:
        decode_packet(bytes(bad_ver))
        _assert(False, "bad version rejected")
    except PacketDecodeError:
        _assert(True, "bad version rejected")


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: iter_decode_stream skips corrupted frames
# ──────────────────────────────────────────────────────────────────────────────

def test_iter_decode_stream() -> None:
    print("Test 9 – iter_decode_stream recovers good frames around corruption")
    h = _make_hand(0x88)
    pkt0 = encode_packet(frame_id=0, timestamp_ms=0,    hands=[h])
    pkt1 = encode_packet(frame_id=1, timestamp_ms=1000, hands=[h])  # will corrupt
    pkt2 = encode_packet(frame_id=2, timestamp_ms=2000, hands=[h])

    corrupted = bytearray(pkt1)
    corrupted[HEADER_SIZE] ^= 0xFF  # corrupt first payload byte
    stream = pkt0 + bytes(corrupted) + pkt2

    good = list(iter_decode_stream(stream))
    _assert(len(good) == 2,              "2 good frames recovered",         str(len(good)))
    _assert(good[0].frame_id == 0,       "first good frame_id == 0")
    _assert(good[1].frame_id == 2,       "second good frame_id == 2")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_layout_constants,
        test_encode_size,
        test_header_fields,
        test_payload_slots,
        test_round_trip,
        test_quantization_boundaries,
        test_crc_corruption,
        test_structural_rejection,
        test_iter_decode_stream,
    ]

    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failures += 1
            print(f"  *** ASSERTION FAILED: {exc}")
        print()

    if failures == 0:
        print(f"All {len(tests)} test groups passed.")
    else:
        print(f"{failures} test group(s) FAILED.")
