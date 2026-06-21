# Pose Packet Guideline

This guide explains how the pose packet module in `Pose_PacketUp` works, how to run it, and how to use it in future stages (FSK receiver, smoothing, renderer, and live pipeline).

## 1. Purpose

The pose packet module provides a deterministic binary format so your pipeline can:

1. Convert hand landmarks into compact bytes.
2. Serialize each frame into a fixed-size packet.
3. Validate packet integrity with CRC.
4. Decode packets robustly and skip corrupted frames.

This design is ideal for modulation/demodulation testing because every frame has the same byte size.

## 2. Files in `Pose_PacketUp`

- `pose_codec.py`
  - Converts MediaPipe normalized `(x, y, z)` landmarks to compact `(x, y)` uint8 bytes.
- `pose_packet.py`
  - Defines fixed packet layout, encode/decode, and stream helpers.
- `test_pose_packet.py`
  - Golden-vector tests for size, layout, CRC behavior, and decode robustness.

## 3. Coordinate Compression (`pose_codec.py`)

### 3.1 Quantization contract

Input assumptions:
- One hand = 21 landmarks.
- Use only `(x, y)` from normalized range `[0.0, 1.0]`.

Quantization:
- `x_u8 = int(round(clamp(x, 0, 1) * 255))`
- `y_u8 = int(round(clamp(y, 0, 1) * 255))`

Per-hand size:
- `21 points * 2 coords = 42 bytes`.

### 3.2 Main APIs

- `quantize_hand_landmarks_xy(hand_landmarks) -> QuantizedHand`
- `quantize_all_hands(hand_landmarks_list) -> List[QuantizedHand]`
- `quantized_hand_to_xy_pairs(hand) -> List[(x, y)]` (debug/decode helper)

## 4. Packet Format (`pose_packet.py`)

All fields are little-endian.
Packet size is fixed at 104 bytes.

### 4.1 Header + payload layout

| Offset | Size | Field | Description |
|---|---:|---|---|
| 0 | 2 | MAGIC | `0xA5, 0x4C` |
| 2 | 1 | VERSION | protocol version, currently `0x01` |
| 3 | 4 | FRAME_ID | uint32 frame counter |
| 7 | 8 | TIMESTAMP_MS | uint64 timestamp in ms |
| 15 | 1 | HAND_COUNT | number of valid hands in this frame (0..2) |
| 16 | 2 | PAYLOAD_LEN | always 84 |
| 18 | 2 | CHECKSUM | CRC-16/CCITT over packet with checksum bytes zeroed |
| 20 | 42 | HAND_0 | first hand slot (`21 * (x,y)` uint8) |
| 62 | 42 | HAND_1 | second hand slot (`21 * (x,y)` uint8) |

Total:
- Header = 20 bytes
- Payload = 84 bytes
- Packet = 104 bytes

### 4.2 Hand slot rules

- Supports up to 2 hands per frame.
- If fewer than 2 hands are present, remaining slot bytes are all zeros.
- `HAND_COUNT` indicates how many slots are valid.

## 5. CRC Strategy

CRC uses `binascii.crc_hqx(data, 0xFFFF)` (CRC-16/CCITT with initial value `0xFFFF`).

Encode flow:
1. Pack header with checksum = 0.
2. Append payload.
3. Compute CRC on entire raw packet.
4. Write CRC into checksum field.

Decode flow:
1. Parse header fields.
2. Validate magic/version/size/hand_count range.
3. Zero checksum bytes in a copy.
4. Recompute CRC and compare.
5. Reject packet if mismatch.

## 6. Key APIs (`pose_packet.py`)

### 6.1 Encode one frame

- `encode_packet(frame_id, timestamp_ms, hands) -> bytes`

Notes:
- Returns exactly 104 bytes.
- Truncates extra hands beyond 2.
- Raises `ValueError` for out-of-range frame_id/timestamp.

### 6.2 Decode one frame

- `decode_packet(data) -> PosePacket`

Notes:
- Requires exactly 104 bytes.
- Raises `PacketDecodeError` on structural or CRC failures.

### 6.3 Stream helpers

- `encode_packet_stream(packets, source_hands_list) -> bytes`
- `iter_decode_stream(data)`

`iter_decode_stream(data)` behavior:
- Scans fixed 104-byte chunks.
- Yields decoded packets.
- Silently skips corrupt chunks to keep the pipeline alive.

## 7. How to Run

Run from project root:
- `d:/Project/2D-Hand-Estimation-Model-Testcase`

### 7.1 Run packet tests

```powershell
python Pose_PacketUp/test_pose_packet.py
```

Expected summary:
- `All 9 test groups passed.`

### 7.2 Typical import usage in other modules

```python
from Pose_PacketUp.pose_codec import quantize_all_hands
from Pose_PacketUp.pose_packet import encode_packet, decode_packet, iter_decode_stream
```

## 8. Example Usage (Encode + Decode)

```python
import time
from Pose_PacketUp.pose_codec import QuantizedHand
from Pose_PacketUp.pose_packet import encode_packet, decode_packet

# One fake hand with 42 bytes (for demo)
hand = QuantizedHand(bytes([127] * 42))

packet = encode_packet(
    frame_id=1,
    timestamp_ms=int(time.time() * 1000),
    hands=[hand],
)

decoded = decode_packet(packet)
print(decoded.frame_id, decoded.timestamp_ms, decoded.hands[0] is not None)
```

## 9. Integration in the Full Pipeline

Current placement in your architecture:

1. MediaPipe landmarks -> `quantize_all_hands(...)`
2. Quantized hands -> `encode_packet(...)`
3. Packet bytes -> FSK sender module
4. FSK receiver output bytes -> `decode_packet(...)`
5. Decoded poses -> smoothing and renderer

## 10. Best Practices for Future Work

1. Do not change packet size or header offsets unless you also version-bump protocol.
2. Keep `VERSION` backward-compatible checks in receiver.
3. Keep `PACKET_SIZE` fixed for simple DSP framing.
4. Track packet metrics during decode:
   - total packets
   - CRC failures
   - valid packet ratio
5. For live mode, preserve monotonic `frame_id` and real timestamps.

## 11. Common Errors and Fixes

### Error: `Expected 104 bytes, got ...`
Cause:
- Receiver chunking boundary is wrong.
Fix:
- Ensure demodulator outputs exact 104-byte frame chunks before decode.

### Error: `CRC mismatch`
Cause:
- Bit/symbol errors in channel or demodulator.
Fix:
- Improve synchronization/preamble detection and symbol timing recovery.

### Error: import path failure (`Pose_PacketUp.*`)
Cause:
- Running scripts from inside subfolder.
Fix:
- Run commands from project root so package imports resolve.

## 12. What to Build Next

For Phase 2.2 receiver:

1. Detect preamble boundaries in WAV.
2. Recover bitstream per packet.
3. Rebuild 104-byte chunks.
4. Call `decode_packet(...)`.
5. Drop CRC-failed frames and continue using `iter_decode_stream` pattern.

This keeps your packet layer stable while you iterate on DSP/demodulation.
