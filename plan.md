Plan: ASL Pose-to-Audio Prototype
Recommended approach is to build this in two tracks:

A stable offline path first: video/webcam frames -> pose packets -> FSK WAV -> decoded pose -> skeleton MP4.
Then real-time/live integration for webcam and conferencing experiments.
This matches your selected constraints:

FSK first
Offline first
Up to 2 hands
Normalized coordinates 0..1 to uint8 0..255
Skeleton-only receiver video

Steps
Phase 0: Freeze data contract for a single pose frame.
Define one canonical frame schema before coding DSP:
timestamp, frame_id, hand_count, 2 fixed hand slots, each slot 21x2 uint8, plus validity flags and checksum.

Phase 1.1: First milestone you requested (x,y only + normalization).
At extraction points in image_main.py:109 and video_main.py:125:
discard z, keep normalized x,y, clamp to [0,1], quantize with round/clamp to uint8 [0..255].
This guarantees 42 bytes per hand per frame.

Phase 1.2: Packet serialization format.
Implement deterministic byte layout:
header (magic/version/frame_id/timestamp/hand_count/payload_len/checksum) + fixed payload (2 hand slots).
This makes modulation and demodulation testable with golden vectors.

Phase 1.3: Unit tests for preprocessing and packet I/O.
Validate:
boundary mappings (0.0 -> 0, 1.0 -> 255),
shape/size invariants,
packet encode/decode round-trip,
corrupt packet rejection.

Phase 2.1: Offline FSK sender.
Map packet bytes to bits/symbols, generate FSK waveform with fixed sample rate/symbol duration, and write WAV.

Phase 2.2: Offline FSK receiver.
Detect preamble/frame boundaries, demodulate bits, rebuild packets, verify checksum, and drop corrupted frames safely.

Phase 2.3: Pose smoothing and reconstruction.
Decode uint8 back to normalized float, apply temporal smoothing (EMA first), then enforce simple sanity bounds.

Phase 2.4: Skeleton renderer to MP4.
Render recovered keypoints on black canvas and export MP4 at target FPS for intelligibility checks.

Phase 3: Live webcam path.
Add webcam capture mode to replace/extend YouTube path in video_main.py:83, then run real-time sender/receiver loop with queueing and latency metrics.

Phase 4: Conferencing readiness.
Run virtual-audio loop tests through meeting apps, measure symbol/frame loss and end-to-end latency, then tune FSK parameters and fallback behavior.