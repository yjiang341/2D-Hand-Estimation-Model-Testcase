# FSK Module Guideline

This document explains the current offline FSK module status (Phase 2.1 + Phase 2.2), including sender and receiver paths, how to run tests/CLIs, and how to extend toward later phases.

## 1) Scope of Current Implementation

Implemented now:
- Phase 2.1 sender:
  - Serialize pose packets to framed byte stream.
  - Convert bytes to MSB-first bits.
  - BFSK modulation to waveform.
  - Write PCM16 `.wav`.
- Phase 2.2 receiver:
  - Read PCM16 `.wav`.
  - Symbol alignment search (optional auto-align).
  - BFSK demodulation.
  - Preamble/frame extraction.
  - Packet reconstruction and CRC-backed validation via `Pose_PacketUp.decode_packet`.
  - Drop corrupted frames and continue.

Not implemented yet:
- Phase 2.3 pose smoothing and reconstruction.
- Phase 2.4 skeleton rendering to MP4.

## 2) Files and Responsibilities

- `FSK_Module/fsk_modem.py`
  - Shared DSP and framing helpers for both sender and receiver.
  - Includes modulation and demodulation utilities.
- `FSK_Module/fsk_sender_main.py`
  - Offline sender CLI (`packet stream -> wav`).
- `FSK_Module/fsk_receiver.py`
  - Receiver pipeline utilities (`wav -> recovered packets`).
- `FSK_Module/fsk_receiver_main.py`
  - Offline receiver CLI.
- `FSK_Module/test_fsk_sender.py`
  - Sender-focused tests.
- `FSK_Module/test_fsk_receiver.py`
  - Receiver-focused tests (clean + corrupted conditions).
- `Pose_PacketUp/pose_packet.py`
  - Packet spec (`104` bytes) and CRC validation.
- `Pose_PacketUp/pose_codec.py`
  - Quantized hand payload (`42` bytes/hand).

## 3) End-to-End Data Flow

### 3.1 Sender

1. Pose packet bytes (104 each).
2. Add preamble to each packet.
3. Convert framed bytes to bitstream (MSB-first).
4. BFSK modulation:
   - bit `0` -> `freq0_hz`
   - bit `1` -> `freq1_hz`
5. Insert inter-frame silence.
6. Concatenate waveform.
7. Convert float `[-1, 1]` to PCM16 and write WAV.

### 3.2 Receiver

1. Read PCM16 WAV (mono/stereo supported; stereo is down-mixed).
2. Optional symbol timing offset search (`0..samples_per_symbol-1`) using preamble-hit scoring.
3. Demodulate bits symbol-by-symbol via Goertzel energy at `freq0_hz` and `freq1_hz`.
4. Detect preamble in bitstream and extract following `104*8` bits per frame.
5. Convert bits to bytes.
6. Validate packets with `Pose_PacketUp.decode_packet` (CRC + structure).
7. Keep valid frames, reject corrupted ones.

## 4) Parameters and Derived Values

### 4.1 Packet assumptions

- Pose packet size: `104` bytes.
- Preamble size: `4` bytes (`0x55 0x55 0x55 0xD5`).
- Framed bytes per packet: `108` bytes.
- Framed bits per packet: `108 * 8 = 864` bits.

### 4.2 Default `FSKConfig`

- `sample_rate = 48000`
- `symbol_rate = 1200`
- `freq0_hz = 1200`
- `freq1_hz = 2200`
- `amplitude = 0.8`
- `preamble = b"\x55\x55\x55\xD5"`
- `inter_frame_silence_ms = 3`

Derived:

- `samples_per_symbol = sample_rate / symbol_rate = 40`

Constraint:

- `sample_rate % symbol_rate == 0`

## 5) Core APIs

### 5.1 Sender-related (`fsk_modem.py`)

- `bytes_to_bits(data)`
- `frame_packet_bytes(packet_bytes, config)`
- `frame_packet_stream(packets, config)`
- `modulate_bits_fsk(bits, config)`
- `modulate_packet_stream(packets, config)`
- `write_wav_pcm16(path, waveform, sample_rate)`

### 5.2 Receiver-related (`fsk_modem.py` + `fsk_receiver.py`)

- `read_wav_pcm16(path)`
- `demodulate_bits_fsk(waveform, config, sample_offset=0)`
- `find_best_symbol_offset(waveform, config, probe_bits=...)`
- `extract_packets_from_demod_bits(bits, config, packet_size)`
- `demodulate_packet_stream(waveform, config, packet_size, auto_align=True)`
- `recover_packets_from_waveform(waveform, config, detection_threshold=...)`
- `recover_packets_from_wav(wav_path, config, detection_threshold=...)`

## 6) How to Run

Run from project root:

- `d:/Project/2D-Hand-Estimation-Model-Testcase`

### 6.1 Run tests

```powershell
python -m FSK_Module.test_fsk_sender
python -m FSK_Module.test_fsk_receiver
```

Expected:

- `All Phase 2.1 FSK sender tests passed.`
- `All Phase 2.2 FSK receiver tests passed.`

### 6.2 Generate sender artifacts

```powershell
python -m FSK_Module.fsk_sender_main --frames 15 --fps 15 --out-wav logs/phase2_1_sender.wav --out-packets logs/phase2_1_packets.bin
```

### 6.3 Recover packets from WAV

```powershell
python -m FSK_Module.fsk_receiver_main --in-wav logs/phase2_1_sender.wav --out-recovered logs/phase2_2_recovered_packets.bin
```

Typical successful report:

- preamble candidates = frame count
- valid frames = attempted frames
- rejected frames = 0 (clean generated WAV)

### 6.4 Custom receiver parameters

```powershell
python -m FSK_Module.fsk_receiver_main --in-wav logs/phase2_1_sender.wav --sample-rate 48000 --symbol-rate 1200 --freq0 1200 --freq1 2200 --silence-ms 3 --detect-threshold 0.55 --out-recovered logs/recovered_custom.bin
```

## 7) Quick Integration Snippets

### 7.1 Sender snippet

```python
from FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream, write_wav_pcm16
from Pose_PacketUp.pose_packet import encode_packet

packets = [encode_packet(frame_id=i, timestamp_ms=1000 + i * 66, hands=[]) for i in range(10)]
cfg = FSKConfig()
wave = modulate_packet_stream(packets, cfg)
write_wav_pcm16("logs/example_sender.wav", wave, cfg.sample_rate)
```

### 7.2 Receiver snippet

```python
from FSK_Module.fsk_modem import FSKConfig
from FSK_Module.fsk_receiver import recover_packets_from_wav

cfg = FSKConfig()
report = recover_packets_from_wav("logs/phase2_1_sender.wav", cfg, detection_threshold=0.55)
print(report.valid_frames, report.rejected_frames)
```

## 8) Validation Checklist

1. Sender WAV exists and is larger than 44-byte header.
2. Sender packet bin size is `packet_count * 104` bytes.
3. Receiver recovers packet count close to expected frame count.
4. CRC-rejected frames are nonzero only under noise/corruption.
5. In clean generated WAV tests, receiver should recover all frames.

## 9) Troubleshooting

### Import error for `FSK_Module.*` or `Pose_PacketUp.*`

- Use module execution from project root:
  - `python -m FSK_Module.fsk_sender_main`
  - `python -m FSK_Module.fsk_receiver_main`

### Error: sample rate mismatch at receiver

- Ensure `--sample-rate` matches WAV metadata used during sender generation.

### Low recovery / many rejected frames

- Verify `freq0/freq1/symbol_rate` exactly match sender.
- Tune `--detect-threshold` (e.g. 0.45 to 0.70).
- Increase inter-frame silence in sender for easier segmentation.

### Distorted output

- Reduce sender amplitude (example: `--amplitude 0.6`) to avoid clipping.

## 10) Next Steps (Phase 2.3+)

1. Feed recovered valid packets into pose reconstruction (`decode_packet -> keypoints`).
2. Add temporal smoothing (EMA).
3. Render skeleton-only MP4 from recovered frames.
4. Measure end-to-end metrics:
   - recovered FPS
   - valid/rejected frame ratio
   - pose continuity under noise.
