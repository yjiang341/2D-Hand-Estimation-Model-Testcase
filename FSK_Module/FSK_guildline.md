# FSK Module Guideline

This document explains how the current Phase 2.1 FSK sender works, how to run it, and how to use it later when you connect receiver and rendering stages.

## 1) Scope of Current Implementation

Implemented now (Phase 2.1):
- Convert pose packets (104 bytes each) into framed byte streams.
- Convert bytes into bits (MSB-first).
- BFSK modulate bits into an audio waveform.
- Save waveform to `.wav` and optionally save packet binary stream.

Not implemented yet:
- Receiver synchronization and demodulation (Phase 2.2).
- Pose reconstruction video output (Phase 2.3 / 2.4).

## 2) Files and Responsibilities

- `FSK_Module/fsk_modem.py`
  - Core modulation functions.
  - Contains `FSKConfig`, framing, bit conversion, BFSK synthesis, WAV writer.
- `FSK_Module/fsk_sender_main.py`
  - CLI pipeline for offline sender generation.
  - Builds demo packets or loads existing packet stream from `.bin`.
- `FSK_Module/test_fsk_sender.py`
  - Quick tests for bit mapping, waveform length, and WAV writing.
- `pose_packet.py`
  - Defines deterministic packet format (104 bytes).
- `pose_codec.py`
  - Defines quantized hand representation (`42 bytes / hand`).

## 3) End-to-End Data Flow

The sender pipeline is:

1. Pose packets (`bytes`, each 104 bytes)
2. Frame each packet with preamble:
   - `framed_packet = preamble + packet`
3. Convert framed bytes to bit sequence (MSB-first)
4. BFSK modulation
   - bit `0` -> `freq0_hz`
   - bit `1` -> `freq1_hz`
5. Add optional inter-frame silence
6. Concatenate all packet wave chunks
7. Convert float waveform to PCM16
8. Write mono WAV

## 4) Wire and Modulation Parameters

### 4.1 Packet size assumptions

- Pose packet size: `104` bytes (from `pose_packet.py`)
- Preamble size: `4` bytes (default `0x55 0x55 0x55 0xD5`)
- Framed bytes per packet: `108` bytes
- Framed bits per packet: `108 * 8 = 864` bits

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
- `sample_rate` must be divisible by `symbol_rate`.

### 4.3 Time-per-frame estimate

With default settings:

- Symbol duration:
  - $T_s = 1 / 1200 \approx 0.833\text{ ms}$
- Payload duration per framed packet:
  - $864 * T_s \approx 0.72\text{ s}$
- With 3 ms silence:
  - total about `0.723 s / packet`

This is intentionally slow for robust offline validation and easy debugging.
Later optimization can increase symbol rate and reduce preamble/silence.

## 5) Core APIs

### 5.1 `FSK_Module/fsk_modem.py`

- `bytes_to_bits(data: bytes) -> np.ndarray`
  - MSB-first bit conversion.
- `frame_packet_bytes(packet_bytes, config) -> bytes`
  - Adds preamble to one packet.
- `modulate_bits_fsk(bits, config) -> np.ndarray`
  - Continuous-phase BFSK generation.
- `modulate_packet_stream(packets, config) -> np.ndarray`
  - Full stream modulation.
- `write_wav_pcm16(path, waveform, sample_rate)`
  - Writes mono PCM16 WAV.

### 5.2 `FSK_Module/fsk_sender_main.py`

- Demo mode:
  - Generates deterministic test packets.
- Packet file mode:
  - Loads concatenated packets from `.bin` where length is multiple of 104.
- Output:
  - `.wav` signal
  - `.bin` packets used for modulation

## 6) How to Run

Run commands from project root:
- `d:/Project/2D-Hand-Estimation-Model-Testcase`

### 6.1 Run tests first

```powershell
python FSK_Module/test_fsk_sender.py
```

Expected output:
- `All Phase 2.1 FSK sender tests passed.`

### 6.2 Generate demo packet stream + WAV

```powershell
python FSK_Module/fsk_sender_main.py --frames 15 --fps 15 --out-wav logs/phase2_1_sender.wav --out-packets logs/phase2_1_packets.bin
```

What this does:
- Creates deterministic packets for 15 frames.
- Modulates them to BFSK waveform.
- Saves:
  - `logs/phase2_1_sender.wav`
  - `logs/phase2_1_packets.bin`

### 6.3 Use custom modulation parameters

```powershell
python FSK_Module/fsk_sender_main.py --frames 30 --sample-rate 48000 --symbol-rate 2400 --freq0 1500 --freq1 3000 --silence-ms 1 --out-wav logs/sender_fast.wav --out-packets logs/sender_fast.bin
```

### 6.4 Modulate from existing packet binary

```powershell
python FSK_Module/fsk_sender_main.py --packet-bin logs/phase2_1_packets.bin --out-wav logs/replay_sender.wav --out-packets logs/replay_packets.bin
```

## 7) Example Integration Snippet (Python)

```python
from FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream, write_wav_pcm16
from pose_packet import encode_packet

packets = []
for i in range(10):
    packets.append(encode_packet(frame_id=i, timestamp_ms=1000 + i * 66, hands=[]))

cfg = FSKConfig(sample_rate=48000, symbol_rate=1200, freq0_hz=1200, freq1_hz=2200)
wave = modulate_packet_stream(packets, cfg)
write_wav_pcm16("logs/example_sender.wav", wave, cfg.sample_rate)
```

## 8) Output Validation Checklist

After running sender:

1. Check output files exist:
   - WAV file size should be > 44 bytes (WAV header only is 44).
2. Confirm packet binary size:
   - `packet_count * 104` bytes.
3. Confirm waveform duration is reasonable:
   - Longer with lower symbol rate / more frames / more silence.
4. Confirm no clipping:
   - Keep `amplitude <= 0.9` to avoid hard clipping.

## 9) Troubleshooting

### Error: sample_rate must be divisible by symbol_rate
- Use combinations like:
  - `48000/1200`, `48000/2400`, `44100/1470`.

### Import error for `FSK_Module.*`
- Run scripts from project root, not from inside `FSK_Module`.

### Very long WAV duration
- Increase `--symbol-rate`.
- Reduce `--frames`.
- Reduce `--silence-ms`.

### Distorted audio
- Lower `--amplitude` (e.g., `0.6`).

## 10) Recommended Next Usage Path

For future phases:

1. Keep this sender unchanged as reference baseline.
2. Implement receiver in `FSK_Module/fsk_receiver_main.py` with:
   - preamble detection
   - symbol timing recovery
   - BFSK demodulation
3. Reconstruct bytes and pass 104-byte chunks into `decode_packet()`.
4. Reject corrupt packets via CRC from `pose_packet.py`.
5. Add packet recovery metrics:
   - frame loss rate
   - bit error approximation
   - recovered FPS

This preserves deterministic sender behavior while allowing receiver experimentation without changing the trusted transmit format.
