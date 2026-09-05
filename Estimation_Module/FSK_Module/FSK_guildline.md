# FSK Module Guideline

This document describes the active FSK transport layer in the `2D-Hand`
pipeline. The FSK module is no longer an offline sender/receiver application.
It provides shared modulation, demodulation, framing, and packet-recovery APIs
that are called by the live and conferencing pipelines.

## 1. Current Scope

The active FSK module is responsible for:

- Converting framed Pose Packet bytes to and from an MSB-first bitstream.
- BFSK modulation and demodulation.
- Preamble generation and detection.
- Symbol alignment and packet extraction.
- Pose Packet validation through the packet decoder and CRC.
- Returning structured receiver statistics through `ReceiverReport`.

The FSK module does not own:

- Hand landmark estimation.
- Pose quantization or Pose Packet schema definitions.
- Microphone or meeting-device capture.
- FoundHand pose adaptation or image/video generation.
- A standalone production sender or receiver CLI.

The legacy fake sender and offline receiver command-line entry points have been
retired. Their removal does not remove the shared FSK DSP or receiver APIs used
by the active pipeline.

## 2. Active Architecture

### 2.1 Live receive path

```text
audio/waveform source
    -> Estimation_Module/Live_Module/live_main.py
    -> recover_packets_from_waveform(...)
    -> FSK demodulation and preamble detection
    -> Pose Packet decode and CRC validation
    -> validated PosePacket objects
    -> FoundHand bridge / downstream generation
```

`Live_Module/live_main.py` supplies waveform data to the receiver library. The
FSK receiver therefore does not need to open an audio device itself; it is the
state-independent decoding layer called by the live orchestration code.

### 2.2 Conferencing and readiness path

The conferencing tools also depend on the receiver API:

- `Estimation_Module/Conferencing_Module/metrics/readiness_metrics.py`
  consumes `ReceiverReport`.
- `Estimation_Module/Conferencing_Module/tuning/fsk_tuner.py`
  calls `recover_packets_from_waveform(...)` while evaluating FSK parameters.

These dependencies are active reasons to retain the receiver implementation.

### 2.3 Live send path

The canonical live sender belongs to the Meeting Bridge layer:

```text
Estimation_Module/Meeting_Bridge_Module/sender/meeting_sender.py
```

The retired FSK fake sender generated synthetic hand payloads or loaded packet
bytes from a file and wrote a WAV artifact. It was a manual experiment, not the
production sender.

## 3. Files and Responsibilities

### 3.1 FSK public import layer

- `Estimation_Module/FSK_Module/fsk_receiver.py`
  - Stable public receiver import path.
  - Preserved because live and conferencing modules import from this path.
- `Estimation_Module/FSK_Module/fsk_modem.py`
  - Stable public modem/DSP import path.
  - Provides the symbols used by the receiver and sender-side transport code.

Thin wrapper files are intentional compatibility layers. Do not delete them
while active modules still use their public import paths.

### 3.2 Receiver implementation

- `Estimation_Module/FSK_Module/receiver/fsk_receiver.py`
  - Defines `ReceiverReport`.
  - Detects preamble positions in waveform data.
  - Demodulates fixed-size Pose Packet payloads.
  - Validates decoded packets and rejects invalid frames.
  - Provides waveform and WAV diagnostic entry functions.

### 3.3 Modem implementation

The modem layer provides:

- `FSKConfig`
- byte-to-bit and bit-to-byte conversion
- preamble framing
- BFSK waveform modulation
- Goertzel-based tone decisions
- symbol-offset search
- packet extraction
- PCM16 WAV helpers for diagnostics

If both a public wrapper and a nested implementation exist, keep one canonical
implementation and keep the wrapper until every caller has been migrated.
Do not maintain two independent copies of the DSP logic.

### 3.4 Related packet and adapter modules

- `Estimation_Module/Pose_PacketUp/pose_packet.py`
  - Public Pose Packet import path.
- `Estimation_Module/Pose_PacketUp/packet/pose_packet.py`
  - Canonical Pose Packet implementation.
  - Defines the current packet size, packet version, decoder, and CRC rules.
- `Estimation_Module/Pose_PacketUp/pose_codec.py`
  - Public pose codec import path.
- `Estimation_Module/FoundHand_Bridge/pose_adapter.py`
  - Converts validated Pose Packet data into FoundHand-compatible keypoints.

## 4. Retired Offline Entry Points

The following legacy entry points are retired and should not be restored as
production modules:

- `Estimation_Module/FSK_Module/fsk_sender_main.py`
- `Estimation_Module/FSK_Module/sender/fsk_sender_main.py`
- `Estimation_Module/FSK_Module/fsk_receiver_main.py`
- `Estimation_Module/FSK_Module/receiver/fsk_receiver_main.py`

The fake sender created synthetic payloads and WAV files. The receiver CLI read
a complete WAV file and wrote recovered binary/NPZ artifacts. Neither entry
point participated directly in the live Meeting Bridge pipeline.

Do not leave a thin wrapper that imports one of these deleted modules. A wrapper
whose implementation target has been removed is a broken entry point.

## 5. Packet and FSK Parameters

The current modem defaults are:

| Parameter | Default |
| --- | ---: |
| Sample rate | `48000 Hz` |
| Symbol rate | `1200 symbols/s` |
| Frequency for bit 0 | `1200 Hz` |
| Frequency for bit 1 | `2200 Hz` |
| Amplitude | `0.8` |
| Preamble | `55 55 55 D5` |
| Inter-frame silence | `3 ms` |

With the default sample and symbol rates:

```text
samples_per_symbol = 48000 / 1200 = 40
```

The implementation requires:

```text
sample_rate % symbol_rate == 0
```

Packet size and version must come from the Pose Packet module. Do not duplicate
those constants inside orchestration or test scripts.

All participants in one transport session must use the same sample rate,
symbol rate, FSK frequencies, preamble, and framing parameters. Avoid separate
hard-coded sender and receiver defaults.

## 6. Active Receiver API

### 6.1 `ReceiverReport`

The receiver returns a structured report containing:

- `preamble_candidates`
- `attempted_frames`
- `valid_frames`
- `rejected_frames`
- `valid_packets`
- `valid_packet_bytes`

Production callers should consume this report instead of parsing console text.

### 6.2 Waveform recovery

```python
from Estimation_Module.FSK_Module.fsk_modem import FSKConfig
from Estimation_Module.FSK_Module.fsk_receiver import (
    recover_packets_from_waveform,
)

config = FSKConfig()
report = recover_packets_from_waveform(
    waveform,
    config,
    detection_threshold=0.55,
)

for packet in report.valid_packets:
    process(packet)
```

`recover_packets_from_waveform(...)` is the active integration API used by the
live and tuning paths. The upstream caller owns audio capture, buffering, and
threading.

### 6.3 WAV recovery for diagnostics

```python
from Estimation_Module.FSK_Module.fsk_modem import FSKConfig
from Estimation_Module.FSK_Module.fsk_receiver import recover_packets_from_wav

config = FSKConfig()
report = recover_packets_from_wav(
    "logs/diagnostic_capture.wav",
    config,
    detection_threshold=0.55,
)
```

`recover_packets_from_wav(...)` remains useful for debugging recorded audio,
but it is not a production CLI or live device interface.

## 7. Modem API

The public modem API may include:

- `bytes_to_bits(data)`
- `frame_packet_bytes(packet_bytes, config)`
- `frame_packet_stream(packets, config)`
- `modulate_bits_fsk(bits, config)`
- `modulate_packet_stream(packets, config)`
- `demodulate_bits_fsk(waveform, config, sample_offset=0)`
- `find_best_symbol_offset(waveform, config, probe_bits=...)`
- `extract_packets_from_demod_bits(bits, config, packet_size)`
- `demodulate_packet_stream(waveform, config, packet_size, auto_align=True)`
- `read_wav_pcm16(path)`
- `write_wav_pcm16(path, waveform, sample_rate)`

WAV helpers are diagnostic utilities. The modulation and demodulation
primitives may be used by active transport code and must not be removed merely
because the retired CLIs also used them.

## 8. Validation

Run validation from the repository root:

```powershell
cd D:\Project\2D-Hand
```

Use the current comparison scripts when present:

```powershell
python .\compare_raw_vs_fsk.py
python .\compare_raw_vs_v3_pose.py
```

The validation should confirm:

1. Recovered frame IDs match the transmitted frame IDs.
2. Packet CRC and structure validation reject corrupted frames.
3. Valid/rejected counts are reported through `ReceiverReport`.
4. Recovered keypoint error remains within the accepted baseline.
5. Protocol-v3 hand presence and orientation metadata survive the round trip.
6. Live receiver parameters match the sender parameters.

Do not use the retired fake sender or offline receiver CLI as the production
acceptance test.

## 9. Troubleshooting

### Import errors

Run commands from the repository root and use the full package prefix:

```python
from Estimation_Module.FSK_Module.fsk_receiver import ReceiverReport
from Estimation_Module.FSK_Module.fsk_modem import FSKConfig
```

Do not use legacy imports such as:

```python
from FSK_Module.fsk_receiver import ReceiverReport
```

### No preamble candidates

- Verify sample rate, symbol rate, and both FSK frequencies.
- Confirm that sender and receiver use the same preamble.
- Confirm that the input waveform is normalized and not empty.
- Inspect clipping, resampling, filtering, and meeting-application processing.

### Many rejected frames

- Verify Pose Packet version and packet size.
- Check whether audio processing changed the waveform timing.
- Inspect CRC failures separately from preamble-detection failures.
- Tune the detection threshold with the conferencing tuner rather than changing
  production constants blindly.

### Clean WAV works but live audio fails

This usually indicates a transport or buffering issue rather than a packet
codec problem. Check:

- chunk boundaries and retained overlap
- resampling
- automatic gain control
- echo cancellation and noise suppression
- symbol alignment across waveform items
- sender/receiver configuration drift

## 10. Maintenance Rules

- Keep live orchestration outside the low-level FSK DSP module.
- Keep one canonical implementation for each receiver/modem component.
- Preserve public wrappers while active imports depend on them.
- Do not reintroduce the fake sender as the canonical live sender.
- Do not reintroduce offline `_main` modules into the production package.
- Keep `recover_packets_from_wav(...)` only as a diagnostic library API.
- Update this document whenever public import paths or transport parameters
  change.
- Generated WAV, packet-bin, NPZ, logs, environments, and model weights must
  not be committed unless they are deliberate small test fixtures.

## 11. Current Cleanup Decision

Retain:

- FSK modem/DSP implementation and public import path.
- FSK receiver implementation and public import path.
- `ReceiverReport`.
- `recover_packets_from_waveform(...)`.
- `recover_packets_from_wav(...)` as a diagnostic API.
- Conferencing readiness and tuning integrations.
- Live Module receiver integration.

Remove:

- Legacy fake sender implementation and wrapper.
- Legacy offline receiver CLI implementation and wrapper.
- Documentation and commands that instruct users to run those retired CLIs.

