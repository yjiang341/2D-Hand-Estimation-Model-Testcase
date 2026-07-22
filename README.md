# 2D-Hand-Estimation-Render


This project benchmarks 2D hand landmark estimation with MediaPipe in three pipelines:

- Image batch inference from a local dataset folder
- Video batch inference from YouTube URLs listed in a text file
- Live webcam inference for real-time tracking

Both pipelines log runtime and system-resource metrics for quick comparison and reproducibility.

## Project Objectives

- Detect up to 2 hands per frame/image
- Visualize 21 keypoints per hand
- Draw a hand skeleton (fingers + palm links)
- Record benchmark-friendly metrics (latency, throughput/FPS, memory, CPU)
- Convert visual ASL hand pose into sound wave on the sender side, then rendering the sound wave back to Visualize 2D estimate hand pose

## Folder Structure

```text
2D-Hand-Estimation-Model-Testcase/
|-- README.md
|-- webcam_main.py
|-- live_main.py
|-- readiness_main.py
|-- bridge_main.py
|-- Models/
|   |-- hand_landmarker.task
|-- Test_image/
|-- Result_image/
|-- result_video/
|-- local_sender_copy/
|-- logs/
|-- Live_Module/
|-- FSK_Module/
|-- Conferencing_Module/
|-- Meeting_Bridge_Module/
|-- Pose_PacketUp/
```

## File and Folder Roles

### Core Scripts

- `webcam_main.py`
	- Captures webcam input in real time (`cv2.VideoCapture(0)`)
	- Runs MediaPipe hand tracking in VIDEO mode
	- Shows live overlay (`FPS`, `Memory`, `CPU`, `Pose payload bytes`)
	- Writes benchmark report to `logs/webcam_usage.log`

- `live_main.py`
	- Captures webcam frames and quantizes up to 2 hands into pose packets
	- Runs live in-memory BFSK sender and receiver per frame
	- Reconstructs and smooths decoded pose stream in real time

- `pose_codec.py`
	- Converts MediaPipe hand landmarks from `(x, y, z)` to compact `(x, y)` only
	- Uses normalized coordinate quantization: `x_u8 = round(clamp(x, 0, 1) * 255)`
	- Payload size per hand: `21 points × 2 channels = 42 bytes/frame`

### Used Model

- `Models/hand_landmarker.task`
	- MediaPipe hand landmark model used by both scripts

## Environment Requirements

### OS and Runtime

- Windows (scripts currently use Windows absolute paths)
- Python 3.9+ recommended
- Internet connection required for YouTube video benchmarking

### Python Dependencies

```bash
pip install opencv-python mediapipe yt-dlp psutil
```

## Setup Guide

1. Install required Python packages.
2. Confirm hardcoded absolute paths in scripts match your local machine.

## How To Run

### 1) Live Sender/Receiver

```bash
python live_main.py --queue-capacity 8 --rx-delay-frames 1 --output-mode display
```

Live RX bridge to virtual camera (OBS-style virtual device):

```bash
python live_main.py --output-mode virtual-cam --max-frames 0
```

Show windows and publish virtual camera simultaneously:

```bash
python live_main.py --output-mode both
```

Expected behavior:

- Captures webcam frames and quantizes up to 2 hands into pose packets
- Runs live in-memory BFSK sender and receiver per frame
- Reconstructs and smooths decoded pose stream in real time
- Shows TX webcam and RX skeleton windows simultaneously
- Can publish RX skeleton frames directly to virtual camera in real time
- Reports queue depth, dropped frames, receiver validity, and latency metrics
- Writes a runtime report to `logs/live_usage.log`
- Press `q` or `esc` to stop

### 2) Conferencing Readiness (Auto Envrionment Configuration)

```bash
python readiness_main.py --mode sweep
```

Optional virtual camera probe:

```bash
python readiness_main.py --mode both
```

Preset profiles (Choose one of three):

```bash
python readiness_main.py --mode sweep --profile high-reliability
python readiness_main.py --mode sweep --profile balanced
python readiness_main.py --mode sweep --profile low-latency
```

Expected behavior:

- Runs a modular FSK parameter sweep under synthetic conferencing-style channel impairments
- Measures frame loss and CRC rejection rate across candidate modem settings
- Auto-selects a winner by preset target profile (`high-reliability`, `balanced`, `low-latency`)
- Produces fallback recommendation when channel quality degrades
- Optionally probes virtual camera output path (requires `pyvirtualcam`)
- Saves structured report to `logs/readiness_report.json`

### 3) Integrated sender/receiver Bridge

This bridge is designed for teams where each participant runs this project locally.

List local audio devices first:

```bash
python bridge_main.py --mode list-devices
```

Sender side (webcam -> pose -> BFSK audio output device):

```bash
python bridge_main.py --mode sender --audio-output-device "[YOUR AUDIO OUPUT DEVICE]" --tx-fps 1.8
```

Sender side with explicit local WAV copy location:

```bash
python bridge_main.py --mode sender --audio-output-device "[YOUR AUDIO OUPUT DEVICE]" --local-wav-copy-dir local_sender_copy
```

Receiver side (audio input device -> decode -> skeleton -> virtual camera):

```bash
python bridge_main.py --mode receiver --audio-input-device "[YOUR AUDIO INPUT DEVICE]" --publish-virtual-cam --display
```

Offline decode with timestamp-aligned hold behavior:

```bash
python bridge_main.py --mode decode-wav --in-wav local_sender_copy/sender_capture_YYYYMMDD_HHMMSS.wav --result-video-dir result_video --timestamp-timing --timestamp-max-hold-ms 2500
```

Expected behavior:

- Sender captures webcam hand pose and modulates packets to a selected audio output device
- Sender also stores a local WAV copy at `local_sender_copy/` by default for reproducible offline validation
- Receiver captures a selected audio input stream, demodulates and decodes packets in near-real-time
- Receiver renders skeleton preview and can publish to virtual camera for OBS/meeting app selection
- Offline mode can decode a saved WAV and render reconstructed skeleton video to `result_video/*.mp4`
- Offline decode supports timestamp-based frame hold to better match original gesture timing and pauses
- Workflow is API-independent from Zoom/Google Meet; meeting apps act as transport surfaces

## Processing Workflow

### Shared Detection Steps

1. Initialize resource tracking (`psutil`, timers)
2. Load `hand_landmarker.task`
3. Convert BGR frames/images to RGB
4. Run MediaPipe hand landmark detection
5. Draw landmarks and skeleton connections
6. Record metrics and write logs/results

### Running Modes

## Landmark Quantization Details

- Original MediaPipe per-landmark data: `(x, y, z)`
- This project now keeps only `(x, y)` for transmission
- Coordinates are clamped to `[0.0, 1.0]` and mapped to uint8 `[0, 255]`

Formula:

```python
x_int = int(round(max(0.0, min(1.0, x)) * 255))
y_int = int(round(max(0.0, min(1.0, y)) * 255))
```

Bandwidth estimate (single hand):

- `21 × 2 = 42 bytes/frame`
- At 15 FPS: `42 × 15 = 630 bytes/sec`

## Landmark Drawing Details

- Green circles: keypoints
- Blue lines: skeletal connections
- Connection sets:
	- Thumb
	- Index finger
	- Middle finger
	- Ring finger
	- Pinky
	- Palm bridge

## Important Notes

- Current paths are hardcoded to:
	- `D:\Project\2D-Hand-Estimation-Render\...`
- If your project location is different, update path variables in both scripts.

- Log files are append mode (`mode='a'`), so historical runs are preserved.

- The scripts create required output folders (`logs/`, `Result_image/`) automatically.

## Troubleshooting

### OpenCV window does not show

- Run from local desktop terminal (not headless environment)
- Ensure GUI/OpenCV display support is available

### Model load error

- Confirm `Models/hand_landmarker.task` exists
- Confirm model path string in scripts matches your local path

## Quick Checklist

- [ ] Install dependencies
- [ ] Verify model path
- [ ] Verify image dataset path
- [ ] Verify YouTube URL list file path
- [ ] Run `python image_main.py`
- [ ] Run `python video_main.py`
- [ ] Run `python webcam_main.py`

