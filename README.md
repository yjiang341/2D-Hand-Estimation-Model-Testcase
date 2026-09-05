# 2D-Hand

A test pipeline that includes:
Near real-time 2D hand-pose estimation, FSK transport, skeleton reconstruction,
and controlled hand-image generation

## Overview

This repository contains two connected research components:

1. `Estimation_Module` captures hand landmarks with 2D hand-pose estimation, quantizes them
	 into pose packets, transports them through BFSK audio modulation, and
	 reconstructs the received 2D skeleton.
2. `Generation_Module` contains the controlled hand-image genration model source integration and the
	 current non-notebook Image2Video baseline.

The intended end-to-end pipeline is:

```text
Webcam
	-> 2D hand-pose estimation hand landmarks
	-> canonical RIGHT/LEFT 21-point slots
	-> Pose Packet
	-> FSK audio transport
	-> packet recovery and pose smoothing
	-> controlled hand-image generation model conditioning adapter
	-> generated hand frames
```

## Repository Layout (As Sep 4th, 2026)

```text
.
|-- Estimation_Module/
|   |-- FSK_Module/              # BFSK modulation, demodulation, receiver APIs
|   |-- Meeting_Bridge_Module/   # Audio sender/receiver bridge
|   |-- Conferencing_Module/     # Channel simulation and readiness checks
|   |-- Live_Module/             # Webcam -> packet -> FSK -> skeleton loop
|   |-- Pose_PacketUp/           # Packet codec, reconstruction, rendering
|   |-- Models/                  # MediaPipe hand landmark model location
|-- Generation_Module/
|   |-- FoundHand/               # Pinned FoundHand source
|   |-- scripts/                 # Baseline and reusable runner scripts
|   `-- .venv/                   # Local plan and agent execution metadata
`-- README.md
```

Large runtime assets and generated outputs are intentionally not described as
source files here. In particular, local environments, model weights, build
directories, generated outputs, some test data, and Git metadata.

## Requirements

### Estimation and transport (As Sep 4th, 2026)

- Windows is the primary development environment.
- Python 3.9 or newer is recommended.
- MediaPipe Tasks hand landmarker model at
	`Estimation_Module/Models/hand_landmarker.task`.
- Core runtime dependencies include OpenCV, NumPy, MediaPipe, psutil, and
	sounddevice. `pyvirtualcam` is optional for virtual-camera output.

### FoundHand generation (As Sep 4th, 2026)

The validated local FoundHand environment uses:

- Python 3.9.25
- PyTorch 2.3.0 with CUDA 12.1
- Torchvision 0.18.0 with CUDA 12.1
- Lightning 2.3.0
- timm 1.0.7
- NumPy 1.26.4
- MediaPipe 1.0.0
- OpenCV 5.0.0
- Segment Anything 1.0
- An NVIDIA GPU with a compatible driver

FoundHand also requires three local checkpoints. Their paths and sizes are
recorded in `Generation_Module/reports/environment_report.md`; the files are
not part of the normal source checkout.

## Installation

Create or activate the environment appropriate for the component you are
running. 

The editable install is provided by `Generation_Module/FoundHand/setup.py`.
For a normal Python environment, run `pip install -e .` from inside
`Generation_Module/FoundHand`.

The estimation environment can be installed with the project dependencies used
by the active scripts:

```powershell
python -m pip install opencv-python mediapipe psutil sounddevice
```

Install `pyvirtualcam` only when using virtual-camera output:

```powershell
python -m pip install pyvirtualcam
```

## Running Estimation

Run commands from the repository root so package imports and relative paths
resolve consistently.

### Live webcam, FSK, and skeleton reconstruction

```powershell
python Estimation_Module/live_main.py --output-mode display
```

Useful variants:

```powershell
python Estimation_Module/live_main.py --output-mode headless --max-frames 100
python Estimation_Module/live_main.py --output-mode virtual-cam
python Estimation_Module/live_main.py --output-mode both
```

Press `q` or `Esc` to stop display-mode runs. The live pipeline supports queue
capacity, receiver delay, EMA smoothing, FSK sample rate, symbol rate, and
detection threshold options.

### Integrated meeting bridge

List audio devices:

```powershell
python Estimation_Module/Meeting_Bridge_Module/bridge_main.py --mode list-devices
```

Run sender, receiver, or decode a saved WAV using `--mode sender`,
`--mode receiver`, or `--mode decode-wav`. Use `--help` to view the complete
device and output configuration.

### Pose reconstruction and rendering

```powershell
python Estimation_Module/Pose_PacketUp/pose_reconstruct_main.py
python Estimation_Module/Pose_PacketUp/pose_render_main.py --mode mp4
```

The reconstruction tools use packet bytes as input and write normalized pose
streams or rendered skeleton video to the configured output paths.

## Running FoundHand Baseline

The original FoundHand demo is an interactive notebook. The repository also
contains a behavior-preserving standalone baseline:

```powershell
& "C:\Users\.conda\envs\foundhand\python.exe" `
	Generation_Module/scripts/run_image2video_baseline.py `
	--idx IMG_1087 `
	--start-frame 0 `
	--max-frames 5 `
	--cfg-scale 2.5
```

The script loads the DiT, VAE, and SAM checkpoints, converts the supplied
42-point pose sequence into FoundHand heatmaps, samples generated frames, and
writes an AVI preview. Output is written under:

```text
Generation_Module/outputs/image2video_original/
|-- frames/
|-- trajectory_vis.jpg
|-- ref_frame_0000.jpg
|-- IMG_1087.avi
`-- execution.log
```

