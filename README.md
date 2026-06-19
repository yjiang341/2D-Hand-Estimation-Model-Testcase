# 2D Hand Estimation Model Testcase

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

## Updated Folder Structure

```text
2D-Hand-Estimation-Model-Testcase/
|-- image_main.py
|-- video_main.py
|-- webcam_main.py
|-- pose_codec.py
|-- ytb_urls.txt
|-- README.md
|-- Models/
|   |-- hand_landmarker.task
|   |-- rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
|-- Test_image/
|   |-- test.jpg
|   |-- test2.jpg
|   |-- test3.jpg
|   |-- test4.jpg
|   |-- test5.jpg
|   |-- ...
|-- logs/
|   |-- image_usage.log
|   |-- video_usage.log
```

## File and Folder Roles

### Core Scripts

- `image_main.py`
	- Runs batch image benchmark on all supported image files in `Test_image/`
	- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`
	- Saves visualized outputs into `Result_image/`
	- Writes runtime report to console and `logs/image_usage.log`

- `video_main.py`
	- Reads one or more YouTube URLs from `ytb_urls.txt`
	- Streams each video with `yt-dlp` + OpenCV
	- Runs frame-by-frame hand landmark detection in VIDEO mode
	- Shows live overlay (`FPS`, `Memory`, `CPU`)
	- Writes per-video benchmark report to `logs/video_usage.log`

- `webcam_main.py`
	- Captures webcam input in real time (`cv2.VideoCapture(0)`)
	- Runs MediaPipe hand tracking in VIDEO mode
	- Shows live overlay (`FPS`, `Memory`, `CPU`, `Pose payload bytes`)
	- Writes benchmark report to `logs/webcam_usage.log`

- `pose_codec.py`
	- Converts MediaPipe hand landmarks from `(x, y, z)` to compact `(x, y)` only
	- Uses normalized coordinate quantization: `x_u8 = round(clamp(x, 0, 1) * 255)`
	- Payload size per hand: `21 points × 2 channels = 42 bytes/frame`

### Data and Outputs

- `Models/hand_landmarker.task`
	- MediaPipe hand landmark model used by both scripts

- `Test_image/`
	- Input dataset for image batch benchmark

- `logs/`
	- Persistent benchmark logs for image and video runs

- `ytb_urls.txt`
	- Source list of YouTube links for video batch benchmark
	- Empty lines and lines starting with `#` are ignored

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

1. Make sure model files exist in `Models/`.
2. Put test images in `Test_image/`.
3. Add one or more YouTube URLs to `ytb_urls.txt`.
4. Install required Python packages.
5. Confirm hardcoded absolute paths in scripts match your local machine.

## How To Run

### 1) Batch Image Benchmark

```bash
python image_main.py
```

Expected behavior:

- Detects images from `Test_image/`
- Processes each image independently in IMAGE mode
- Saves each result to `Result_image/result_<original_name>`
- Appends benchmark report to `logs/image_usage.log`

### 2) Batch Video Benchmark (YouTube List)

```bash
python video_main.py
```

Expected behavior:

- Loads URLs from `ytb_urls.txt`
- Processes each video one by one
- Displays real-time inference window (`YouTube ASL Tracking`)
- Shows runtime overlays per frame:
	- FPS
	- Memory usage (MB)
	- CPU usage (%)
- Appends per-video benchmark report to `logs/video_usage.log`

Keyboard behavior:

- Press `q` to stop current stream and end the batch run early

### 3) Live Webcam Benchmark

```bash
python webcam_main.py
```

Expected behavior:

- Opens your default webcam
- Detects up to 2 hands in real time
- Quantizes each hand pose into compact uint8 payloads
- Displays payload size per frame (0, 42, or 84 bytes)
- Press `q` to quit

## Benchmark Metrics Reported

### Image Pipeline (`image_main.py`)

- Total images processed
- Total program runtime (includes file IO and saving)
- Total pure inference time (model-only)
- Average latency per image (ms)
- Throughput (images/sec)
- Peak memory usage (MB)
- Final process CPU usage (%)

### Video Pipeline (`video_main.py`)

- Total runtime per video
- Video loop time
- Pure model compute time
- Total frames processed
- Average system FPS
- Theoretical model FPS
- Peak memory usage (MB)
- Final process CPU usage (%)

### Webcam Pipeline (`webcam_main.py`)

- Total runtime
- Total frames processed
- Average system FPS
- Theoretical model FPS
- Max pose payload size (bytes/frame)
- Peak memory usage (MB)
- Final process CPU usage (%)

## Processing Workflow

### Shared Detection Steps

1. Initialize resource tracking (`psutil`, timers)
2. Load `hand_landmarker.task`
3. Convert BGR frames/images to RGB
4. Run MediaPipe hand landmark detection
5. Draw landmarks and skeleton connections
6. Record metrics and write logs/results

### Running Modes

- `image_main.py` uses `vision.RunningMode.IMAGE` + `detector.detect(...)`
- `video_main.py` uses `vision.RunningMode.VIDEO` + `detector.detect_for_video(...)`
- `webcam_main.py` uses `vision.RunningMode.VIDEO` + `detector.detect_for_video(...)`

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
	- `D:\Project\2D-Hand-Estimation-Model-Testcase\...`
- If your project location is different, update path variables in both scripts.

- Log files are append mode (`mode='a'`), so historical runs are preserved.

- The scripts create required output folders (`logs/`, `Result_image/`) automatically.

## Troubleshooting

### No images processed

- Ensure `Test_image/` contains supported image extensions
- Check `IMAGE_SET_DIR` path in `image_main.py`

### URL list is not loaded

- Ensure `ytb_urls.txt` exists and is readable
- Confirm each URL is on its own line
- Remove accidental leading/trailing spaces if needed

### YouTube stream fetch fails

- Check internet access
- Verify URL availability
- Update `yt-dlp`:

```bash
pip install -U yt-dlp
```

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

