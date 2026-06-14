# 2D Hand Estimation Model Testcase

This project demonstrates 2D hand landmark detection using MediaPipe on:

- A single image input
- A YouTube video stream input

Both pipelines draw hand keypoints and finger/palm connections, then report runtime statistics such as processing time and memory usage.

## Project Goal

- Detect up to 2 hands in each frame/image
- Visualize 21 keypoints per hand
- Connect keypoints to form a hand skeleton
- Monitor runtime metrics for quick performance checks

## Current Folder Structure

```text
2D-Hand-Estimation-Model-Testcase/
|-- image_main.py
|-- video_main.py
|-- README.md
|-- Models/
|   |-- hand_landmarker.task
|   |-- rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
|-- Test_image/
|   |-- test.jpg
```

## What Each File Does

- `image_main.py`
- Runs hand landmark detection on one local image (`Test_image/test.jpg`)
- Draws landmarks and connections
- Shows the result in an OpenCV window
- Prints processing time, loop time, and peak memory usage

- `video_main.py`
- Reads a YouTube video stream using `yt-dlp`
- Runs frame-by-frame hand landmark detection in VIDEO mode
- Draws landmarks, FPS, memory usage, and CPU usage on each frame
- Press `q` to stop playback
- Prints final usage report (time, frames, average FPS, memory)

- `Models/hand_landmarker.task`
- MediaPipe hand landmark model file used by both scripts

- `Models/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth`
- Additional model checkpoint currently not used directly by the two main scripts

- `Test_image/test.jpg`
- Default test image for `image_main.py`

## Requirements

### System

- Windows (current code paths are Windows-style absolute paths)
- Python 3.9 or newer recommended
- Internet connection for YouTube streaming in `video_main.py`

### Python Packages

Install required packages:

```bash
pip install opencv-python mediapipe yt-dlp psutil
```

## Setup Steps

1. Clone or download this project.
2. Ensure model files are inside `Models/`.
3. Ensure test image is inside `Test_image/`.
4. Install Python dependencies.
5. Verify the hardcoded paths in scripts match your local project location.

## How To Run

### 1) Image Inference

Run:

```bash
python image_main.py
```

What to expect:

- A window named `Hand Detection Result` appears
- Hand keypoints and skeleton lines are rendered on the image
- Console prints a usage report

### 2) Video Inference (YouTube)

Run:

```bash
python video_main.py
```

What to expect:

- Script fetches direct stream URL from YouTube
- A window named `YouTube ASL Tracking` appears
- Overlays include:
- Live FPS
- Current memory usage (MB)
- Current CPU usage (%)
- Press `q` to exit
- Console prints final usage report

## Script Workflow Summary

### Common Flow (Both Scripts)

1. Start resource monitoring (`psutil`, `time`)
2. Load MediaPipe model from `Models/hand_landmarker.task`
3. Convert BGR input to RGB
4. Run hand landmark detection
5. Draw keypoints and hand connections
6. Display output using OpenCV
7. Print performance summary

### Detection Modes

- `image_main.py`: uses `detector.detect(...)` for single image mode
- `video_main.py`: uses `detector.detect_for_video(...)` with frame timestamps for video mode

## Hand Landmark Visualization

The scripts draw:

- Landmarks as green circles
- Hand skeleton as blue lines

Connection groups include:

- Thumb chain
- Index chain
- Middle chain
- Ring chain
- Pinky chain
- Palm bridge connections

## Performance Metrics Reported

### Image Script

- Total processing time
- Total loop time
- Peak memory usage

### Video Script

- Total processing time
- Total loop time
- Total frames processed
- Average FPS
- Peak memory usage
- Per-frame CPU usage overlay

## Important Notes

- Current scripts use hardcoded absolute paths like:
- `D:\Project\2D-Hand-Estimation-Model-Testcase\...`
- If your project path is different, update those path strings first.

- The `.pth` file in `Models/` is present but not used by the current code.

- If no hand is detected in a frame/image, scripts still run and display/report normally.

## Troubleshooting

### Issue: "Could not read the image"

- Check that `Test_image/test.jpg` exists
- Confirm the image path in `image_main.py`

### Issue: "Error fetching video URL"

- Check internet connection
- Check YouTube URL validity
- Update `yt-dlp`:

```bash
pip install -U yt-dlp
```

### Issue: OpenCV window does not appear

- Make sure you are not running in a headless environment
- Try running from a local terminal (PowerShell/CMD)

### Issue: Model load failure

- Confirm `Models/hand_landmarker.task` exists
- Confirm path string in code matches your local path

## Suggested Improvements

- Replace hardcoded absolute paths with relative paths
- Add a `requirements.txt`
- Add command-line arguments for input path and YouTube URL
- Save output image/video to a dedicated results folder
- Add logging levels instead of print-only output

## Quick Start Checklist

- [ ] Install dependencies
- [ ] Verify model file path
- [ ] Verify input image path
- [ ] Run `python image_main.py`
- [ ] Run `python video_main.py`

