import cv2
import time
import psutil
import os
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_codec import BYTES_PER_HAND, quantize_all_hands


MODEL_NAME = "MediaPipe_Hands_Webcam"
model_path = "D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Models\\hand_landmarker.task"
LOG_DIR = "logs"

# Webcam settings
CAMERA_ID = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30

os.makedirs(LOG_DIR, exist_ok=True)

full_log_path = os.path.join(LOG_DIR, "webcam_usage.log")
logger = logging.getLogger("WebcamUsageLog")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(full_log_path, encoding="utf-8", mode="a")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info(f"Performance logging initialized for model: {MODEL_NAME}")

process = psutil.Process(os.getpid())
process.cpu_percent(interval=None)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    logger.error("Unable to open webcam device.")
    detector.close()
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

logger.info(
    f"Webcam opened: id={CAMERA_ID}, "
    f"resolution={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
    f"requested_fps={TARGET_FPS}"
)

frame_count = 0
max_memory_usage = 0.0
max_payload_bytes = 0
total_pure_inference_time = 0.0
loop_start_time = time.time()

hand_connections = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [0, 17, 18, 19, 20],
    [5, 9, 13, 17],
]

while cap.isOpened():
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        logger.warning("Webcam frame read failed. Stopping capture loop.")
        break

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms <= 0:
        timestamp_ms = int((time.time() - loop_start_time) * 1000)

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    infer_start = time.time()
    result = detector.detect_for_video(mp_image, timestamp_ms)
    infer_end = time.time()
    total_pure_inference_time += infer_end - infer_start

    quantized_hands = []
    if result.hand_landmarks:
        quantized_hands = quantize_all_hands(result.hand_landmarks)
        for hand_landmarks in result.hand_landmarks:
            points = []
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            for path in hand_connections:
                for idx in range(len(path) - 1):
                    cv2.line(frame, points[path[idx]], points[path[idx + 1]], (255, 0, 0), 2)

    payload_bytes = len(quantized_hands) * BYTES_PER_HAND
    max_payload_bytes = max(max_payload_bytes, payload_bytes)

    frame_processing_time = time.time() - frame_start_time
    frame_count += 1
    current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0.0

    memory_usage = process.memory_info().rss / (1024 * 1024)
    max_memory_usage = max(max_memory_usage, memory_usage)
    cpu_usage = process.cpu_percent(interval=None)

    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory Usage: {memory_usage:.2f} MB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU Usage: {cpu_usage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Pose payload: {payload_bytes} bytes ({len(quantized_hands)} hand)",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Webcam ASL Tracking", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        logger.info("User interrupted webcam stream.")
        break

run_time = time.time() - loop_start_time
final_cpu_usage = process.cpu_percent(interval=None)

logger.info("\n" + "=" * 20 + f" Webcam Report ({MODEL_NAME}) " + "=" * 20)
logger.info(f"Total Run Time          : {run_time:.2f} seconds")
logger.info(f"Total Frames Processed  : {frame_count} frames")
if run_time > 0:
    logger.info(f"Average System FPS      : {frame_count / run_time:.2f} frames/sec")
if total_pure_inference_time > 0:
    logger.info(f"Theoretical Model FPS   : {frame_count / total_pure_inference_time:.2f} frames/sec")
logger.info(f"Max Pose Payload Size   : {max_payload_bytes} bytes/frame")
logger.info(f"Peak Memory Usage       : {max_memory_usage:.2f} MB")
logger.info(f"Final Process CPU Usage : {final_cpu_usage:.1f}%")
logger.info("=" * 62 + "\n")

cap.release()
cv2.destroyAllWindows()
detector.close()
logger.info("Webcam benchmark test complete.")
