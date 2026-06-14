import cv2
import yt_dlp
import time
import psutil
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_youtube_url(youtube_url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']
    
# ==================== Monitoring Resources ====================
process = psutil.Process(os.getpid())
total_start_time = time.time()
print("Starting video processing...")
frame_count = 0
max_memory_usage = 0
# ==============================================================

# STEP 1&2: Create an HandLandmarker object.
model_path = 'D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Models\\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input video.
youtube_url = "https://www.youtube.com/watch?v=2Euof4PnjDk"  # Replace with your YouTube video URL
print("Fetching video URL...")
try:
    youtube_video_url = get_youtube_url(youtube_url)
    cap = cv2.VideoCapture(youtube_video_url)
except Exception as e:
    print(f"Error fetching video URL: {e}")
    exit(1)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

print("Video stream opened successfully. Starting hand detection... (Press 'q' to quit)")

# STEP 4: Detect hand landmarks from the input video.

loop_start_time = time.time()
while cap.isOpened():
    frame_start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    hand_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # STEP 5: Process the classification result. In this case, visualize it.
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            points = []
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            
            hand_connections = [
                [0, 1, 2, 3, 4],  # Thumb
                [0, 5, 6, 7, 8],  # Index
                [9, 10, 11, 12],  # Middle
                [13, 14, 15, 16],  # Ring
                [0, 17, 18, 19, 20],   # Pinky
                [5, 9, 13, 17]  # Palm connections
            ]

            for path in hand_connections:
                for i in range(len(path) - 1):
                    start_point = points[path[i]]
                    end_point = points[path[i + 1]]
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    # Monitoring stops here
    frame_end_time = time.time()
    frame_processing_time = frame_end_time - frame_start_time
    current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
    frame_count += 1

    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    max_memory_usage = max(max_memory_usage, memory_usage)

    cpu_usage = process.cpu_percent(interval=None)

    # STEP 6: Display the output video.
    cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory Usage: {memory_usage:.2f} MB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU Usage: {cpu_usage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('YouTube ASL Tracking', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting video stream.")
        break
    
total_end_time = time.time()
total_processing_time = total_end_time - total_start_time
total_loop_time = total_end_time - loop_start_time

print("\n" + "="*20 + " Usage Report " + "="*20)
print(f"Total processing time: {total_processing_time:.2f} seconds")
print(f"Total loop time: {total_loop_time:.2f} seconds")
print(f"Total frames processed: {frame_count}")
if frame_count > 0:
    print(f"Average FPS: {frame_count / total_loop_time:.2f} frames per second")
    print(f"Peak Memory Usage: {max_memory_usage:.2f} MB")
print("="*54)
# Release resources
cap.release()
cv2.destroyAllWindows()