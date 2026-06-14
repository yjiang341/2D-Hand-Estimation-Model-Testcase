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

# ==============================================================

# STEP 1&2: Create an HandLandmarker object.
model_path = 'D:\\Project\\ASL\\hand_landmarker.task'
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
while cap.isOpened():
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

    # STEP 6: Display the output video.
    cv2.imshow('YouTube ASL Tracking', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting video stream.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()