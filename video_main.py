import cv2
import yt_dlp
import time
import psutil
import os
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pose_codec import quantize_all_hands, BYTES_PER_HAND

def get_youtube_url(youtube_url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# ==================== CONFIGURATION FOR BENCHMARK ====================
MODEL_NAME = "MediaPipe_Hands_Video"
model_path = 'D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Models\\hand_landmarker.task'
TXT_FILE_PATH = "D:\\Project\\2D-Hand-Estimation-Model-Testcase\\ytb_urls.txt"
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

# ==================== LOGGER INITIALIZATION ==========================
full_log_path = os.path.join(LOG_DIR, "video_usage.log")
logger = logging.getLogger("VideoUsageLog")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(full_log_path, encoding='utf-8', mode='a')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info(f"Performance logging initialized for model: {MODEL_NAME}")

# ==================== RESOURCE MONITORING INIT =======================
process = psutil.Process(os.getpid())
process.cpu_percent(interval=None)  # Establish baseline reference point

# STEP 1&2: Create an HandLandmarker object (Configured for VIDEO mode).
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)


# ==================== READ YOUTUBE URLS FROM TXT ====================
if not os.path.exists(TXT_FILE_PATH):
    logger.error(f"TXT file not found at: {TXT_FILE_PATH}")
    exit(1)

with open(TXT_FILE_PATH, "r", encoding="utf-8") as f:
    # Read lines, strip whitespaces, and ignore empty lines or comments
    youtube_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

total_videos = len(youtube_urls)
logger.info(f"Loaded {total_videos} video URLs from text file.")


# Outer Loop: Iterate through each YouTube video link
for idx, target_url in enumerate(youtube_urls):
    logger.info(f"\n[Video {idx+1}/{total_videos}] Initializing stream for: {target_url}")
    
    # Reset tracking metrics for the current video
    frame_count = 0
    max_memory_usage = 0
    total_pure_inference_time = 0.0
    
    video_start_time = time.time()
    
    # STEP 3: Load the input video from streaming URL
    try:
        youtube_video_url = get_youtube_url(target_url)
        cap = cv2.VideoCapture(youtube_video_url)
    except Exception as e:
        logger.error(f"Error fetching video URL for {target_url}: {e}")
        continue  # Skip to the next video if streaming link generation fails

    if not cap.isOpened():
        logger.error(f"Error opening video stream for: {target_url}")
        continue

    logger.info("Video stream opened successfully. Starting hand detection...")
    loop_start_time = time.time()
    user_exit = False

    # STEP 4: Inner Loop - Detect hand landmarks frame by frame
    while cap.isOpened():
        frame_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --- Precise Core Loop Timing Starts Here ---
        single_inference_start = time.time()
        
        hand_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        
        single_inference_end = time.time()
        # ---------------------------------------------
        
        total_pure_inference_time += (single_inference_end - single_inference_start)

        # STEP 5: Process the classification result and draw skeleton lines
        quantized_hands = []
        if hand_landmarker_result.hand_landmarks:
            quantized_hands = quantize_all_hands(hand_landmarker_result.hand_landmarks)
            for hand_landmarks in hand_landmarker_result.hand_landmarks:
                points = []
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    points.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                
                hand_connections = [
                    [0, 1, 2, 3, 4],       # Thumb
                    [0, 5, 6, 7, 8],       # Index
                    [9, 10, 11, 12],       # Middle
                    [13, 14, 15, 16],      # Ring
                    [0, 17, 18, 19, 20],   # Pinky
                    [5, 9, 13, 17]         # Palm connections
                ]

                for path in hand_connections:
                    for i in range(len(path) - 1):
                        start_point = points[path[i]]
                        end_point = points[path[i + 1]]
                        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        
        # Calculate single-frame rendering runtime overhead
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
        frame_count += 1

        # Track hardware resource spikes
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        max_memory_usage = max(max_memory_usage, memory_usage)
        cpu_usage = process.cpu_percent(interval=None)

        # STEP 6: Display the output video window
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Memory Usage: {memory_usage:.2f} MB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"CPU Usage: {cpu_usage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Pose payload: {len(quantized_hands) * BYTES_PER_HAND} bytes ({len(quantized_hands)} hand)",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow('YouTube ASL Tracking', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            logger.info("User interrupted current video stream.")
            user_exit = True
            break
        
    # ==================== INDIVIDUAL VIDEO USAGE REPORT ====================
    video_end_time = time.time()
    total_processing_time = video_end_time - video_start_time
    total_loop_time = video_end_time - loop_start_time
    final_cpu_usage = process.cpu_percent(interval=None)

    logger.info("\n" + "="*20 + f" Video {idx+1} Report ({MODEL_NAME}) " + "="*20)
    logger.info(f"Target URL              : {target_url}")
    logger.info(f"Total Run Time (inc. IO): {total_processing_time:.2f} seconds")
    logger.info(f"Total Video Loop Time   : {total_loop_time:.2f} seconds")
    logger.info(f"Pure Model Compute Time : {total_pure_inference_time:.2f} seconds")
    logger.info(f"Total Frames Processed  : {frame_count} frames")
    if total_loop_time > 0:
        logger.info(f"Average System FPS      : {frame_count / total_loop_time:.2f} frames/sec")
    if total_pure_inference_time > 0:
        logger.info(f"Theoretical Model FPS   : {frame_count / total_pure_inference_time:.2f} frames/sec")
    logger.info(f"Peak Memory Usage       : {max_memory_usage:.2f} MB")
    logger.info(f"Final Process CPU Usage : {final_cpu_usage:.1f}%")
    logger.info("="*65 + "\n")

    # Release resources for current capture stream
    cap.release()
    
    # If user pressed 'q' to quit, stop entire batch execution completely
    if user_exit:
        break

# Cleanup global desktop assets
cv2.destroyAllWindows()
detector.close()
logger.info("Batch video benchmark test complete.")