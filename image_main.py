import cv2
import time
import psutil
import os
import logging
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pose_codec import quantize_all_hands, BYTES_PER_HAND

# ==================== CONFIGURATION FOR BENCHMARK ====================
# CHANGE THIS WHEN CHANGING MODELS (e.g., "MediaPipe_Hands", "Lite-HRNet", etc.)
MODEL_NAME = "MediaPipe_Hands_Task"
model_path = 'D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Models\\hand_landmarker.task'  

# Directory setup
IMAGE_SET_DIR = "D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Test_image"
RESULT_SET_DIR = "D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Result_image"
LOG_DIR = "logs"

# Ensure directories exist
os.makedirs(RESULT_SET_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==================== LOGGER INITIALIZATION ==========================
full_log_path = os.path.join(LOG_DIR, "image_usage.log")
logger = logging.getLogger("ImageUsageLog")
logger.setLevel(logging.INFO)

# Prevent duplicating handlers if script runs multiple times in the same session
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
# Call cpu_percent once ahead of time to establish a baseline reference point
process.cpu_percent(interval=None) 
max_memory_usage = 0
# ======================================================================


# STEP 1&2: Create an HandLandmarker object (Configured for single standalone IMAGE mode).
base_options = python.BaseOptions(model_asset_path=model_path)
# explicit running_mode=IMAGE optimization ensures no inter-frame dependencies
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)


# ==================== FIND IMAGES IN IMAGE SET =======================
# Grab all common image types from the specified folder
valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
image_paths = []
for ext in valid_extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_SET_DIR, ext)))

total_images = len(image_paths)
if total_images == 0:
    logger.error(f"No images found in {IMAGE_SET_DIR}. Please check the path.")
    exit(1)

logger.info(f"Found {total_images} images in the dataset. Starting batch benchmark...")

# Start timing the total execution loop
total_start_time = time.time()
processed_count = 0
total_pure_inference_time = 0.0

# STEP 3 & 4: Process image set in a loop
for img_path in image_paths:
    filename = os.path.basename(img_path)
    
    bgr_image = cv2.imread(img_path)
    if bgr_image is None:
        logger.warning(f"Skipping unreadable image: {filename}")
        continue

    h, w, _ = bgr_image.shape
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # --- Precise Core Loop Timing Starts Here ---
    single_inference_start = time.time()
    
    hand_landmarker_result = detector.detect(mp_image)
    
    single_inference_end = time.time()
    # ---------------------------------------------
    
    # Calculate pure processing time for this specific image
    inference_duration = single_inference_end - single_inference_start
    total_pure_inference_time += inference_duration

    # STEP 5: Visualize and draw skeletal structures
    if hand_landmarker_result.hand_landmarks:
        quantized_hands = quantize_all_hands(hand_landmarker_result.hand_landmarks)
        total_payload_bytes = len(quantized_hands) * BYTES_PER_HAND

        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            points = []
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                points.append((cx, cy))
                cv2.circle(bgr_image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            
            hand_connections = [
                [0, 1, 2, 3, 4],       # Thumb
                [0, 5, 6, 7, 8],       # Index Finger
                [9, 10, 11, 12],       # Middle Finger
                [13, 14, 15, 16],      # Ring Finger
                [0, 17, 18, 19, 20],   # Pinky
                [5, 9, 13, 17]         # Palm connections
            ]

            for path in hand_connections:
                for i in range(len(path) - 1):
                    start_point = points[path[i]]
                    end_point = points[path[i + 1]]
                    cv2.line(bgr_image, start_point, end_point, (255, 0, 0), 2)

        cv2.putText(
            bgr_image,
            f"Pose payload: {total_payload_bytes} bytes ({len(quantized_hands)} hand)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # Save output visualization to disk
    output_path = os.path.join(RESULT_SET_DIR, f"result_{filename}")
    cv2.imwrite(output_path, bgr_image)
    
    # Monitor hardware metrics tracking spikes during runtime
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    max_memory_usage = max(max_memory_usage, memory_usage)
    
    processed_count += 1
    logger.info(f"Processed [{processed_count}/{total_images}] -> {filename} | Single Img Time: {inference_duration*1000:.1f} ms")


# ==================== GENERATING PERFORMANCE LOG REPORT ====================
total_end_time = time.time()
total_processing_time = total_end_time - total_start_time
cpu_usage = process.cpu_percent(interval=None)

logger.info("\n" + "="*20 + f" {MODEL_NAME} Usage Report " + "="*20)
logger.info(f"Target Model Tested     : {MODEL_NAME}")
logger.info(f"Total Images Processed  : {processed_count} images")
logger.info(f"Total Program Run Time  : {total_processing_time:.2f} seconds (includes IO/saving)")
logger.info(f"Total Pure Inference Time: {total_pure_inference_time:.2f} seconds (only model work)")
if processed_count > 0:
    avg_ms = (total_pure_inference_time / processed_count) * 1000
    logger.info(f"Average Latency per Img : {avg_ms:.1f} ms")
    logger.info(f"Calculated Throughput   : {processed_count / total_pure_inference_time:.2f} images/sec (FPS)")
logger.info(f"Peak Memory Usage       : {max_memory_usage:.2f} MB")
logger.info(f"Final Process CPU Usage : {cpu_usage:.1f}%")
logger.info("="*60 + "\n")

# Cleanup resources
detector.close()
print("Batch benchmarking test complete. Check your logs/image_usage.log file.")