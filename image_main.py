import cv2
import time
import psutil
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================== Monitoring Resources ====================
process = psutil.Process(os.getpid())
total_start_time = time.time()
print("Starting image processing...")
max_memory_usage = 0
# ==============================================================

# STEP 1&2: Create an HandLandmarker object.
model_path = 'D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Models\\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
bgr_image = cv2.imread("D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Test_image\\test.jpg")
if bgr_image is None:
    raise FileNotFoundError("Could not read the image. Please check the path and file name.")

h, w, _ = bgr_image.shape

rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# STEP 4: Detect hand landmarks from the input image.
# Perform hand landmarks detection on the provided single image.
# The hand landmarker must be created with the image mode.
hand_landmarker_result = detector.detect(mp_image)

# STEP 5: Process the classification result. In this case, visualize it.
if hand_landmarker_result.hand_landmarks:
    loop_start_time = time.time()
    for hand_landmarks in hand_landmarker_result.hand_landmarks:
        points = []
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            points.append((cx, cy))
            cv2.circle(bgr_image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        
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
                cv2.line(bgr_image, start_point, end_point, (255, 0, 0), 2)

# Monitoring stops here
memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
max_memory_usage = max(max_memory_usage, memory_usage)
cpu_usage = process.cpu_percent(interval=None)

# STEP 6: Display the output image.
cv2.imshow('Hand Detection Result', bgr_image)

# Optionally, remove the "#" on next line to save the output image to disk
#cv2.imwrite('D:\\Project\\2D-Hand-Estimation-Model-Testcase\\Result_image\\result.jpg', bgr_image)

print('Detection complete. Press any key to exit.')
cv2.waitKey(0)

total_end_time = time.time()
total_processing_time = total_end_time - total_start_time
total_loop_time = time.time() - loop_start_time

print("\n" + "="*20 + " Usage Report " + "="*20)
print(f"Total processing time: {total_processing_time:.2f} seconds")
print(f"Total loop time: {total_loop_time:.2f} seconds")
print(f"Peak Memory Usage: {max_memory_usage:.2f} MB")
print("="*54)

cv2.destroyAllWindows()