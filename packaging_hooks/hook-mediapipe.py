from PyInstaller.utils.hooks import collect_submodules

# Keep only the runtime pieces used by HandLandmarker.
datas = []
binaries = []

hiddenimports = [
    "mediapipe.tasks.c",
    "mediapipe.tasks.python.core.base_options",
    "mediapipe.tasks.python.vision.core.vision_task_running_mode",
    "mediapipe.tasks.python.vision.hand_landmarker",
]

for package_name in (
    "mediapipe.tasks.python.core",
    "mediapipe.tasks.python.components",
    "mediapipe.tasks.python.vision.core",
):
    hiddenimports += collect_submodules(package_name)
