from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Optional


@dataclass
class VirtualCameraConfig:
    width: int
    height: int
    fps: float = 15.0
    device: Optional[str] = None


class VirtualCameraPublisher:
    """Thin wrapper around pyvirtualcam with graceful import failure handling."""

    def __init__(self, config: VirtualCameraConfig) -> None:
        self.config = config
        self._cam = None

    def open(self) -> None:
        try:
            import pyvirtualcam  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyvirtualcam is required for virtual camera output. "
                "Install with: pip install pyvirtualcam"
            ) from exc

        self._cam = pyvirtualcam.Camera(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            device=self.config.device,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )

    def send(self, frame_bgr) -> None:
        if self._cam is None:
            raise RuntimeError("Virtual camera is not open")

        self._cam.send(frame_bgr)
        self._cam.sleep_until_next_frame()

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None


def run_virtual_camera_probe(width: int, height: int, fps: float, frames: int) -> dict:
    pub = VirtualCameraPublisher(VirtualCameraConfig(width=width, height=height, fps=fps))
    info = {
        "virtual_cam_ok": False,
        "frames_sent": 0,
        "message": "",
    }

    try:
        np = importlib.import_module("numpy")
        pub.open()
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            value = int((i * 7) % 255)
            frame[:, :, 1] = value
            pub.send(frame)
            info["frames_sent"] = i + 1
        info["virtual_cam_ok"] = True
        info["message"] = "Virtual camera test succeeded."
    except Exception as exc:  # pragma: no cover - depends on local device/runtime
        info["message"] = str(exc)
    finally:
        pub.close()

    return info
