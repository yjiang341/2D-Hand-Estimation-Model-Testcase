from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions

from FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream, write_wav_pcm16
from Meeting_Bridge_Module.audio.device_io import open_output_stream
from Meeting_Bridge_Module.common.config import BridgeFSKConfig, estimate_packet_tx_ms
from Pose_PacketUp.pose_codec import quantize_all_hands
from Pose_PacketUp.pose_packet import encode_packet


@dataclass(frozen=True)
class MeetingSenderConfig:
    model_path: str
    camera_id: int = 0
    width: int = 1280
    height: int = 720
    tx_fps: float = 1.8
    max_frames: int = 0
    show_preview: bool = True
    audio_output_device: Optional[str] = None
    audio_output_fallback: bool = True
    camera_backend: str = "auto"
    save_local_wav_copy: bool = True
    local_wav_copy_dir: str = "local_sender_copy"
    local_wav_copy_path: Optional[str] = None
    warmup_enabled: bool = False
    warmup_frame_tries: int = 5


@dataclass
class PreparedSenderSession:
    cfg: MeetingSenderConfig
    fsk_cfg: BridgeFSKConfig
    model_path: str
    detector: HandLandmarker
    cap: cv2.VideoCapture
    stream: object
    modem_cfg: FSKConfig
    est_tx_ms: float
    min_period_s: float

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.detector.close()
        except Exception:
            pass
        try:
            self.stream.stop()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass


class MeetingSender:
    def __init__(self, cfg: MeetingSenderConfig, fsk_cfg: BridgeFSKConfig) -> None:
        self.cfg = cfg
        self.fsk_cfg = fsk_cfg

    def run(
        self,
        stop_event=None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _emit(msg: str) -> None:
            if status_callback is not None:
                status_callback(msg)

        session = self.prepare_session(status_callback=status_callback)
        try:
            self.run_with_session(session, stop_event=stop_event, status_callback=status_callback)
        finally:
            session.close()

    def prepare_session(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> PreparedSenderSession:
        def _emit(msg: str) -> None:
            if status_callback is not None:
                status_callback(msg)

        startup_t0 = time.perf_counter()
        model_path = self._resolve_model_path(self.cfg.model_path)

        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=VisionTaskRunningMode.VIDEO,
        )
        detector = HandLandmarker.create_from_options(options)
        _emit(f"sender detector ready startup_ms={(time.perf_counter() - startup_t0) * 1000.0:.1f}")

        backend = str(self.cfg.camera_backend or "auto").strip().lower()
        cap = None
        if backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(self.cfg.camera_id, cv2.CAP_DSHOW)
        elif backend == "msmf" and hasattr(cv2, "CAP_MSMF"):
            cap = cv2.VideoCapture(self.cfg.camera_id, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(self.cfg.camera_id)

        if not cap.isOpened() and backend in ("dshow", "msmf"):
            cap.release()
            cap = cv2.VideoCapture(self.cfg.camera_id)

        if not cap.isOpened():
            detector.close()
            raise RuntimeError(f"Unable to open webcam id={self.cfg.camera_id}")
        _emit(f"sender webcam opened startup_ms={(time.perf_counter() - startup_t0) * 1000.0:.1f}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)

        stream = open_output_stream(
            self.fsk_cfg.sample_rate,
            device=self.cfg.audio_output_device,
            fallback_to_default=bool(self.cfg.audio_output_fallback),
        )
        try:
            stream.start()
        except Exception:
            cap.release()
            detector.close()
            try:
                stream.close()
            except Exception:
                pass
            raise

        if self.cfg.warmup_enabled:
            _emit("sender warmup started")
            warmed = False
            max_tries = max(1, int(self.cfg.warmup_frame_tries))
            for _ in range(max_tries):
                ok, warm_frame = cap.read()
                if not ok:
                    continue
                capture_timestamp_ms = int(time.time() * 1000)
                rgb_frame = cv2.cvtColor(warm_frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                detector.detect_for_video(mp_image, capture_timestamp_ms)
                warmed = True
                break
            _emit("sender warmup finished" if warmed else "sender warmup skipped (no frame)")

        modem_cfg = FSKConfig(
            sample_rate=self.fsk_cfg.sample_rate,
            symbol_rate=self.fsk_cfg.symbol_rate,
            freq0_hz=self.fsk_cfg.freq0_hz,
            freq1_hz=self.fsk_cfg.freq1_hz,
            amplitude=self.fsk_cfg.amplitude,
            inter_frame_silence_ms=self.fsk_cfg.silence_ms,
        )

        min_period_s = 1.0 / max(1e-6, self.cfg.tx_fps)
        est_tx_ms = estimate_packet_tx_ms(
            symbol_rate=self.fsk_cfg.symbol_rate,
            silence_ms=self.fsk_cfg.silence_ms,
        )

        _emit(f"sender startup complete startup_ms={(time.perf_counter() - startup_t0) * 1000.0:.1f}")

        return PreparedSenderSession(
            cfg=self.cfg,
            fsk_cfg=self.fsk_cfg,
            model_path=model_path,
            detector=detector,
            cap=cap,
            stream=stream,
            modem_cfg=modem_cfg,
            est_tx_ms=est_tx_ms,
            min_period_s=min_period_s,
        )

    def run_with_session(
        self,
        session: PreparedSenderSession,
        stop_event=None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _emit(msg: str) -> None:
            if status_callback is not None:
                status_callback(msg)

        frame_id = 0
        sent = 0
        last_tx_s = 0.0
        local_wave_chunks: list[np.ndarray] = []

        _emit("sender started")

        try:
            while session.cap.isOpened():
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    _emit("sender stop requested")
                    break

                ok, frame = session.cap.read()
                if not ok:
                    _emit("webcam frame read failed")
                    break

                now_s = time.perf_counter()
                if (now_s - last_tx_s) < session.min_period_s:
                    if self.cfg.show_preview:
                        cv2.putText(
                            frame,
                            f"TX waiting... target_fps={self.cfg.tx_fps:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        cv2.imshow("Bridge Sender", frame)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break
                    continue

                capture_timestamp_ms = int(time.time() * 1000)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                result = session.detector.detect_for_video(mp_image, capture_timestamp_ms)

                quantized_hands = []
                if result.hand_landmarks:
                    quantized_hands = quantize_all_hands(result.hand_landmarks[:2])

                packet = encode_packet(
                    frame_id=frame_id,
                    timestamp_ms=capture_timestamp_ms,
                    hands=quantized_hands,
                )
                wave_chunk = modulate_packet_stream([packet], session.modem_cfg).astype(np.float32, copy=False)

                session.stream.write(wave_chunk.reshape(-1, 1))
                if self.cfg.save_local_wav_copy:
                    local_wave_chunks.append(wave_chunk.copy())

                sent += 1
                frame_id += 1
                last_tx_s = time.perf_counter()

                if self.cfg.show_preview:
                    cv2.putText(
                        frame,
                        f"sent={sent} est_tx={session.est_tx_ms:.1f}ms hands={len(quantized_hands)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Bridge Sender", frame)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        _emit("sender stopped by keyboard")
                        break

                if self.cfg.max_frames > 0 and sent >= self.cfg.max_frames:
                    _emit(f"sender reached max-frames={self.cfg.max_frames}")
                    break
        finally:
            cv2.destroyAllWindows()
            if self.cfg.save_local_wav_copy and local_wave_chunks:
                os.makedirs(self.cfg.local_wav_copy_dir, exist_ok=True)
                out_wav = self.cfg.local_wav_copy_path
                if not out_wav:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_wav = os.path.join(self.cfg.local_wav_copy_dir, f"sender_capture_{stamp}.wav")
                merged_wave = np.concatenate(local_wave_chunks)
                write_wav_pcm16(out_wav, merged_wave, session.modem_cfg.sample_rate)
                _emit(f"sender local wav copy saved: {out_wav}")
            _emit(f"sender finished, sent_frames={sent}")

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path

        candidates: list[str] = []
        if os.path.isabs(model_path):
            candidates.append(model_path)
        else:
            candidates.append(os.path.abspath(os.path.join(os.getcwd(), model_path)))
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.abspath(os.path.join(exe_dir, model_path)))

            meipass = getattr(sys, "_MEIPASS", None)
            if isinstance(meipass, str) and meipass:
                candidates.append(os.path.abspath(os.path.join(meipass, model_path)))

            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidates.append(os.path.abspath(os.path.join(repo_root, model_path)))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        tried = "\n".join(f"- {path}" for path in candidates)
        raise FileNotFoundError(
            "Unable to open file at "
            f"{model_path}. Tried:\n{tried}"
        )
