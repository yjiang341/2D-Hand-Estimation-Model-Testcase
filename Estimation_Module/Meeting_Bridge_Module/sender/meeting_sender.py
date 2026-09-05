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

from Estimation_Module.FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream, write_wav_pcm16
from Estimation_Module.Meeting_Bridge_Module.audio.device_io import open_output_stream
from Estimation_Module.Meeting_Bridge_Module.common.config import BridgeFSKConfig, estimate_packet_tx_ms
from Estimation_Module.Pose_PacketUp.pose_codec import quantize_all_hands
from Estimation_Module.Pose_PacketUp.pose_packet import (
    encode_packet,
    ORIENTATION_UNKNOWN as PACKET_ORIENTATION_UNKNOWN,
    ORIENTATION_PALM as PACKET_ORIENTATION_PALM,
    ORIENTATION_BACK as PACKET_ORIENTATION_BACK,
)

PALM_SCORE_SIGN: float = -1.0
PALM_ORIENTATION_EDGE_THRESHOLD: float = 0.25
PALM_ORIENTATION_EMA_ALPHA: float = 0.50

# Temporal detection stabilizer:
# if a hand is missed for exactly one transmitted frame, reuse its most
# recent transmitted pose + stable orientation instead of sending ABSENT.
# A second consecutive miss drops the hand and resets its temporal state.
HAND_MISS_HOLD_FRAMES: int = 1

ORIENTATION_ABSENT = 0
ORIENTATION_PALM = 1
ORIENTATION_BACK = 2
ORIENTATION_EDGE = 3

ORIENTATION_CODE = {
    "ABSENT": ORIENTATION_ABSENT,
    "PALM": ORIENTATION_PALM,
    "BACK": ORIENTATION_BACK,
    "EDGE": ORIENTATION_EDGE,
}


@dataclass(frozen=True)
class PalmOrientationEstimate:
    orientation: str
    score: float
    normal_xyz: tuple[float, float, float]
    source: str


def estimate_palm_orientation_3d(
    landmarks,
    handedness: str,
    *,
    source: str,
) -> PalmOrientationEstimate:
    """
    Estimate PALM / BACK / EDGE from a 3D MediaPipe hand.

    Geometry:
        wrist      = landmark 0
        index MCP  = landmark 5
        pinky MCP  = landmark 17

        palm_normal = cross(index_mcp - wrist, pinky_mcp - wrist)

    The z component of the UNIT normal is used as a camera-facing score.
    Right/Left hands have opposite intrinsic chirality, so Left is multiplied
    by -1 to place both hands in a common orientation convention.

    Important:
        The final PALM-vs-BACK sign depends on MediaPipe/camera coordinate
        convention. Calibrate PALM_SCORE_SIGN once with an obvious open palm.
    """

    if landmarks is None or len(landmarks) < 21:
        raise ValueError("Expected 21 MediaPipe hand landmarks")

    xyz = np.asarray(
        [[float(lm.x), float(lm.y), float(lm.z)] for lm in landmarks],
        dtype=np.float64,
    )

    wrist = xyz[0]
    index_mcp = xyz[5]
    pinky_mcp = xyz[17]

    v_index = index_mcp - wrist
    v_pinky = pinky_mcp - wrist

    normal = np.cross(v_index, v_pinky)
    normal_norm = float(np.linalg.norm(normal))

    if normal_norm <= 1e-12:
        return PalmOrientationEstimate(
            orientation="EDGE",
            score=0.0,
            normal_xyz=(0.0, 0.0, 0.0),
            source=source,
        )

    unit_normal = normal / normal_norm

    if handedness == "Right":
        handedness_sign = +1.0
    elif handedness == "Left":
        handedness_sign = -1.0
    else:
        handedness_sign = +1.0

    score = float(
        PALM_SCORE_SIGN
        * handedness_sign
        * unit_normal[2]
    )

    if abs(score) < PALM_ORIENTATION_EDGE_THRESHOLD:
        orientation = "EDGE"
    elif score > 0.0:
        orientation = "PALM"
    else:
        orientation = "BACK"

    return PalmOrientationEstimate(
        orientation=orientation,
        score=score,
        normal_xyz=(
            float(unit_normal[0]),
            float(unit_normal[1]),
            float(unit_normal[2]),
        ),
        source=source,
    )


def classify_smoothed_orientation(
    score: float,
    previous_stable: str,
) -> str:
    """
    Hysteresis-like behavior for video:
      - face-on scores become PALM/BACK
      - edge-on scores keep the previous stable state when available
    """

    if abs(score) < PALM_ORIENTATION_EDGE_THRESHOLD:
        if previous_stable in ("PALM", "BACK"):
            return previous_stable
        return "EDGE"

    return "PALM" if score > 0.0 else "BACK"


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
    local_wav_copy_dir: str = r"D:\Project\2D-Hand\Estimation_Module\Estimation_result_media"
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
        raw_pose_frames: list[np.ndarray] = []
        raw_pose_frame_ids: list[int] = []

        # Local diagnostics. Orientation is transmitted in protocol v3,
        # while the additional masks below let us distinguish true MediaPipe
        # detections from one-frame temporal holds.
        raw_orientation_scores: list[np.ndarray] = []
        raw_orientation_ema_scores: list[np.ndarray] = []
        raw_orientation_labels: list[np.ndarray] = []
        raw_orientation_stable_labels: list[np.ndarray] = []
        raw_detected_hand_masks: list[np.ndarray] = []
        raw_held_hand_masks: list[np.ndarray] = []
        raw_tx_hand_present_masks: list[np.ndarray] = []
        raw_packet_orientations: list[np.ndarray] = []

        orientation_ema = {"Right": None, "Left": None}
        orientation_stable = {"Right": "EDGE", "Left": "EDGE"}

        # Per-hand continuity cache. These values are updated only from a
        # genuine MediaPipe detection; HOLD frames never become new history.
        last_quantized_hand = {"Right": None, "Left": None}
        last_packet_orientation = {
            "Right": PACKET_ORIENTATION_UNKNOWN,
            "Left": PACKET_ORIENTATION_UNKNOWN,
        }
        miss_streak = {"Right": 0, "Left": 0}

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

                quantized_hands = [None, None]
                raw_keypoints = np.zeros((42, 2), dtype=np.float32)

                # Protocol v3 orientation metadata, same fixed slot convention:
                #   slot 0 = RIGHT
                #   slot 1 = LEFT
                #
                # Only the temporally stable PALM/BACK state is transmitted.
                # EDGE/unknown stays UNKNOWN on wire.
                packet_orientations = [
                    PACKET_ORIENTATION_UNKNOWN,
                    PACKET_ORIENTATION_UNKNOWN,
                ]

                orientation_scores_frame = np.full(2, np.nan, dtype=np.float32)
                orientation_ema_scores_frame = np.full(2, np.nan, dtype=np.float32)
                orientation_labels_frame = np.zeros(2, dtype=np.uint8)
                orientation_stable_labels_frame = np.zeros(2, dtype=np.uint8)

                # Slot convention for every diagnostic array:
                #   [0] = RIGHT, [1] = LEFT
                detected_hand_mask_frame = np.zeros(2, dtype=np.uint8)
                held_hand_mask_frame = np.zeros(2, dtype=np.uint8)
                slot_sources = ["ABSENT", "ABSENT"]

                # MediaPipe can occasionally return two candidates with the
                # same handedness label in one frame. We first choose only the
                # highest-confidence candidate per side, so a hand's EMA is
                # never updated twice in one transmitted frame.
                best_candidates = {"Right": None, "Left": None}
                candidate_counts = {"Right": 0, "Left": 0}

                if result.hand_landmarks:
                    for i, landmarks in enumerate(result.hand_landmarks[:2]):
                        if (
                            not result.handedness
                            or i >= len(result.handedness)
                            or not result.handedness[i]
                        ):
                            continue

                        category = result.handedness[i][0]
                        handedness = str(category.category_name)

                        if handedness not in ("Right", "Left"):
                            continue

                        handedness_confidence = float(
                            getattr(category, "score", 0.0)
                        )

                        candidate_counts[handedness] += 1

                        current_best = best_candidates[handedness]
                        if (
                            current_best is None
                            or handedness_confidence > current_best["confidence"]
                        ):
                            best_candidates[handedness] = {
                                "index": i,
                                "landmarks": landmarks,
                                "confidence": handedness_confidence,
                            }

                # Process at most one RIGHT and one LEFT candidate.
                for handedness in ("Right", "Left"):
                    candidate = best_candidates[handedness]
                    count = candidate_counts[handedness]

                    if count > 1 and candidate is not None:
                        print(
                            f"[dedup] frame={frame_id} "
                            f"hand={handedness} "
                            f"candidates={count} "
                            f"selected_index={candidate['index']} "
                            f"confidence={candidate['confidence']:.3f} "
                            f"dropped={count - 1}"
                        )

                    if candidate is None:
                        continue

                    i = int(candidate["index"])
                    landmarks = candidate["landmarks"]
                    handedness_confidence = float(candidate["confidence"])

                    orientation_slot = 0 if handedness == "Right" else 1

                    # A real detection ends any miss streak.
                    miss_streak[handedness] = 0
                    detected_hand_mask_frame[orientation_slot] = 1
                    slot_sources[orientation_slot] = "DETECTED"

                    # Prefer MediaPipe world-space 3D landmarks.
                    world_sets = getattr(result, "hand_world_landmarks", None)
                    if (
                        world_sets
                        and i < len(world_sets)
                        and world_sets[i]
                    ):
                        orientation_landmarks = world_sets[i]
                        orientation_source = "world"
                    else:
                        orientation_landmarks = landmarks
                        orientation_source = "normalized"

                    orientation = estimate_palm_orientation_3d(
                        orientation_landmarks,
                        handedness,
                        source=orientation_source,
                    )

                    prev_ema = orientation_ema.get(handedness)
                    if prev_ema is None:
                        ema_score = orientation.score
                    else:
                        alpha = PALM_ORIENTATION_EMA_ALPHA
                        ema_score = (
                            alpha * orientation.score
                            + (1.0 - alpha) * float(prev_ema)
                        )

                    orientation_ema[handedness] = ema_score

                    stable_label = classify_smoothed_orientation(
                        ema_score,
                        orientation_stable.get(handedness, "EDGE"),
                    )

                    if stable_label in ("PALM", "BACK"):
                        orientation_stable[handedness] = stable_label

                    orientation_scores_frame[orientation_slot] = orientation.score
                    orientation_ema_scores_frame[orientation_slot] = ema_score
                    orientation_labels_frame[orientation_slot] = ORIENTATION_CODE[
                        orientation.orientation
                    ]
                    orientation_stable_labels_frame[orientation_slot] = ORIENTATION_CODE[
                        stable_label
                    ]

                    if stable_label == "PALM":
                        packet_orientations[orientation_slot] = PACKET_ORIENTATION_PALM
                    elif stable_label == "BACK":
                        packet_orientations[orientation_slot] = PACKET_ORIENTATION_BACK
                    else:
                        packet_orientations[orientation_slot] = PACKET_ORIENTATION_UNKNOWN

                    print(
                        f"[orientation] frame={frame_id} "
                        f"hand={handedness} "
                        f"instant={orientation.orientation} "
                        f"stable={stable_label} "
                        f"score={orientation.score:+.3f} "
                        f"ema={ema_score:+.3f} "
                        f"normal=({orientation.normal_xyz[0]:+.3f},"
                        f"{orientation.normal_xyz[1]:+.3f},"
                        f"{orientation.normal_xyz[2]:+.3f}) "
                        f"source={orientation.source}"
                    )

                    quantized_hand = quantize_all_hands([landmarks])[0]
                    quantized_hands[orientation_slot] = quantized_hand

                    keypoint_offset = 0 if handedness == "Right" else 21
                    for j, lm in enumerate(landmarks):
                        raw_keypoints[keypoint_offset + j, 0] = lm.x * 256.0
                        raw_keypoints[keypoint_offset + j, 1] = lm.y * 256.0

                    # Update continuity cache only from a true detection.
                    last_quantized_hand[handedness] = quantized_hand
                    last_packet_orientation[handedness] = packet_orientations[
                        orientation_slot
                    ]

                    print(
                        f"frame={frame_id} "
                        f"mediapipe_index={i} "
                        f"handedness={handedness} "
                        f"confidence={handedness_confidence:.3f}"
                    )

                # One-frame miss hold:
                # reuse the previous transmitted pose/orientation for one
                # missing transmitted frame; on the second consecutive miss,
                # send ABSENT and reset that side's temporal state.
                for handedness, orientation_slot in (("Right", 0), ("Left", 1)):
                    if detected_hand_mask_frame[orientation_slot]:
                        continue

                    had_history = last_quantized_hand[handedness] is not None
                    if not had_history:
                        continue

                    miss_streak[handedness] += 1

                    if miss_streak[handedness] <= HAND_MISS_HOLD_FRAMES:
                        quantized_hands[orientation_slot] = last_quantized_hand[
                            handedness
                        ]
                        packet_orientations[orientation_slot] = (
                            last_packet_orientation[handedness]
                        )
                        held_hand_mask_frame[orientation_slot] = 1
                        slot_sources[orientation_slot] = "HOLD"

                        prev_ema = orientation_ema.get(handedness)
                        if prev_ema is not None:
                            orientation_ema_scores_frame[orientation_slot] = float(
                                prev_ema
                            )

                        stable_label = orientation_stable.get(handedness, "EDGE")
                        orientation_stable_labels_frame[orientation_slot] = (
                            ORIENTATION_CODE.get(
                                stable_label,
                                ORIENTATION_ABSENT,
                            )
                        )

                        print(
                            f"[hold] frame={frame_id} "
                            f"hand={handedness} "
                            f"miss_streak={miss_streak[handedness]} "
                            f"action=reuse_previous_pose_and_orientation"
                        )
                    else:
                        print(
                            f"[drop] frame={frame_id} "
                            f"hand={handedness} "
                            f"miss_streak={miss_streak[handedness]} "
                            f"action=send_absent_and_reset"
                        )

                        last_quantized_hand[handedness] = None
                        last_packet_orientation[handedness] = (
                            PACKET_ORIENTATION_UNKNOWN
                        )
                        orientation_ema[handedness] = None
                        orientation_stable[handedness] = "EDGE"
                        miss_streak[handedness] = 0

                def _packet_orientation_name(value: int) -> str:
                    if value == PACKET_ORIENTATION_PALM:
                        return "PALM"
                    if value == PACKET_ORIENTATION_BACK:
                        return "BACK"
                    return "UNKNOWN"

                tx_hand_present_mask_frame = np.asarray(
                    [
                        int(quantized_hands[0] is not None),
                        int(quantized_hands[1] is not None),
                    ],
                    dtype=np.uint8,
                )

                print(
                    f"[slots] frame={frame_id} "
                    f"RIGHT={'Y' if quantized_hands[0] is not None else 'N'} "
                    f"R_SRC={slot_sources[0]} "
                    f"R_ORI={_packet_orientation_name(packet_orientations[0])} "
                    f"LEFT={'Y' if quantized_hands[1] is not None else 'N'} "
                    f"L_SRC={slot_sources[1]} "
                    f"L_ORI={_packet_orientation_name(packet_orientations[1])}"
                )

                raw_pose_frames.append(raw_keypoints.copy())
                raw_pose_frame_ids.append(frame_id)
                raw_orientation_scores.append(orientation_scores_frame.copy())
                raw_orientation_ema_scores.append(orientation_ema_scores_frame.copy())
                raw_orientation_labels.append(orientation_labels_frame.copy())
                raw_orientation_stable_labels.append(orientation_stable_labels_frame.copy())
                raw_detected_hand_masks.append(detected_hand_mask_frame.copy())
                raw_held_hand_masks.append(held_hand_mask_frame.copy())
                raw_tx_hand_present_masks.append(tx_hand_present_mask_frame.copy())
                raw_packet_orientations.append(
                    np.asarray(packet_orientations, dtype=np.uint8)
                )

                packet = encode_packet(
                    frame_id=frame_id,
                    timestamp_ms=capture_timestamp_ms,
                    hands=quantized_hands,
                    orientations=packet_orientations,
                )
                wave_chunk = modulate_packet_stream([packet], session.modem_cfg).astype(np.float32, copy=False)

                session.stream.write(wave_chunk.reshape(-1, 1))
                if self.cfg.save_local_wav_copy:
                    local_wave_chunks.append(wave_chunk.copy())

                sent += 1
                frame_id += 1
                last_tx_s = time.perf_counter()

                present_hand_count = sum(
                    hand is not None
                    for hand in quantized_hands
                )

                if self.cfg.show_preview:
                    cv2.putText(
                        frame,
                        f"sent={sent} est_tx={session.est_tx_ms:.1f}ms hands={present_hand_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                         2,
                    )
                    cv2.putText(
                        frame,
                        f"R:{orientation_stable['Right']} L:{orientation_stable['Left']}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
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
                if raw_pose_frames:
                    raw_pose_path = os.path.join(
                    self.cfg.local_wav_copy_dir,
                    "raw_mediapipe_pose.npz",
                    )

                    np.savez(
                        raw_pose_path,
                        keypoints=np.stack(raw_pose_frames, axis=0),
                        frame_ids=np.asarray(raw_pose_frame_ids, dtype=np.int32),
                        orientation_score=np.stack(raw_orientation_scores, axis=0),
                        orientation_ema_score=np.stack(raw_orientation_ema_scores, axis=0),
                        orientation_label=np.stack(raw_orientation_labels, axis=0),
                        orientation_stable_label=np.stack(raw_orientation_stable_labels, axis=0),
                        orientation_label_names=np.asarray(
                            ["ABSENT", "PALM", "BACK", "EDGE"]
                        ),
                        detected_hand_mask=np.stack(raw_detected_hand_masks, axis=0),
                        held_hand_mask=np.stack(raw_held_hand_masks, axis=0),
                        tx_hand_present_mask=np.stack(raw_tx_hand_present_masks, axis=0),
                        packet_orientation=np.stack(raw_packet_orientations, axis=0),
                        packet_orientation_names=np.asarray(
                            ["UNKNOWN", "PALM", "BACK"]
                        ),
                        hand_miss_hold_frames=np.asarray(
                            HAND_MISS_HOLD_FRAMES,
                            dtype=np.int32,
                        ),
                    )

                    _emit(
                        f"sender raw MediaPipe pose saved: {raw_pose_path}"
                    )
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
