from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from Conferencing_Module.virtual.virtual_camera import VirtualCameraConfig, VirtualCameraPublisher
from FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream
from FSK_Module.fsk_receiver import ReceiverReport, recover_packets_from_waveform
from Pose_PacketUp.pose_codec import quantize_all_hands, quantized_hand_to_xy_pairs
from Pose_PacketUp.pose_packet import HAND_SLOT_COUNT, PosePacket, encode_packet
from Pose_PacketUp.pose_render import RenderConfig, render_skeleton_frame


@dataclass
class TxWaveItem:
    waveform: np.ndarray
    enqueue_time_s: float


@dataclass
class LiveStats:
    captured_frames: int = 0
    enqueued_frames: int = 0
    dropped_frames: int = 0
    queue_depth_max: int = 0
    queue_depth_sum: int = 0
    queue_depth_samples: int = 0
    receiver_preamble_candidates: int = 0
    receiver_attempted: int = 0
    receiver_valid: int = 0
    receiver_rejected: int = 0
    decoded_frames: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_samples: int = 0
    transport_sum_ms: float = 0.0
    transport_max_ms: float = 0.0
    transport_samples: int = 0


class StreamingPoseFilter:
    """Incremental sanity constraint + EMA smoothing for realtime frames."""

    def __init__(self, max_step: float, ema_alpha: float) -> None:
        if max_step <= 0.0:
            raise ValueError("max_step must be > 0")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")

        self.max_step = max_step
        self.ema_alpha = ema_alpha
        self._prev_constrained: list[Optional[np.ndarray]] = [None] * HAND_SLOT_COUNT
        self._prev_smoothed: list[Optional[np.ndarray]] = [None] * HAND_SLOT_COUNT

    def apply(
        self,
        hand_present: np.ndarray,
        hand_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        out_present = hand_present.astype(np.uint8, copy=True)
        out_xy = np.zeros_like(hand_xy, dtype=np.float32)

        for slot in range(HAND_SLOT_COUNT):
            if int(hand_present[slot]) == 0:
                self._prev_constrained[slot] = None
                self._prev_smoothed[slot] = None
                continue

            cur = np.clip(hand_xy[slot].astype(np.float32, copy=True), 0.0, 1.0)
            prev = self._prev_constrained[slot]

            if prev is not None:
                delta = cur - prev
                dist = np.linalg.norm(delta, axis=1, keepdims=True)
                scale = np.minimum(1.0, self.max_step / np.maximum(dist, 1e-8))
                cur = prev + delta * scale

            cur = np.clip(cur, 0.0, 1.0)
            self._prev_constrained[slot] = cur.copy()

            prev_smoothed = self._prev_smoothed[slot]
            if prev_smoothed is None:
                smoothed = cur
            else:
                smoothed = self.ema_alpha * cur + (1.0 - self.ema_alpha) * prev_smoothed

            smoothed = np.clip(smoothed, 0.0, 1.0)
            self._prev_smoothed[slot] = smoothed.copy()
            out_xy[slot] = smoothed

        return out_present, out_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="live webcam sender->receiver loop with queue and latency metrics"
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/esc")
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--rx-delay-frames", type=int, default=1)
    parser.add_argument("--detection-threshold", type=float, default=0.55)
    parser.add_argument("--ema-alpha", type=float, default=0.65)
    parser.add_argument("--max-step", type=float, default=0.20)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--symbol-rate", type=int, default=1_200)
    parser.add_argument("--freq0", type=float, default=1_200.0)
    parser.add_argument("--freq1", type=float, default=2_200.0)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--silence-ms", type=int, default=3)
    parser.add_argument("--render-fps", type=float, default=15.0)
    parser.add_argument("--model-path", type=str, default=os.path.join("Models", "hand_landmarker.task"))
    parser.add_argument("--log-path", type=str, default=os.path.join("logs", "live_usage.log"))
    parser.add_argument("--output-mode", choices=["display", "virtual-cam", "both", "headless"], default="display")
    parser.add_argument("--vcam-device", type=str, default=None)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    logger = logging.getLogger("Live")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _packet_to_render_rows(packet: PosePacket) -> tuple[np.ndarray, np.ndarray]:
    hand_present = np.zeros((HAND_SLOT_COUNT,), dtype=np.uint8)
    hand_xy = np.zeros((HAND_SLOT_COUNT, 21, 2), dtype=np.float32)

    for slot in range(HAND_SLOT_COUNT):
        hand = packet.hands[slot]
        if hand is None:
            continue
        hand_present[slot] = 1
        hand_xy[slot] = np.array(quantized_hand_to_xy_pairs(hand), dtype=np.float32)

    return hand_present, hand_xy


def _accumulate_receiver_report(stats: LiveStats, report: ReceiverReport) -> None:
    stats.receiver_preamble_candidates += report.preamble_candidates
    stats.receiver_attempted += report.attempted_frames
    stats.receiver_valid += report.valid_frames
    stats.receiver_rejected += report.rejected_frames


def _drain_one_wave(
    tx_queue: Deque[TxWaveItem],
    fsk_cfg: FSKConfig,
    render_cfg: RenderConfig,
    pose_filter: StreamingPoseFilter,
    stats: LiveStats,
    detection_threshold: float,
    display: bool,
) -> Optional[np.ndarray]:
    if not tx_queue:
        return None

    item = tx_queue.popleft()
    report = recover_packets_from_waveform(item.waveform, fsk_cfg, detection_threshold=detection_threshold)
    _accumulate_receiver_report(stats, report)

    now_s = time.perf_counter()
    transport_ms = (now_s - item.enqueue_time_s) * 1000.0
    stats.transport_sum_ms += transport_ms
    stats.transport_max_ms = max(stats.transport_max_ms, transport_ms)
    stats.transport_samples += 1

    if not report.valid_packets:
        return None

    packet = report.valid_packets[-1]
    hand_present, hand_xy = _packet_to_render_rows(packet)
    hand_present, hand_xy = pose_filter.apply(hand_present, hand_xy)

    rx_frame = render_skeleton_frame(
        hand_present_row=hand_present,
        hand_xy_row=hand_xy,
        frame_id=packet.frame_id,
        timestamp_ms=packet.timestamp_ms,
        config=render_cfg,
    )

    stats.decoded_frames += 1
    e2e_latency_ms = max(0.0, float(int(time.time() * 1000) - int(packet.timestamp_ms)))
    stats.latency_sum_ms += e2e_latency_ms
    stats.latency_max_ms = max(stats.latency_max_ms, e2e_latency_ms)
    stats.latency_samples += 1

    if display:
        cv2.putText(
            rx_frame,
            f"E2E latency: {e2e_latency_ms:.1f} ms",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            render_cfg.text_color_bgr,
            2,
        )

    return rx_frame


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_path)

    if args.queue_capacity <= 0:
        raise ValueError("queue-capacity must be > 0")
    if args.rx_delay_frames < 0:
        raise ValueError("rx-delay-frames must be >= 0")

    fsk_cfg = FSKConfig(
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        freq0_hz=args.freq0,
        freq1_hz=args.freq1,
        amplitude=args.amplitude,
        inter_frame_silence_ms=args.silence_ms,
    )
    render_cfg = RenderConfig(width=args.width, height=args.height, fps=args.render_fps)
    pose_filter = StreamingPoseFilter(max_step=args.max_step, ema_alpha=args.ema_alpha)
    show_windows = args.output_mode in ("display", "both") and bool(args.display)
    publish_virtual_cam = args.output_mode in ("virtual-cam", "both")

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Unable to open webcam id={args.camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    virtual_cam: Optional[VirtualCameraPublisher] = None
    if publish_virtual_cam:
        virtual_cam = VirtualCameraPublisher(
            VirtualCameraConfig(
                width=render_cfg.width,
                height=render_cfg.height,
                fps=render_cfg.fps,
                device=args.vcam_device,
            )
        )
        virtual_cam.open()

    stats = LiveStats()
    tx_queue: Deque[TxWaveItem] = deque(maxlen=args.queue_capacity)
    frame_id = 0
    start_s = time.perf_counter()

    logger.info("=== Live Loop Started ===")
    logger.info(
        "camera=%d size=%dx%d queue_capacity=%d rx_delay=%d symbol_rate=%d sample_rate=%d",
        args.camera_id,
        args.width,
        args.height,
        args.queue_capacity,
        args.rx_delay_frames,
        args.symbol_rate,
        args.sample_rate,
    )

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                logger.warning("Webcam frame read failed; stopping loop.")
                break

            stats.captured_frames += 1
            capture_timestamp_ms = int(time.time() * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            hand_landmarker_result = detector.detect_for_video(mp_image, capture_timestamp_ms)

            quantized_hands = []
            if hand_landmarker_result.hand_landmarks:
                quantized_hands = quantize_all_hands(hand_landmarker_result.hand_landmarks[:2])

            packet = encode_packet(
                frame_id=frame_id,
                timestamp_ms=capture_timestamp_ms,
                hands=quantized_hands,
            )

            wave_chunk = modulate_packet_stream([packet], fsk_cfg)

            if len(tx_queue) == args.queue_capacity:
                tx_queue.popleft()
                stats.dropped_frames += 1

            tx_queue.append(TxWaveItem(waveform=wave_chunk, enqueue_time_s=time.perf_counter()))
            stats.enqueued_frames += 1
            stats.queue_depth_max = max(stats.queue_depth_max, len(tx_queue))
            stats.queue_depth_sum += len(tx_queue)
            stats.queue_depth_samples += 1

            rx_frame: Optional[np.ndarray] = None
            if len(tx_queue) > args.rx_delay_frames:
                rx_frame = _drain_one_wave(
                    tx_queue=tx_queue,
                    fsk_cfg=fsk_cfg,
                    render_cfg=render_cfg,
                    pose_filter=pose_filter,
                    stats=stats,
                    detection_threshold=args.detection_threshold,
                    display=show_windows,
                )

            if publish_virtual_cam and rx_frame is not None and virtual_cam is not None:
                virtual_cam.send(rx_frame)

            if show_windows:
                cv2.putText(
                    frame,
                    f"TX queue: {len(tx_queue)}/{args.queue_capacity}  dropped: {stats.dropped_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Decoded: {stats.decoded_frames}  Valid/Attempted: {stats.receiver_valid}/{stats.receiver_attempted}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("TX Webcam", frame)
                if rx_frame is not None:
                    cv2.imshow("RX Skeleton", rx_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("User requested stop via keyboard.")
                    break

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                logger.info("Reached max-frames=%d; stopping.", args.max_frames)
                break

        while len(tx_queue) > 0:
            _drain_one_wave(
                tx_queue=tx_queue,
                fsk_cfg=fsk_cfg,
                render_cfg=render_cfg,
                pose_filter=pose_filter,
                stats=stats,
                detection_threshold=args.detection_threshold,
                display=False,
            )

    finally:
        cap.release()
        detector.close()
        if virtual_cam is not None:
            virtual_cam.close()
        cv2.destroyAllWindows()

    run_s = max(1e-6, time.perf_counter() - start_s)
    avg_queue_depth = (
        float(stats.queue_depth_sum) / float(stats.queue_depth_samples)
        if stats.queue_depth_samples > 0
        else 0.0
    )
    avg_latency_ms = (
        stats.latency_sum_ms / stats.latency_samples if stats.latency_samples > 0 else 0.0
    )
    avg_transport_ms = (
        stats.transport_sum_ms / stats.transport_samples
        if stats.transport_samples > 0
        else 0.0
    )

    logger.info("=== Live Loop Report ===")
    logger.info("Total runtime (s)                 : %.2f", run_s)
    logger.info("Captured frames                   : %d", stats.captured_frames)
    logger.info("Enqueued frames                   : %d", stats.enqueued_frames)
    logger.info("Dropped (queue overflow)          : %d", stats.dropped_frames)
    logger.info("Queue depth avg/max               : %.2f / %d", avg_queue_depth, stats.queue_depth_max)
    logger.info("Receiver preamble candidates      : %d", stats.receiver_preamble_candidates)
    logger.info("Receiver attempted/valid/rejected : %d / %d / %d", stats.receiver_attempted, stats.receiver_valid, stats.receiver_rejected)
    logger.info("Decoded frames                    : %d", stats.decoded_frames)
    logger.info("Throughput capture/decode FPS     : %.2f / %.2f", stats.captured_frames / run_s, stats.decoded_frames / run_s)
    logger.info("Transport latency avg/max (ms)    : %.2f / %.2f", avg_transport_ms, stats.transport_max_ms)
    logger.info("End-to-end latency avg/max (ms)   : %.2f / %.2f", avg_latency_ms, stats.latency_max_ms)
    logger.info("Log path                          : %s", args.log_path)


if __name__ == "__main__":
    main()
