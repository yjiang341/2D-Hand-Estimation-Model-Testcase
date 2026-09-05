from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional, Set

import cv2
import numpy as np

from Estimation_Module.Conferencing_Module.virtual.virtual_camera import VirtualCameraConfig, VirtualCameraPublisher
from Estimation_Module.FSK_Module.fsk_modem import FSKConfig, demodulate_packet_stream, read_wav_pcm16
from Estimation_Module.Meeting_Bridge_Module.audio.device_io import open_input_stream
from Estimation_Module.Meeting_Bridge_Module.common.config import BridgeFSKConfig, BridgeRenderConfig
from Estimation_Module.Pose_PacketUp.pose_codec import quantized_hand_to_xy_pairs
from Estimation_Module.Pose_PacketUp.pose_packet import HAND_SLOT_COUNT, PacketDecodeError, decode_packet
from Estimation_Module.Pose_PacketUp.pose_render import RenderConfig, render_skeleton_frame


@dataclass(frozen=True)
class MeetingReceiverConfig:
    audio_input_device: Optional[str] = None
    chunk_ms: int = 40
    max_buffer_seconds: float = 8.0
    display: bool = True
    publish_virtual_cam: bool = True
    vcam_device: Optional[str] = None


class MeetingReceiver:
    def __init__(self, cfg: MeetingReceiverConfig, fsk_cfg: BridgeFSKConfig, render_cfg: BridgeRenderConfig) -> None:
        self.cfg = cfg
        self.fsk_cfg = fsk_cfg
        self.render_cfg = render_cfg

    def run(
        self,
        stop_event=None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _emit(msg: str) -> None:
            if status_callback is not None:
                status_callback(msg)

        modem_cfg = FSKConfig(
            sample_rate=self.fsk_cfg.sample_rate,
            symbol_rate=self.fsk_cfg.symbol_rate,
            freq0_hz=self.fsk_cfg.freq0_hz,
            freq1_hz=self.fsk_cfg.freq1_hz,
            amplitude=self.fsk_cfg.amplitude,
            inter_frame_silence_ms=self.fsk_cfg.silence_ms,
        )
        render = RenderConfig(width=self.render_cfg.width, height=self.render_cfg.height, fps=self.render_cfg.fps)

        chunk_samples = max(64, int(self.fsk_cfg.sample_rate * self.cfg.chunk_ms / 1000.0))
        max_samples = int(self.fsk_cfg.sample_rate * self.cfg.max_buffer_seconds)

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        audio_buffer = np.zeros((0,), dtype=np.float32)
        seen_frame_ids: Set[int] = set()

        def _audio_callback(indata, _frames, _time, _status) -> None:
            mono = np.asarray(indata[:, 0], dtype=np.float32)
            audio_queue.put(mono.copy())

        virtual_cam: Optional[VirtualCameraPublisher] = None
        if self.cfg.publish_virtual_cam:
            virtual_cam = VirtualCameraPublisher(
                VirtualCameraConfig(
                    width=render.width,
                    height=render.height,
                    fps=render.fps,
                    device=self.cfg.vcam_device,
                )
            )
            virtual_cam.open()

        processed = 0
        _emit("receiver started")

        with open_input_stream(
            sample_rate=self.fsk_cfg.sample_rate,
            blocksize=chunk_samples,
            callback=_audio_callback,
            device=self.cfg.audio_input_device,
        ):
            try:
                while True:
                    if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                        _emit("receiver stop requested")
                        break

                    drained = 0
                    while True:
                        try:
                            chunk = audio_queue.get_nowait()
                        except queue.Empty:
                            break
                        drained += 1
                        audio_buffer = np.concatenate([audio_buffer, chunk])

                    if drained == 0:
                        time.sleep(0.01)
                        if self.cfg.display:
                            key = cv2.waitKey(1) & 0xFF
                            if key in (ord("q"), 27):
                                _emit("receiver stopped by keyboard")
                                break
                        continue

                    if audio_buffer.size > max_samples:
                        audio_buffer = audio_buffer[-max_samples:]

                    packets, _stats = demodulate_packet_stream(
                        waveform=audio_buffer,
                        config=modem_cfg,
                        packet_size=104,
                        auto_align=True,
                    )

                    for raw in packets:
                        try:
                            packet = decode_packet(raw)
                        except PacketDecodeError:
                            continue

                        fid = int(packet.frame_id)
                        if fid in seen_frame_ids:
                            continue
                        seen_frame_ids.add(fid)
                        if len(seen_frame_ids) > 4096:
                            seen_frame_ids = set(sorted(seen_frame_ids)[-2048:])

                        hand_present = np.zeros((HAND_SLOT_COUNT,), dtype=np.uint8)
                        hand_xy = np.zeros((HAND_SLOT_COUNT, 21, 2), dtype=np.float32)

                        for slot in range(HAND_SLOT_COUNT):
                            hand = packet.hands[slot]
                            if hand is None:
                                continue
                            hand_present[slot] = 1
                            hand_xy[slot] = np.array(quantized_hand_to_xy_pairs(hand), dtype=np.float32)

                        frame = render_skeleton_frame(
                            hand_present_row=hand_present,
                            hand_xy_row=hand_xy,
                            frame_id=fid,
                            timestamp_ms=int(packet.timestamp_ms),
                            config=render,
                        )

                        e2e_ms = max(0.0, float(int(time.time() * 1000) - int(packet.timestamp_ms)))
                        cv2.putText(
                            frame,
                            f"bridge rx frame={fid} e2e={e2e_ms:.1f}ms",
                            (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            render.text_color_bgr,
                            2,
                        )

                        if virtual_cam is not None:
                            virtual_cam.send(frame)
                        if self.cfg.display:
                            cv2.imshow("Bridge Receiver", frame)

                        processed += 1

                    if self.cfg.display:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            _emit("receiver stopped by keyboard")
                            break
            finally:
                if virtual_cam is not None:
                    virtual_cam.close()
                cv2.destroyAllWindows()
                _emit(f"receiver finished, processed_frames={processed}")


def decode_wav_to_video(
    wav_path: str,
    out_video_path: str,
    fsk_cfg: BridgeFSKConfig,
    render_cfg: BridgeRenderConfig,
    use_timestamp_timing: bool = True,
    max_hold_ms: int = 2500,
) -> int:
    waveform, wav_sample_rate = read_wav_pcm16(wav_path)

    modem_cfg = FSKConfig(
        sample_rate=int(wav_sample_rate),
        symbol_rate=fsk_cfg.symbol_rate,
        freq0_hz=fsk_cfg.freq0_hz,
        freq1_hz=fsk_cfg.freq1_hz,
        amplitude=fsk_cfg.amplitude,
        inter_frame_silence_ms=fsk_cfg.silence_ms,
    )
    render = RenderConfig(width=render_cfg.width, height=render_cfg.height, fps=render_cfg.fps)

    packets, _stats = demodulate_packet_stream(
        waveform=waveform,
        config=modem_cfg,
        packet_size=104,
        auto_align=True,
    )

    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(render.fps),
        (render.width, render.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {out_video_path}")

    rendered = 0
    seen_frame_ids: Set[int] = set()
    prev_frame: Optional[np.ndarray] = None
    prev_ts_ms: Optional[int] = None
    try:
        for raw in packets:
            try:
                packet = decode_packet(raw)
            except PacketDecodeError:
                continue

            fid = int(packet.frame_id)
            if fid in seen_frame_ids:
                continue
            seen_frame_ids.add(fid)

            hand_present = np.zeros((HAND_SLOT_COUNT,), dtype=np.uint8)
            hand_xy = np.zeros((HAND_SLOT_COUNT, 21, 2), dtype=np.float32)
            for slot in range(HAND_SLOT_COUNT):
                hand = packet.hands[slot]
                if hand is None:
                    continue
                hand_present[slot] = 1
                hand_xy[slot] = np.array(quantized_hand_to_xy_pairs(hand), dtype=np.float32)

            frame = render_skeleton_frame(
                hand_present_row=hand_present,
                hand_xy_row=hand_xy,
                frame_id=fid,
                timestamp_ms=int(packet.timestamp_ms),
                config=render,
            )
            cv2.putText(
                frame,
                f"offline decode frame={fid}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                render.text_color_bgr,
                2,
            )

            ts_ms = int(packet.timestamp_ms)
            if prev_frame is None:
                prev_frame = frame
                prev_ts_ms = ts_ms
                continue

            repeats = 1
            if use_timestamp_timing and prev_ts_ms is not None:
                delta_ms = ts_ms - prev_ts_ms
                if delta_ms > 0:
                    clamped_ms = min(int(delta_ms), int(max_hold_ms)) if max_hold_ms > 0 else int(delta_ms)
                    repeats = max(1, int(round(clamped_ms * float(render.fps) / 1000.0)))

            for _ in range(repeats):
                writer.write(prev_frame)
            rendered += repeats

            prev_frame = frame
            prev_ts_ms = ts_ms

        if prev_frame is not None:
            writer.write(prev_frame)
            rendered += 1
    finally:
        writer.release()

    return rendered
