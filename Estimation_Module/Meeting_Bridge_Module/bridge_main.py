from __future__ import annotations

import argparse
import os
import time

from Estimation_Module.Meeting_Bridge_Module.common.config import BridgeFSKConfig, BridgeRenderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated local meeting bridge (sender/receiver/device listing)"
    )
    parser.add_argument("--mode", choices=["list-devices", "sender", "receiver", "decode-wav"], required=True)

    parser.add_argument("--model-path", type=str, default=os.path.join("Models", "hand_landmarker.task"))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tx-fps", type=float, default=1.8)
    parser.add_argument("--show-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-backend", choices=["auto", "dshow", "msmf"], default=("dshow" if os.name == "nt" else "auto"))
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-frame-tries", type=int, default=5)

    parser.add_argument("--audio-output-device", type=str, default=None)
    parser.add_argument("--audio-output-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-input-device", type=str, default=None)
    parser.add_argument("--save-local-wav-copy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-wav-copy-dir", type=str, default="local_sender_copy")
    parser.add_argument("--local-wav-copy-path", type=str, default=None)

    parser.add_argument("--in-wav", type=str, default=None)
    parser.add_argument("--result-video-dir", type=str, default="result_video")
    parser.add_argument("--out-video", type=str, default=None)
    parser.add_argument("--timestamp-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp-max-hold-ms", type=int, default=2500)

    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--symbol-rate", type=int, default=1_600)
    parser.add_argument("--freq0", type=float, default=1_200.0)
    parser.add_argument("--freq1", type=float, default=2_200.0)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--silence-ms", type=int, default=2)
    parser.add_argument("--detect-threshold", type=float, default=0.55)

    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--max-buffer-seconds", type=float, default=8.0)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--publish-virtual-cam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcam-device", type=str, default=None)

    parser.add_argument("--render-fps", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd_t0 = time.perf_counter()

    if args.mode == "list-devices":
        from Estimation_Module.Meeting_Bridge_Module.audio.device_io import list_audio_devices

        devices = list_audio_devices()
        print("=== Audio Devices ===")
        for dev in devices:
            print(
                f"[{dev.index}] {dev.name} | in={dev.max_input_channels} out={dev.max_output_channels} "
                f"default_sr={dev.default_samplerate:.0f}"
            )
        return

    fsk_cfg = BridgeFSKConfig(
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        freq0_hz=args.freq0,
        freq1_hz=args.freq1,
        amplitude=args.amplitude,
        silence_ms=args.silence_ms,
        detection_threshold=args.detect_threshold,
    )

    if args.mode == "sender":
        from Estimation_Module.Meeting_Bridge_Module.sender.meeting_sender import MeetingSender, MeetingSenderConfig

        sender_cfg = MeetingSenderConfig(
            model_path=args.model_path,
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            tx_fps=args.tx_fps,
            max_frames=args.max_frames,
            show_preview=args.show_preview,
            audio_output_device=args.audio_output_device,
            audio_output_fallback=args.audio_output_fallback,
            camera_backend=args.camera_backend,
            save_local_wav_copy=args.save_local_wav_copy,
            local_wav_copy_dir=args.local_wav_copy_dir,
            local_wav_copy_path=args.local_wav_copy_path,
            warmup_enabled=args.warmup,
            warmup_frame_tries=args.warmup_frame_tries,
        )

        def _emit(msg: str) -> None:
            print(f"[sender] t={(time.perf_counter() - cmd_t0) * 1000.0:.1f}ms {msg}")

        MeetingSender(sender_cfg, fsk_cfg).run(status_callback=_emit)
        return

    if args.mode == "decode-wav":
        from Estimation_Module.Meeting_Bridge_Module.receiver.meeting_receiver import decode_wav_to_video

        if not args.in_wav:
            raise ValueError("--in-wav is required when --mode decode-wav")
        os.makedirs(args.result_video_dir, exist_ok=True)
        out_video = args.out_video
        if not out_video:
            stem = os.path.splitext(os.path.basename(args.in_wav))[0]
            out_video = os.path.join(args.result_video_dir, f"{stem}_decoded.mp4")
        render_cfg = BridgeRenderConfig(width=args.width, height=args.height, fps=args.render_fps)
        rendered = decode_wav_to_video(
            wav_path=args.in_wav,
            out_video_path=out_video,
            fsk_cfg=fsk_cfg,
            render_cfg=render_cfg,
            use_timestamp_timing=args.timestamp_timing,
            max_hold_ms=args.timestamp_max_hold_ms,
        )
        print("=== Offline Decode Result ===")
        print(f"Input WAV   : {args.in_wav}")
        print(f"Output video: {out_video}")
        print(f"Frames      : {rendered}")
        print(f"Timestamp timing : {args.timestamp_timing}")
        print(f"Max hold (ms)    : {args.timestamp_max_hold_ms}")
        return

    render_cfg = BridgeRenderConfig(width=args.width, height=args.height, fps=args.render_fps)
    from Estimation_Module.Meeting_Bridge_Module.receiver.meeting_receiver import MeetingReceiver, MeetingReceiverConfig

    receiver_cfg = MeetingReceiverConfig(
        audio_input_device=args.audio_input_device,
        chunk_ms=args.chunk_ms,
        max_buffer_seconds=args.max_buffer_seconds,
        display=args.display,
        publish_virtual_cam=args.publish_virtual_cam,
        vcam_device=args.vcam_device,
    )
    MeetingReceiver(receiver_cfg, fsk_cfg, render_cfg).run()


if __name__ == "__main__":
    main()
