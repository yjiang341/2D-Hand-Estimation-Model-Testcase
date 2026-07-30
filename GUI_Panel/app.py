from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import traceback

from Meeting_Bridge_Module.common.config import BridgeFSKConfig, BridgeRenderConfig
from Meeting_Bridge_Module.receiver.meeting_receiver import MeetingReceiver, MeetingReceiverConfig, decode_wav_to_video
from Meeting_Bridge_Module.sender.meeting_sender import MeetingSender, MeetingSenderConfig, PreparedSenderSession

from Meeting_Bridge_Module.audio.device_io import AudioBackendError, list_audio_devices


class GuiPanelApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hand Pose GUI Panel")
        self.root.geometry("1024x760")

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._devices_refreshing = False
        self._fsk_cfg = BridgeFSKConfig()
        self._render_cfg = BridgeRenderConfig()

        self._sender_thread: threading.Thread | None = None
        self._sender_stop = threading.Event()
        self._sender_session: PreparedSenderSession | None = None
        self._sender_session_key: tuple | None = None
        self._sender_session_thread: threading.Thread | None = None
        self._sender_session_ready = threading.Event()
        self._sender_session_state = "idle"
        self._sender_session_error: str | None = None
        self._closing = False

        self._receiver_thread: threading.Thread | None = None
        self._receiver_stop = threading.Event()

        self._decode_thread: threading.Thread | None = None

        self._live_proc: subprocess.Popen[str] | None = None
        self._live_monitor_thread: threading.Thread | None = None

        self._build_ui()
        self._pump_logs()
        self.refresh_devices_async()
        self._update_latest_wav_preview()
        self._refresh_sender_session_async(force=True)
        self._refresh_action_states()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_devices = ttk.Frame(notebook)
        self.tab_sender = ttk.Frame(notebook)
        self.tab_receiver_decode = ttk.Frame(notebook)
        self.tab_live = ttk.Frame(notebook)
        self.tab_logs = ttk.Frame(notebook)

        notebook.add(self.tab_devices, text="Devices")
        notebook.add(self.tab_sender, text="Sender")
        notebook.add(self.tab_receiver_decode, text="Receiver / Decode")
        notebook.add(self.tab_live, text="Live")
        notebook.add(self.tab_logs, text="Logs")

        self._build_devices_tab()
        self._build_sender_tab()
        self._build_receiver_decode_tab()
        self._build_live_tab()
        self._build_logs_tab()

    def _build_devices_tab(self) -> None:
        controls = ttk.Frame(self.tab_devices)
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.refresh_btn = ttk.Button(controls, text="Refresh Devices", command=self.refresh_devices_async)
        self.refresh_btn.pack(side=tk.LEFT)

        self.devices_text = tk.Text(self.tab_devices, wrap=tk.WORD, height=28)
        self.devices_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_sender_tab(self) -> None:
        frame = ttk.Frame(self.tab_sender)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frame, text="Output Device").grid(row=0, column=0, sticky=tk.W)
        self.sender_output_device_var = tk.StringVar(value="")
        self.sender_output_device_combo = ttk.Combobox(frame, textvariable=self.sender_output_device_var, width=72, state="readonly")
        self.sender_output_device_combo.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(frame, text="TX FPS").grid(row=1, column=0, sticky=tk.W)
        self.sender_tx_fps_var = tk.DoubleVar(value=1.8)
        ttk.Entry(frame, textvariable=self.sender_tx_fps_var, width=12).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(frame, text="Camera ID").grid(row=2, column=0, sticky=tk.W)
        self.sender_camera_id_var = tk.IntVar(value=0)
        ttk.Entry(frame, textvariable=self.sender_camera_id_var, width=12).grid(row=2, column=1, sticky=tk.W)

        self.sender_show_preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Show Preview", variable=self.sender_show_preview_var).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(frame, text="Local WAV Dir").grid(row=4, column=0, sticky=tk.W)
        self.sender_wav_dir_var = tk.StringVar(value="local_sender_copy")
        ttk.Entry(frame, textvariable=self.sender_wav_dir_var, width=40).grid(row=4, column=1, sticky=tk.W)

        btns = ttk.Frame(frame)
        btns.grid(row=5, column=1, sticky=tk.W, pady=10)
        self.btn_sender_tx = ttk.Button(btns, text="Start Sender", command=self._start_sender_tx_mode)
        self.btn_sender_tx.pack(side=tk.LEFT, padx=4)
        self.btn_sender_local = ttk.Button(btns, text="Start Sender (With Local Copy)", command=self._start_sender_local_mode)
        self.btn_sender_local.pack(side=tk.LEFT, padx=4)
        self.btn_sender_stop = ttk.Button(btns, text="Stop Sender", command=self._stop_sender)
        self.btn_sender_stop.pack(side=tk.LEFT, padx=4)

        self.sender_status_var = tk.StringVar(value="idle")
        ttk.Label(frame, textvariable=self.sender_status_var).grid(row=6, column=0, columnspan=2, sticky=tk.W)

    def _build_receiver_decode_tab(self) -> None:
        frame = ttk.Frame(self.tab_receiver_decode)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frame, text="Input Device").grid(row=0, column=0, sticky=tk.W)
        self.receiver_input_device_var = tk.StringVar(value="")
        self.receiver_input_device_combo = ttk.Combobox(frame, textvariable=self.receiver_input_device_var, width=72, state="readonly")
        self.receiver_input_device_combo.grid(row=0, column=1, sticky=tk.W)

        self.receiver_display_var = tk.BooleanVar(value=True)
        self.receiver_publish_vcam_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Display", variable=self.receiver_display_var).grid(row=1, column=1, sticky=tk.W)
        ttk.Checkbutton(frame, text="Publish Virtual Cam", variable=self.receiver_publish_vcam_var).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(frame, text="Latest WAV Source Dir").grid(row=3, column=0, sticky=tk.W)
        self.decode_wav_source_var = tk.StringVar(value="local_sender_copy")
        ttk.Entry(frame, textvariable=self.decode_wav_source_var, width=40).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(frame, text="Result Video Dir").grid(row=4, column=0, sticky=tk.W)
        self.decode_result_dir_var = tk.StringVar(value="result_video")
        ttk.Entry(frame, textvariable=self.decode_result_dir_var, width=40).grid(row=4, column=1, sticky=tk.W)

        self.decode_timestamp_timing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Timestamp Timing", variable=self.decode_timestamp_timing_var).grid(row=5, column=1, sticky=tk.W)

        ttk.Label(frame, text="Max Hold (ms)").grid(row=6, column=0, sticky=tk.W)
        self.decode_max_hold_ms_var = tk.IntVar(value=2500)
        ttk.Entry(frame, textvariable=self.decode_max_hold_ms_var, width=12).grid(row=6, column=1, sticky=tk.W)

        ttk.Label(frame, text="Latest WAV").grid(row=7, column=0, sticky=tk.W)
        self.latest_wav_var = tk.StringVar(value="(none)")
        ttk.Label(frame, textvariable=self.latest_wav_var).grid(row=7, column=1, sticky=tk.W)

        btns = ttk.Frame(frame)
        btns.grid(row=8, column=1, sticky=tk.W, pady=10)
        self.btn_receiver_start = ttk.Button(btns, text="Start Receiver", command=self._start_receiver)
        self.btn_receiver_start.pack(side=tk.LEFT, padx=4)
        self.btn_receiver_stop = ttk.Button(btns, text="Stop Receiver", command=self._stop_receiver)
        self.btn_receiver_stop.pack(side=tk.LEFT, padx=4)
        self.btn_decode = ttk.Button(btns, text="Decode Latest WAV", command=self._decode_latest_wav)
        self.btn_decode.pack(side=tk.LEFT, padx=4)

        self.receiver_status_var = tk.StringVar(value="idle")
        ttk.Label(frame, textvariable=self.receiver_status_var).grid(row=9, column=0, columnspan=2, sticky=tk.W)

    def _build_live_tab(self) -> None:
        frame = ttk.Frame(self.tab_live)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frame, text="Preset").grid(row=0, column=0, sticky=tk.W)
        self.live_preset_var = tk.StringVar(value="Beginner")
        ttk.Combobox(frame, textvariable=self.live_preset_var, values=["Beginner", "Advanced"], state="readonly", width=18).grid(
            row=0, column=1, sticky=tk.W
        )

        ttk.Label(frame, text="Output Mode").grid(row=1, column=0, sticky=tk.W)
        self.live_output_mode_var = tk.StringVar(value="display")
        ttk.Combobox(
            frame,
            textvariable=self.live_output_mode_var,
            values=["display", "virtual-cam", "both", "headless"],
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(frame, text="Max Frames (0 = run until stop)").grid(row=2, column=0, sticky=tk.W)
        self.live_max_frames_var = tk.IntVar(value=0)
        ttk.Entry(frame, textvariable=self.live_max_frames_var, width=12).grid(row=2, column=1, sticky=tk.W)

        self.live_display_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Display Windows", variable=self.live_display_var).grid(row=3, column=1, sticky=tk.W)

        btns = ttk.Frame(frame)
        btns.grid(row=4, column=1, sticky=tk.W, pady=10)
        ttk.Button(btns, text="Apply Preset", command=self._apply_live_preset).pack(side=tk.LEFT, padx=4)
        self.btn_live_start = ttk.Button(btns, text="Start Live", command=self._start_live)
        self.btn_live_start.pack(side=tk.LEFT, padx=4)
        self.btn_live_stop = ttk.Button(btns, text="Stop Live", command=self._stop_live)
        self.btn_live_stop.pack(side=tk.LEFT, padx=4)

        self.live_status_var = tk.StringVar(value="idle")
        ttk.Label(frame, textvariable=self.live_status_var).grid(row=5, column=0, columnspan=2, sticky=tk.W)

    def _build_logs_tab(self) -> None:
        self.log_text = tk.Text(self.tab_logs, wrap=tk.WORD, height=28)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def refresh_devices_async(self) -> None:
        if self._devices_refreshing:
            return
        self._devices_refreshing = True
        self.refresh_btn.configure(state=tk.DISABLED)
        self._log("[devices] refreshing...")

        def _worker() -> None:
            try:
                devices = list_audio_devices()
                self.root.after(0, lambda: self._apply_devices(devices))
            except AudioBackendError as exc:
                self._log(f"[devices][error] {exc}")
            except Exception as exc:  # pragma: no cover
                self._log(f"[devices][error] unexpected: {exc}")
            finally:
                self.root.after(0, self._finish_refresh)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_devices(self, devices) -> None:
        microphones = [d for d in devices if int(getattr(d, "max_input_channels", 0)) > 0]
        speakers = [d for d in devices if int(getattr(d, "max_output_channels", 0)) > 0]

        lines = []
        lines.append("=== Microphones (input channels > 0) ===")
        if microphones:
            for d in microphones:
                lines.append(
                    f"[{d.index}] {d.name} | in={d.max_input_channels} out={d.max_output_channels} default_sr={d.default_samplerate:.0f}"
                )
        else:
            lines.append("(none)")

        lines.append("")
        lines.append("=== Speakers (output channels > 0) ===")
        if speakers:
            for d in speakers:
                lines.append(
                    f"[{d.index}] {d.name} | in={d.max_input_channels} out={d.max_output_channels} default_sr={d.default_samplerate:.0f}"
                )
        else:
            lines.append("(none)")

        lines.append("")
        lines.append("=== All Devices ===")
        for d in devices:
            lines.append(
                f"[{d.index}] {d.name} | in={d.max_input_channels} out={d.max_output_channels} default_sr={d.default_samplerate:.0f}"
            )

        self.devices_text.delete("1.0", tk.END)
        self.devices_text.insert(tk.END, "\n".join(lines))

        output_names = [d.name for d in speakers]
        input_names = [d.name for d in microphones]

        self.sender_output_device_combo["values"] = output_names
        self.receiver_input_device_combo["values"] = input_names

        if output_names and not self.sender_output_device_var.get():
            self.sender_output_device_var.set(output_names[0])
        if input_names and not self.receiver_input_device_var.get():
            self.receiver_input_device_var.set(input_names[0])

        self._log(f"[devices] loaded total={len(devices)} microphones={len(microphones)} speakers={len(speakers)}")
        self._update_latest_wav_preview()

    def _finish_refresh(self) -> None:
        self._devices_refreshing = False
        self.refresh_btn.configure(state=tk.NORMAL)
        self._refresh_sender_session_async(force=True)

    def _sender_prep_cfg(self) -> MeetingSenderConfig | None:
        output_device = self.sender_output_device_var.get().strip()
        if not output_device:
            return None

        return MeetingSenderConfig(
            model_path=os.path.join("Models", "hand_landmarker.task"),
            camera_id=int(self.sender_camera_id_var.get()),
            width=1280,
            height=720,
            tx_fps=float(self.sender_tx_fps_var.get()),
            show_preview=bool(self.sender_show_preview_var.get()),
            audio_output_device=output_device,
            audio_output_fallback=True,
            camera_backend="dshow" if os.name == "nt" else "auto",
            save_local_wav_copy=False,
            local_wav_copy_dir=str(self.sender_wav_dir_var.get()).strip() or "local_sender_copy",
            warmup_enabled=True,
            warmup_frame_tries=5,
        )

    def _sender_session_signature(self, cfg: MeetingSenderConfig) -> tuple:
        return (
            os.path.abspath(cfg.model_path),
            int(cfg.camera_id),
            int(cfg.width),
            int(cfg.height),
            str(cfg.audio_output_device or ""),
            bool(cfg.audio_output_fallback),
            str(cfg.camera_backend),
        )

    def _refresh_sender_session_async(self, force: bool = False) -> None:
        if self._closing:
            return

        cfg = self._sender_prep_cfg()
        if cfg is None:
            self._sender_session_state = "idle"
            self._sender_session_error = None
            self._refresh_action_states()
            return

        signature = self._sender_session_signature(cfg)
        if not force and self._sender_session is not None and self._sender_session_key == signature:
            self._sender_session_state = "ready"
            self._sender_session_error = None
            self._refresh_action_states()
            return

        if self._sender_session_thread is not None and self._sender_session_thread.is_alive():
            return

        if self._sender_session is not None:
            try:
                self._sender_session.close()
            except Exception:
                pass
            self._sender_session = None
            self._sender_session_key = None

        self._sender_session_state = "warming"
        self._sender_session_error = None
        self._sender_session_ready.clear()
        self.sender_status_var.set("warming")
        self._refresh_action_states()

        def _worker() -> None:
            try:
                sender = MeetingSender(cfg, self._fsk_cfg)
                session = sender.prepare_session(status_callback=lambda m: self._log(f"[sender-warm] {m}"))
                self._sender_session = session
                self._sender_session_key = signature
                self._sender_session_state = "ready"
                self._sender_session_ready.set()
                self.root.after(0, self._on_sender_session_ready)
            except Exception as exc:  # pragma: no cover
                self._sender_session_error = str(exc)
                self._sender_session_state = "idle"
                self._sender_session_ready.clear()
                self.root.after(0, lambda: self._log(f"[sender-warm][error] {exc}"))
                self.root.after(0, self._refresh_action_states)

        self._sender_session_thread = threading.Thread(target=_worker, daemon=True)
        self._sender_session_thread.start()

    def _on_sender_session_ready(self) -> None:
        self.sender_status_var.set("ready")
        self._refresh_action_states()

    def _start_sender_tx_mode(self) -> None:
        self._start_sender(use_local_copy_override=False)

    def _start_sender_local_mode(self) -> None:
        self._start_sender(use_local_copy_override=True)

    def _start_sender(self, use_local_copy_override: bool) -> None:
        if self._is_receiver_running() or self._is_live_running():
            self._log("[sender][guard] stop receiver/live before starting sender")
            return

        if self._sender_thread is not None and self._sender_thread.is_alive():
            self._log("[sender] already running")
            return

        output_device = self.sender_output_device_var.get().strip()
        if not output_device:
            self._log("[sender][error] no output device selected")
            return

        prep_cfg = self._sender_prep_cfg()
        prep_key = self._sender_session_signature(prep_cfg) if prep_cfg is not None else None

        if self._sender_session is not None and prep_key is not None and self._sender_session_key != prep_key:
            try:
                self._sender_session.close()
            except Exception:
                pass
            self._sender_session = None
            self._sender_session_key = None

        self._sender_stop.clear()
        self.sender_status_var.set("running")
        self._refresh_action_states()

        runtime_cfg = MeetingSenderConfig(
            model_path=os.path.join("Models", "hand_landmarker.task"),
            camera_id=int(self.sender_camera_id_var.get()),
            tx_fps=float(self.sender_tx_fps_var.get()),
            show_preview=bool(self.sender_show_preview_var.get()),
            audio_output_device=output_device,
            save_local_wav_copy=use_local_copy_override,
            local_wav_copy_dir=str(self.sender_wav_dir_var.get()).strip() or "local_sender_copy",
        )
        if use_local_copy_override:
            runtime_cfg = MeetingSenderConfig(
                model_path=runtime_cfg.model_path,
                camera_id=runtime_cfg.camera_id,
                width=runtime_cfg.width,
                height=runtime_cfg.height,
                tx_fps=runtime_cfg.tx_fps,
                max_frames=runtime_cfg.max_frames,
                show_preview=runtime_cfg.show_preview,
                audio_output_device=runtime_cfg.audio_output_device,
                audio_output_fallback=runtime_cfg.audio_output_fallback,
                camera_backend=runtime_cfg.camera_backend,
                save_local_wav_copy=True,
                local_wav_copy_dir=runtime_cfg.local_wav_copy_dir,
                local_wav_copy_path=None,
                warmup_enabled=runtime_cfg.warmup_enabled,
                warmup_frame_tries=runtime_cfg.warmup_frame_tries,
            )

        use_warm_session = self._sender_session is not None and prep_key is not None and self._sender_session_key == prep_key

        def _worker() -> None:
            try:
                self._log(f"[sender] starting output_device={output_device}")
                sender = MeetingSender(runtime_cfg, self._fsk_cfg)
                if use_warm_session and self._sender_session is not None:
                    sender.run_with_session(self._sender_session, stop_event=self._sender_stop, status_callback=lambda m: self._log(f"[sender] {m}"))
                else:
                    if self._sender_session_state == "warming":
                        self._log("[sender] warm session still preparing; waiting briefly")
                        self._sender_session_ready.wait(timeout=15.0)
                    if self._sender_session is not None and prep_key is not None and self._sender_session_key == prep_key:
                        sender.run_with_session(self._sender_session, stop_event=self._sender_stop, status_callback=lambda m: self._log(f"[sender] {m}"))
                    else:
                        sender.run(stop_event=self._sender_stop, status_callback=lambda m: self._log(f"[sender] {m}"))
            except Exception as exc:  # pragma: no cover
                self._log(f"[sender][error] {exc}")
                self._log(traceback.format_exc().rstrip())
            finally:
                self.root.after(0, self._on_sender_finished)

        self._sender_thread = threading.Thread(target=_worker, daemon=True)
        self._sender_thread.start()
        self._refresh_action_states()

    def _stop_sender(self) -> None:
        if self._sender_thread is None or not self._sender_thread.is_alive():
            self._log("[sender] not running")
            return
        self._sender_stop.set()
        self._log("[sender] stop requested")

    def _on_sender_finished(self) -> None:
        self.sender_status_var.set("idle")
        self._update_latest_wav_preview()
        self._refresh_sender_session_async(force=False)
        self._refresh_action_states()

    def _start_receiver(self) -> None:
        if self._is_sender_running() or self._is_live_running():
            self._log("[receiver][guard] stop sender/live before starting receiver")
            return

        if self._receiver_thread is not None and self._receiver_thread.is_alive():
            self._log("[receiver] already running")
            return

        input_device = self.receiver_input_device_var.get().strip()
        if not input_device:
            self._log("[receiver][error] no input device selected")
            return

        self._receiver_stop.clear()
        self.receiver_status_var.set("running")
        self._refresh_action_states()

        cfg = MeetingReceiverConfig(
            audio_input_device=input_device,
            display=bool(self.receiver_display_var.get()),
            publish_virtual_cam=bool(self.receiver_publish_vcam_var.get()),
        )

        def _worker() -> None:
            try:
                self._log(f"[receiver] starting input_device={input_device}")
                receiver = MeetingReceiver(cfg, self._fsk_cfg, self._render_cfg)
                receiver.run(stop_event=self._receiver_stop, status_callback=lambda m: self._log(f"[receiver] {m}"))
            except Exception as exc:  # pragma: no cover
                self._log(f"[receiver][error] {exc}")
                self._log(traceback.format_exc().rstrip())
            finally:
                self.root.after(0, self._on_receiver_finished)

        self._receiver_thread = threading.Thread(target=_worker, daemon=True)
        self._receiver_thread.start()
        self._refresh_action_states()

    def _stop_receiver(self) -> None:
        if self._receiver_thread is None or not self._receiver_thread.is_alive():
            self._log("[receiver] not running")
            return
        self._receiver_stop.set()
        self._log("[receiver] stop requested")

    def _on_receiver_finished(self) -> None:
        self.receiver_status_var.set("idle")
        self._refresh_action_states()

    def _decode_latest_wav(self) -> None:
        if self._is_sender_running() or self._is_receiver_running() or self._is_live_running():
            self._log("[decode][guard] stop sender/receiver/live before decode")
            return

        if self._decode_thread is not None and self._decode_thread.is_alive():
            self._log("[decode] already running")
            return

        latest_wav = self._find_latest_wav(str(self.decode_wav_source_var.get()).strip() or "local_sender_copy")
        if latest_wav is None:
            self._log("[decode][error] no .wav file found in source directory")
            self._update_latest_wav_preview()
            return

        self._refresh_action_states()

        def _worker() -> None:
            try:
                result_dir = str(self.decode_result_dir_var.get()).strip() or "result_video"
                os.makedirs(result_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(latest_wav))[0]
                out_video = os.path.join(result_dir, f"{stem}_decoded.mp4")
                if os.path.exists(out_video):
                    suffix = time.strftime("%Y%m%d_%H%M%S")
                    out_video = os.path.join(result_dir, f"{stem}_decoded_{suffix}.mp4")

                self._log(f"[decode] input={latest_wav}")
                frames = decode_wav_to_video(
                    wav_path=latest_wav,
                    out_video_path=out_video,
                    fsk_cfg=self._fsk_cfg,
                    render_cfg=self._render_cfg,
                    use_timestamp_timing=bool(self.decode_timestamp_timing_var.get()),
                    max_hold_ms=int(self.decode_max_hold_ms_var.get()),
                )
                self._log(f"[decode] output={out_video} frames={frames}")
            except Exception as exc:  # pragma: no cover
                self._log(f"[decode][error] {exc}")
                self._log(traceback.format_exc().rstrip())
            finally:
                self.root.after(0, self._on_decode_finished)

        self._decode_thread = threading.Thread(target=_worker, daemon=True)
        self._decode_thread.start()

    def _on_decode_finished(self) -> None:
        self._update_latest_wav_preview()
        self._refresh_action_states()

    def _apply_live_preset(self) -> None:
        preset = self.live_preset_var.get().strip().lower()
        if preset == "advanced":
            self.live_output_mode_var.set("both")
            self.live_display_var.set(True)
        else:
            self.live_output_mode_var.set("display")
            self.live_display_var.set(True)
        self._log(f"[live] preset applied: {self.live_preset_var.get()}")

    def _start_live(self) -> None:
        if self._is_sender_running() or self._is_receiver_running():
            self._log("[live][guard] stop sender/receiver before starting live")
            return

        if self._live_proc is not None and self._live_proc.poll() is None:
            self._log("[live] already running")
            return

        self.live_status_var.set("running")
        self._refresh_action_states()

        cmd = [
            sys.executable,
            os.path.join(os.getcwd(), "live_main.py"),
            "--output-mode",
            self.live_output_mode_var.get().strip() or "display",
            "--max-frames",
            str(int(self.live_max_frames_var.get())),
        ]
        if bool(self.live_display_var.get()):
            cmd.append("--display")
        else:
            cmd.append("--no-display")

        self._log("[live] starting: " + " ".join(cmd))
        try:
            self._live_proc = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._log(f"[live][error] failed to start: {exc}")
            self.live_status_var.set("idle")
            self._refresh_action_states()
            return

        def _monitor() -> None:
            assert self._live_proc is not None
            proc = self._live_proc
            if proc.stdout is not None:
                for line in proc.stdout:
                    self._log("[live] " + line.rstrip())
            code = proc.wait()
            self._log(f"[live] finished exit_code={code}")
            self.root.after(0, self._on_live_finished)

        self._live_monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self._live_monitor_thread.start()
        self._refresh_action_states()

    def _stop_live(self) -> None:
        if self._live_proc is None or self._live_proc.poll() is not None:
            self._log("[live] not running")
            return
        self._log("[live] stop requested")
        try:
            self._live_proc.terminate()
        except Exception as exc:  # pragma: no cover
            self._log(f"[live][error] terminate failed: {exc}")

    def _on_live_finished(self) -> None:
        self.live_status_var.set("idle")
        self._live_proc = None
        self._refresh_action_states()

    def _is_sender_running(self) -> bool:
        return self._sender_thread is not None and self._sender_thread.is_alive()

    def _is_receiver_running(self) -> bool:
        return self._receiver_thread is not None and self._receiver_thread.is_alive()

    def _is_decode_running(self) -> bool:
        return self._decode_thread is not None and self._decode_thread.is_alive()

    def _is_live_running(self) -> bool:
        return self._live_proc is not None and self._live_proc.poll() is None

    def _refresh_action_states(self) -> None:
        sender_running = self._is_sender_running()
        receiver_running = self._is_receiver_running()
        decode_running = self._is_decode_running()
        live_running = self._is_live_running()
        has_latest_wav = self.latest_wav_var.get().strip() not in ("", "(none)")

        can_start_sender = (not sender_running) and (not receiver_running) and (not live_running)
        can_start_receiver = (not receiver_running) and (not sender_running) and (not live_running)
        can_start_live = (not live_running) and (not sender_running) and (not receiver_running)
        can_decode = (not decode_running) and (not sender_running) and (not receiver_running) and (not live_running) and has_latest_wav

        self.btn_sender_tx.configure(state=(tk.NORMAL if can_start_sender else tk.DISABLED))
        self.btn_sender_local.configure(state=(tk.NORMAL if can_start_sender else tk.DISABLED))
        self.btn_sender_stop.configure(state=(tk.NORMAL if sender_running else tk.DISABLED))

        self.btn_receiver_start.configure(state=(tk.NORMAL if can_start_receiver else tk.DISABLED))
        self.btn_receiver_stop.configure(state=(tk.NORMAL if receiver_running else tk.DISABLED))

        self.btn_decode.configure(state=(tk.NORMAL if can_decode else tk.DISABLED))

        self.btn_live_start.configure(state=(tk.NORMAL if can_start_live else tk.DISABLED))
        self.btn_live_stop.configure(state=(tk.NORMAL if live_running else tk.DISABLED))

    def _find_latest_wav(self, source_dir: str) -> str | None:
        if not source_dir:
            return None
        if not os.path.isdir(source_dir):
            return None
        candidates = []
        for name in os.listdir(source_dir):
            if not name.lower().endswith(".wav"):
                continue
            full = os.path.join(source_dir, name)
            if os.path.isfile(full):
                candidates.append(full)
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _update_latest_wav_preview(self) -> None:
        source_dir = str(self.decode_wav_source_var.get()).strip() or "local_sender_copy"
        latest = self._find_latest_wav(source_dir)
        self.latest_wav_var.set(latest if latest else "(none)")
        self._refresh_action_states()

    def _log(self, message: str) -> None:
        self._log_queue.put(message)

    def _pump_logs(self) -> None:
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)
        self.root.after(100, self._pump_logs)

    def _on_close(self) -> None:
        self._closing = True
        self._sender_stop.set()
        self._receiver_stop.set()

        if self._live_proc is not None and self._live_proc.poll() is None:
            try:
                self._live_proc.terminate()
            except Exception:
                pass

        if self._sender_session is not None:
            try:
                self._sender_session.close()
            except Exception:
                pass

        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GuiPanelApp(root)
    root.mainloop()
