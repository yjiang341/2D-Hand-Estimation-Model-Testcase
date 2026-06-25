from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from itertools import product
from typing import List

from Conferencing_Module.channel.channel_simulator import ChannelConfig
from Conferencing_Module.tuning.fsk_tuner import (
    SweepConfig,
    SweepResult,
    choose_profile_winner,
    recommend_fallback,
    run_fsk_parameter_sweep,
)
from Conferencing_Module.virtual.virtual_camera import run_virtual_camera_probe
from FSK_Module.fsk_sender_main import generate_demo_packets


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def build_default_sweep_configs(
    symbol_rates: List[int],
    separations: List[float],
    detection_thresholds: List[float],
    silence_ms_values: List[int],
    base_freq0: float,
    sample_rate: int,
) -> List[SweepConfig]:
    configs: List[SweepConfig] = []
    for symbol_rate, separation, threshold, silence_ms in product(
        symbol_rates, separations, detection_thresholds, silence_ms_values
    ):
        if sample_rate % int(symbol_rate) != 0:
            # Deterministic modem framing requires integer samples/symbol.
            continue
        configs.append(
            SweepConfig(
                symbol_rate=int(symbol_rate),
                freq0_hz=float(base_freq0),
                freq1_hz=float(base_freq0 + separation),
                silence_ms=int(silence_ms),
                detection_threshold=float(threshold),
            )
        )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conferencing readiness: channel sweep, fallback guidance, and virtual-cam probe"
    )
    parser.add_argument("--mode", choices=["sweep", "virtual-cam", "both"], default="sweep")
    parser.add_argument("--frames", type=int, default=90, help="Demo frame count for sweep packets")
    parser.add_argument("--fps", type=int, default=15, help="Demo timestamp rate")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--base-freq0", type=float, default=1200.0)

    parser.add_argument("--symbol-rates", type=str, default="900,1200,1600")
    parser.add_argument("--freq-separations", type=str, default="800,1000,1400")
    parser.add_argument("--detection-thresholds", type=str, default="0.50,0.55,0.60")
    parser.add_argument("--silence-ms-values", type=str, default="2,3,4")

    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--dropout-prob", type=float, default=0.01)
    parser.add_argument("--amplitude-jitter", type=float, default=0.05)
    parser.add_argument("--delay-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--vcam-width", type=int, default=1280)
    parser.add_argument("--vcam-height", type=int, default=720)
    parser.add_argument("--vcam-fps", type=float, default=15.0)
    parser.add_argument("--vcam-frames", type=int, default=30)

    parser.add_argument(
        "--profile",
        choices=["high-reliability", "balanced", "low-latency"],
        default="balanced",
        help="Auto-select winner using target constraints.",
    )
    parser.add_argument("--out-json", type=str, default=os.path.join("logs", "readiness_report.json"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    report: dict = {
        "mode": args.mode,
        "sweep": None,
        "virtual_cam": None,
    }

    if args.mode in ("sweep", "both"):
        packets = generate_demo_packets(frame_count=args.frames, fps=args.fps)

        channel_cfg = ChannelConfig(
            noise_std=args.noise_std,
            dropout_prob=args.dropout_prob,
            amplitude_jitter=args.amplitude_jitter,
            delay_samples=args.delay_samples,
            seed=args.seed,
        )

        sweep_cfgs = build_default_sweep_configs(
            symbol_rates=_parse_csv_ints(args.symbol_rates),
            separations=_parse_csv_floats(args.freq_separations),
            detection_thresholds=_parse_csv_floats(args.detection_thresholds),
            silence_ms_values=_parse_csv_ints(args.silence_ms_values),
            base_freq0=args.base_freq0,
            sample_rate=args.sample_rate,
        )

        if not sweep_cfgs:
            raise ValueError(
                "No valid sweep configuration remained after filtering. "
                "Ensure sample-rate is divisible by at least one symbol-rate."
            )

        results: List[SweepResult] = run_fsk_parameter_sweep(
            packets=packets,
            sample_rate=args.sample_rate,
            amplitude=args.amplitude,
            channel_config=channel_cfg,
            sweep_configs=sweep_cfgs,
        )

        top_k = max(1, args.top_k)
        report["sweep"] = {
            "channel_config": asdict(channel_cfg),
            "tested_configs": len(sweep_cfgs),
            "top_results": [r.to_dict() for r in results[:top_k]],
            "fallback_recommendation": recommend_fallback(results),
        }

        selected = choose_profile_winner(results, args.profile)
        if selected is not None:
            report["sweep"]["selected_profile"] = args.profile
            report["sweep"]["selected_result"] = selected.to_dict()

        if results:
            best = results[0]
            print("=== Sweep Best Config ===")
            print(f"symbol_rate          : {best.config.symbol_rate}")
            print(f"freq0/freq1          : {best.config.freq0_hz}/{best.config.freq1_hz}")
            print(f"silence_ms           : {best.config.silence_ms}")
            print(f"detection_threshold  : {best.config.detection_threshold}")
            print(f"frame_loss_rate      : {best.summary.frame_loss_rate:.4f}")
            print(f"crc_reject_rate      : {best.summary.crc_reject_rate:.4f}")
            print(f"score                : {best.score:.4f}")
            print(f"est_frame_tx_ms      : {best.estimated_frame_tx_ms:.2f}")
            print(f"fallback             : {report['sweep']['fallback_recommendation']}")

            selected = choose_profile_winner(results, args.profile)
            if selected is not None:
                print("=== Profile Selected Config ===")
                print(f"profile              : {args.profile}")
                print(f"symbol_rate          : {selected.config.symbol_rate}")
                print(f"freq0/freq1          : {selected.config.freq0_hz}/{selected.config.freq1_hz}")
                print(f"silence_ms           : {selected.config.silence_ms}")
                print(f"detection_threshold  : {selected.config.detection_threshold}")
                print(f"frame_loss_rate      : {selected.summary.frame_loss_rate:.4f}")
                print(f"crc_reject_rate      : {selected.summary.crc_reject_rate:.4f}")
                print(f"est_frame_tx_ms      : {selected.estimated_frame_tx_ms:.2f}")

    if args.mode in ("virtual-cam", "both"):
        vcam_result = run_virtual_camera_probe(
            width=args.vcam_width,
            height=args.vcam_height,
            fps=args.vcam_fps,
            frames=args.vcam_frames,
        )
        report["virtual_cam"] = vcam_result

        print("=== Virtual Camera Check ===")
        print(f"virtual_cam_ok       : {vcam_result['virtual_cam_ok']}")
        print(f"frames_sent          : {vcam_result['frames_sent']}")
        print(f"message              : {vcam_result['message']}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Readiness report saved : {args.out_json}")


if __name__ == "__main__":
    main()
