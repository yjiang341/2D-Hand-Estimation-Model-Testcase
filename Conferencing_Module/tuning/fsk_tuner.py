from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Literal

from Conferencing_Module.channel.channel_simulator import ChannelConfig, apply_channel_impairments
from Conferencing_Module.metrics.readiness_metrics import ReadinessSummary, summarize_readiness
from FSK_Module.fsk_modem import FSKConfig, modulate_packet_stream
from FSK_Module.fsk_receiver import recover_packets_from_waveform
from Pose_PacketUp.pose_packet import PacketDecodeError, decode_packet


@dataclass(frozen=True)
class SweepConfig:
    symbol_rate: int
    freq0_hz: float
    freq1_hz: float
    silence_ms: int
    detection_threshold: float


@dataclass(frozen=True)
class SweepResult:
    config: SweepConfig
    summary: ReadinessSummary
    score: float
    estimated_frame_tx_ms: float

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "summary": self.summary.to_dict(),
            "score": self.score,
            "estimated_frame_tx_ms": self.estimated_frame_tx_ms,
        }


ProfileName = Literal["high-reliability", "balanced", "low-latency"]


def _score_summary(summary: ReadinessSummary) -> float:
    # Favor high delivery and low rejection. Latency is often app-dependent in offline sweeps.
    return (
        (1.0 - summary.frame_loss_rate) * 100.0
        - summary.crc_reject_rate * 20.0
        - summary.p95_e2e_latency_ms * 0.01
    )


def _estimate_frame_tx_ms(config: SweepConfig, packet_size_bytes: int = 104, preamble_bytes: int = 4) -> float:
    bits = (packet_size_bytes + preamble_bytes) * 8
    return (bits * 1000.0 / float(config.symbol_rate)) + float(config.silence_ms)


def _profile_objective(result: SweepResult, profile: ProfileName) -> float:
    loss = result.summary.frame_loss_rate
    reject = result.summary.crc_reject_rate
    tx_ms = result.estimated_frame_tx_ms

    if profile == "high-reliability":
        return -(120.0 * loss + 45.0 * reject + 0.05 * tx_ms)
    if profile == "low-latency":
        return -(45.0 * loss + 20.0 * reject + 0.9 * tx_ms)
    return -(70.0 * loss + 25.0 * reject + 0.25 * tx_ms)


def choose_profile_winner(results: List[SweepResult], profile: ProfileName) -> SweepResult | None:
    if not results:
        return None

    if profile == "high-reliability":
        candidates = [
            r
            for r in results
            if r.summary.frame_loss_rate <= 0.02 and r.summary.crc_reject_rate <= 0.03
        ]
        if not candidates:
            candidates = results
    elif profile == "low-latency":
        candidates = [
            r
            for r in results
            if r.summary.frame_loss_rate <= 0.12 and r.summary.crc_reject_rate <= 0.20
        ]
        if not candidates:
            candidates = results
    else:
        candidates = [
            r
            for r in results
            if r.summary.frame_loss_rate <= 0.06 and r.summary.crc_reject_rate <= 0.10
        ]
        if not candidates:
            candidates = results

    return max(candidates, key=lambda r: _profile_objective(r, profile))


def run_fsk_parameter_sweep(
    packets: List[bytes],
    sample_rate: int,
    amplitude: float,
    channel_config: ChannelConfig,
    sweep_configs: Iterable[SweepConfig],
) -> List[SweepResult]:
    tx_frame_ids: List[int] = []
    for packet_bytes in packets:
        try:
            tx_frame_ids.append(int(decode_packet(packet_bytes).frame_id))
        except PacketDecodeError:
            # Sender side test vectors should already be valid; skip if not.
            continue

    results: List[SweepResult] = []

    for cfg in sweep_configs:
        modem_cfg = FSKConfig(
            sample_rate=sample_rate,
            symbol_rate=cfg.symbol_rate,
            freq0_hz=cfg.freq0_hz,
            freq1_hz=cfg.freq1_hz,
            amplitude=amplitude,
            inter_frame_silence_ms=cfg.silence_ms,
        )

        tx_wave = modulate_packet_stream(packets, modem_cfg)
        rx_wave = apply_channel_impairments(tx_wave, channel_config)

        report = recover_packets_from_waveform(
            waveform=rx_wave,
            config=modem_cfg,
            detection_threshold=cfg.detection_threshold,
        )
        summary = summarize_readiness(tx_frame_ids=tx_frame_ids, receiver_report=report)
        results.append(
            SweepResult(
                config=cfg,
                summary=summary,
                score=_score_summary(summary),
                estimated_frame_tx_ms=_estimate_frame_tx_ms(cfg),
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results


def recommend_fallback(results: List[SweepResult]) -> str:
    if not results:
        return "No sweep results available."

    top = results[0]
    if top.summary.frame_loss_rate <= 0.02 and top.summary.crc_reject_rate <= 0.05:
        return "Channel appears stable; no fallback required."

    if top.config.symbol_rate > 1200:
        return "Enable fallback to symbol_rate=1200 and keep wider freq separation (>900 Hz)."

    return "Enable fallback to lower symbol rate (900) and raise inter-frame silence to 4-5 ms."
