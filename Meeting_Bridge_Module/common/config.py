from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeFSKConfig:
    sample_rate: int = 48_000
    symbol_rate: int = 1_600
    freq0_hz: float = 1_200.0
    freq1_hz: float = 2_200.0
    amplitude: float = 0.8
    silence_ms: int = 2
    detection_threshold: float = 0.55


@dataclass(frozen=True)
class BridgeRenderConfig:
    width: int = 1280
    height: int = 720
    fps: float = 15.0


def estimate_packet_tx_ms(symbol_rate: int, payload_bytes: int = 104, preamble_bytes: int = 4, silence_ms: int = 2) -> float:
    bits = (payload_bytes + preamble_bytes) * 8
    return bits * 1000.0 / float(symbol_rate) + float(silence_ms)
