from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelConfig:
    """Simple audio-channel impairment model for conferencing readiness tests."""

    noise_std: float = 0.0
    dropout_prob: float = 0.0
    amplitude_jitter: float = 0.0
    delay_samples: int = 0
    seed: int = 1234


def apply_channel_impairments(waveform: np.ndarray, config: ChannelConfig) -> np.ndarray:
    """
    Apply deterministic synthetic impairments to a waveform.

    This approximates conferencing transport effects (noise, packetized dropouts,
    gain fluctuation, and one-way delay) for repeatable parameter sweeps.
    """
    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1D array")
    if config.noise_std < 0.0:
        raise ValueError("noise_std must be >= 0")
    if not (0.0 <= config.dropout_prob < 1.0):
        raise ValueError("dropout_prob must be in [0, 1)")
    if config.amplitude_jitter < 0.0:
        raise ValueError("amplitude_jitter must be >= 0")
    if config.delay_samples < 0:
        raise ValueError("delay_samples must be >= 0")

    out = waveform.astype(np.float32, copy=True)
    rng = np.random.default_rng(config.seed)

    if config.amplitude_jitter > 0.0:
        # Per-sample gain jitter is a crude proxy for AGC/compression fluctuation.
        gain = rng.uniform(
            low=max(0.0, 1.0 - config.amplitude_jitter),
            high=1.0 + config.amplitude_jitter,
            size=out.shape,
        ).astype(np.float32)
        out *= gain

    if config.dropout_prob > 0.0:
        keep = rng.random(out.shape, dtype=np.float32) >= float(config.dropout_prob)
        out *= keep.astype(np.float32)

    if config.noise_std > 0.0:
        out += rng.normal(loc=0.0, scale=config.noise_std, size=out.shape).astype(np.float32)

    if config.delay_samples > 0:
        out = np.concatenate([np.zeros((config.delay_samples,), dtype=np.float32), out])

    return np.clip(out, -1.0, 1.0)
