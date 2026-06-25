from __future__ import annotations

from Conferencing_Module.channel.channel_simulator import ChannelConfig
from Conferencing_Module.tuning.fsk_tuner import SweepConfig, run_fsk_parameter_sweep
from FSK_Module.fsk_sender_main import generate_demo_packets


def test_sweep_recovers_frames_clean_channel() -> None:
    packets = generate_demo_packets(frame_count=12, fps=15)
    results = run_fsk_parameter_sweep(
        packets=packets,
        sample_rate=48_000,
        amplitude=0.8,
        channel_config=ChannelConfig(noise_std=0.0, dropout_prob=0.0, amplitude_jitter=0.0, delay_samples=0, seed=1),
        sweep_configs=[
            SweepConfig(
                symbol_rate=1_200,
                freq0_hz=1_200.0,
                freq1_hz=2_200.0,
                silence_ms=3,
                detection_threshold=0.55,
            )
        ],
    )

    assert len(results) == 1
    assert results[0].summary.total_tx_frames == 12
    assert results[0].summary.total_rx_frames >= 10
    assert results[0].summary.frame_loss_rate < 0.2
