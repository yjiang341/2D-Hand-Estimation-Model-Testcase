from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from FSK_Module.fsk_receiver import ReceiverReport


@dataclass(frozen=True)
class ReadinessSummary:
    total_tx_frames: int
    total_rx_frames: int
    frame_loss_rate: float
    crc_reject_rate: float
    avg_e2e_latency_ms: float
    p95_e2e_latency_ms: float
    max_e2e_latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_readiness(
    tx_frame_ids: Iterable[int],
    receiver_report: ReceiverReport,
    tx_timestamp_ms_by_frame_id: Optional[Dict[int, int]] = None,
    decode_wallclock_ms_by_frame_id: Optional[Dict[int, int]] = None,
) -> ReadinessSummary:
    tx_ids = list(tx_frame_ids)
    tx_total = len(tx_ids)

    rx_ids = [int(packet.frame_id) for packet in receiver_report.valid_packets]
    unique_rx_ids = set(rx_ids)

    if tx_total == 0:
        loss_rate = 0.0
    else:
        loss_rate = max(0.0, 1.0 - (len(unique_rx_ids) / float(tx_total)))

    if receiver_report.attempted_frames > 0:
        crc_reject_rate = receiver_report.rejected_frames / float(receiver_report.attempted_frames)
    else:
        crc_reject_rate = 0.0

    latencies: List[float] = []
    if tx_timestamp_ms_by_frame_id is not None and decode_wallclock_ms_by_frame_id is not None:
        for frame_id in unique_rx_ids:
            tx_ts = tx_timestamp_ms_by_frame_id.get(frame_id)
            rx_ts = decode_wallclock_ms_by_frame_id.get(frame_id)
            if tx_ts is None or rx_ts is None:
                continue
            latencies.append(max(0.0, float(rx_ts - tx_ts)))

    if latencies:
        lat_np = np.array(latencies, dtype=np.float32)
        avg_latency = float(np.mean(lat_np))
        p95_latency = float(np.percentile(lat_np, 95))
        max_latency = float(np.max(lat_np))
    else:
        avg_latency = 0.0
        p95_latency = 0.0
        max_latency = 0.0

    return ReadinessSummary(
        total_tx_frames=tx_total,
        total_rx_frames=len(unique_rx_ids),
        frame_loss_rate=loss_rate,
        crc_reject_rate=crc_reject_rate,
        avg_e2e_latency_ms=avg_latency,
        p95_e2e_latency_ms=p95_latency,
        max_e2e_latency_ms=max_latency,
    )
