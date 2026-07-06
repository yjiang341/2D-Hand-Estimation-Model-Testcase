from __future__ import annotations

from Meeting_Bridge_Module.common.config import estimate_packet_tx_ms


def test_estimate_packet_tx_ms_decreases_with_higher_symbol_rate() -> None:
    low = estimate_packet_tx_ms(symbol_rate=1200, silence_ms=2)
    high = estimate_packet_tx_ms(symbol_rate=1600, silence_ms=2)
    assert high < low


def test_estimate_packet_tx_ms_positive() -> None:
    val = estimate_packet_tx_ms(symbol_rate=1600, silence_ms=2)
    assert val > 0.0
