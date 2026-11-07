from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from voice_ai_realtime_gateway.telemetry.events import FirstTokenLatency, PacketLatency
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.latency import (
    FirstTokenTracker,
    GlassToGlassTracker,
    LatencyWindow,
)

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder


def test_window_rejects_invalid_configuration_and_samples() -> None:
    with pytest.raises(ValueError, match="capacity"):
        LatencyWindow(capacity=0)
    window = LatencyWindow(capacity=4)
    with pytest.raises(ValueError, match="non-negative"):
        window.record(-1)
    with pytest.raises(ValueError, match="empty"):
        window.percentile(0.5)
    with pytest.raises(ValueError, match="empty"):
        window.summary()
    window.record(1)
    for quantile in (0.0, -0.1, 1.000001):
        with pytest.raises(ValueError, match="quantile"):
            window.percentile(quantile)


def test_window_storage_is_preallocated_and_bounded() -> None:
    window = LatencyWindow(capacity=8)
    storage = window._samples
    for sample in range(10_000):
        window.record(sample)
    assert window._samples is storage
    assert len(storage) == 8
    assert window.count == 10_000


def test_summary_mixes_lifetime_extrema_with_windowed_percentiles() -> None:
    window = LatencyWindow(capacity=4)
    for sample in (100, 1, 2, 3, 4):
        window.record(sample)
    summary = window.summary()
    assert summary.count == 5
    assert summary.window == 4
    assert summary.min_ns == 1
    assert summary.max_ns == 100
    assert summary.mean_ns == pytest.approx(22.0)
    assert (summary.p50_ns, summary.p95_ns, summary.p99_ns) == (2, 4, 4)


@given(
    samples=st.lists(st.integers(min_value=0, max_value=10**12), min_size=1, max_size=200),
    quantile=st.floats(min_value=1e-6, max_value=1.0),
    capacity=st.integers(min_value=1, max_value=64),
)
def test_percentile_matches_nearest_rank_reference(
    samples: list[int], quantile: float, capacity: int
) -> None:
    window = LatencyWindow(capacity=capacity)
    for sample in samples:
        window.record(sample)
    retained = sorted(samples[-capacity:])
    expected = retained[math.ceil(quantile * len(retained)) - 1]
    assert window.percentile(quantile) == expected


def test_glass_to_glass_records_and_emits(recorder: FlightRecorder) -> None:
    trace = TraceContext.new()
    tracker = GlassToGlassTracker(recorder)
    event = tracker.record(trace=trace, sequence=7, origin_ns=1_000, egress_ns=4_500)
    assert event == PacketLatency(trace=trace, at_ns=4_500, sequence=7, latency_ns=3_500)
    assert recorder.events == (event,)
    assert tracker.window.summary().max_ns == 3_500


def test_glass_to_glass_rejects_time_travel(recorder: FlightRecorder) -> None:
    tracker = GlassToGlassTracker(recorder, LatencyWindow(capacity=2))
    with pytest.raises(ValueError, match="non-negative"):
        tracker.record(trace=TraceContext.new(), sequence=0, origin_ns=10, egress_ns=9)
    assert recorder.events == ()


def test_first_token_reports_exactly_once_per_turn(recorder: FlightRecorder) -> None:
    trace = TraceContext.new()
    tracker = FirstTokenTracker(recorder)
    assert tracker.first_token(trace=trace, at_ns=5) is None
    tracker.user_turn_ended(100)
    armed = [tracker.armed]
    first = tracker.first_token(trace=trace, at_ns=350)
    armed.append(tracker.armed)
    assert armed == [True, False]
    assert first == FirstTokenLatency(trace=trace, at_ns=350, latency_ns=250)
    assert tracker.first_token(trace=trace, at_ns=400) is None
    assert recorder.events == (first,)
    assert tracker.window.count == 1


def test_first_token_disarm_abandons_measurement(recorder: FlightRecorder) -> None:
    tracker = FirstTokenTracker(recorder, LatencyWindow(capacity=2))
    tracker.user_turn_ended(1)
    tracker.disarm()
    assert tracker.first_token(trace=TraceContext.new(), at_ns=2) is None
    assert recorder.events == ()
