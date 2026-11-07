from __future__ import annotations

import dataclasses

import pytest

from voice_ai_realtime_gateway.telemetry.events import (
    BargeInHandled,
    BufferOverflow,
    FirstTokenLatency,
    IngressGap,
    PacketLatency,
    StateTransitioned,
    TaskFailed,
    TransitionRejected,
    TransportClosed,
    VadSpeechStarted,
    VadSpeechStopped,
)
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder, NullSink, TelemetrySink

EVENT_TYPES = (
    BargeInHandled,
    BufferOverflow,
    FirstTokenLatency,
    IngressGap,
    PacketLatency,
    StateTransitioned,
    TaskFailed,
    TransitionRejected,
    TransportClosed,
    VadSpeechStarted,
    VadSpeechStopped,
)


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_every_event_is_frozen_slotted_and_trace_correlated(event_type: type) -> None:
    assert "__slots__" in vars(event_type)
    params = event_type.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen
    fields = {field.name for field in dataclasses.fields(event_type)}
    assert {"trace", "at_ns"} <= fields


def test_null_sink_accepts_and_discards() -> None:
    sink: TelemetrySink = NullSink()
    sink.emit(VadSpeechStarted(trace=TraceContext.new(), at_ns=1))


def test_flight_recorder_is_bounded_and_counts_evictions() -> None:
    trace = TraceContext.new()
    recorder = FlightRecorder(capacity=2)
    for at in range(5):
        recorder.emit(VadSpeechStarted(trace=trace, at_ns=at))
    assert [event.at_ns for event in recorder.events] == [3, 4]
    assert recorder.evicted == 3
    assert repr(recorder) == "FlightRecorder(len=2, evicted=3)"


def test_flight_recorder_filters_by_type_and_clears() -> None:
    trace = TraceContext.new()
    recorder = FlightRecorder()
    recorder.emit(VadSpeechStarted(trace=trace, at_ns=1))
    recorder.emit(VadSpeechStopped(trace=trace, at_ns=2, speech_duration_ns=1))
    assert [event.at_ns for event in recorder.of_type(VadSpeechStopped)] == [2]
    recorder.clear()
    assert recorder.events == ()


def test_flight_recorder_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        FlightRecorder(capacity=0)


def test_ingress_gap_reports_lost_packet_count() -> None:
    gap = IngressGap(trace=TraceContext.new(), at_ns=1, expected_sequence=7, received_sequence=10)
    assert gap.lost == 3
