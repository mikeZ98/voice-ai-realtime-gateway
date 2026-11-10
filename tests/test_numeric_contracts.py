"""NaN and +/-Inf never enter a contract.

``nan <= 0`` and ``nan < 1`` are both ``False``, so a plain comparison guard admits NaN,
and ``inf`` passes any lower bound. Every float-carrying configuration and event of the
gateway is pinned here (the drain controller and the Deepgram parser are covered next to
their other validation tests).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from duplex_support import RecordingEgress
from voice_ai_realtime_gateway._validation import (
    is_non_negative_finite,
    is_positive_finite,
    is_unit_interval,
)
from voice_ai_realtime_gateway.domain.duplex import DuplexSession
from voice_ai_realtime_gateway.inference.agent import AgentConfig
from voice_ai_realtime_gateway.inference.ports import TranscriptChunk
from voice_ai_realtime_gateway.production.server import ServerConfig, TenantPolicy
from voice_ai_realtime_gateway.telemetry.events import TranscriptReceived
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.sink import NullSink
from voice_ai_realtime_gateway.transport.websocket import TransportLimits

if TYPE_CHECKING:
    from collections.abc import Callable

    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

NON_FINITE = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="-inf"),
]


@pytest.mark.parametrize("value", NON_FINITE)
def test_predicates_reject_non_finite(value: float) -> None:
    assert not is_positive_finite(value)
    assert not is_non_negative_finite(value)
    assert not is_unit_interval(value)


@pytest.mark.parametrize(
    ("value", "positive", "non_negative", "unit"),
    [
        (0.0, False, True, True),
        (-0.0, False, True, True),
        (5e-324, True, True, True),  # smallest subnormal
        (1.0, True, True, True),
        (math.nextafter(1.0, 2.0), True, True, False),
        (-5e-324, False, False, False),
        (1.7976931348623157e308, True, True, False),  # largest finite double
    ],
)
def test_predicate_boundaries(
    value: float, *, positive: bool, non_negative: bool, unit: bool
) -> None:
    assert is_positive_finite(value) is positive
    assert is_non_negative_finite(value) is non_negative
    assert is_unit_interval(value) is unit


# ----------------------------------------------------------------- configurations


def tenant(value: float) -> object:
    return TenantPolicy(max_concurrent_sessions=1, max_session_duration_s=value)


def server(value: float) -> object:
    return ServerConfig(media_timeout_s=value)


def limits(value: float) -> object:
    return TransportLimits(control_rate_per_s=value)


def agent(value: float) -> object:
    return AgentConfig(drain_timeout_s=value)


def duplex(value: float) -> object:
    return DuplexSession(egress=RecordingEgress(), sink=NullSink(), drain_timeout_s=value)


CONFIGS: list[tuple[str, Callable[[float], object], str]] = [
    ("tenant", tenant, "tenant bounds must be positive and finite"),
    ("server", server, "media_timeout_s must be positive and finite"),
    ("limits", limits, "control_rate_per_s must be positive and finite"),
    ("agent", agent, "drain_timeout_s must be positive and finite"),
    ("duplex", duplex, "drain_timeout_s must be positive and finite"),
]


@pytest.mark.parametrize(("build", "message"), [pytest.param(b, m, id=n) for n, b, m in CONFIGS])
@pytest.mark.parametrize("value", [*NON_FINITE, pytest.param(-1.0, id="negative")])
def test_configurations_reject_non_finite_and_negative(
    build: Callable[[float], object], message: str, value: float
) -> None:
    with pytest.raises(ValueError, match=message):
        build(value)


@pytest.mark.parametrize(("build", "message"), [pytest.param(b, m, id=n) for n, b, m in CONFIGS])
def test_configurations_accept_finite_positive(
    build: Callable[[float], object], message: str
) -> None:
    del message
    assert build(1.5) is not None


def test_agent_zero_drain_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="drain_timeout_s"):
        AgentConfig(drain_timeout_s=0.0)


# ----------------------------------------------------------------- STT events


def chunk(trace: TraceContext, confidence: float) -> TranscriptChunk:
    return TranscriptChunk(
        text="hi", is_final=True, confidence=confidence, words=(), trace=trace, at_ns=0
    )


def received(trace: TraceContext, confidence: float) -> TranscriptReceived:
    return TranscriptReceived(
        trace=trace, at_ns=0, is_final=True, text_chars=2, confidence=confidence
    )


EVENTS = [
    pytest.param(chunk, id="TranscriptChunk"),
    pytest.param(received, id="TranscriptReceived"),
]


@pytest.mark.parametrize("build", EVENTS)
@given(confidence=st.floats(allow_nan=True, allow_infinity=True))
def test_confidence_is_accepted_exactly_on_the_closed_unit_interval(
    build: Callable[[TraceContext, float], object], confidence: float
) -> None:
    trace = TraceContext.new()
    probability = math.isfinite(confidence) and 0.0 <= confidence <= 1.0
    if probability:
        assert build(trace, confidence) is not None
    else:
        with pytest.raises(ValueError, match=r"confidence must be a finite number in \[0, 1\]"):
            build(trace, confidence)


@pytest.mark.parametrize("build", EVENTS)
@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_bounds_are_inclusive(
    build: Callable[[TraceContext, float], object], trace: TraceContext, confidence: float
) -> None:
    assert build(trace, confidence) is not None


def test_recorder_fixture_still_accepts_valid_events(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    recorder.emit(received(trace, 0.5))
    assert recorder.of_type(TranscriptReceived) == [received(trace, 0.5)]
