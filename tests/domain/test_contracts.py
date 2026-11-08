"""Specification constants and boundaries (each pins a behaviour a mutant once escaped)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from duplex_support import RecordingEgress, slow_cleanup, until
from voice_ai_realtime_gateway.domain.duplex import (
    BARGE_IN_BUDGET_NS,
    DEFAULT_EGRESS_RING_BYTES,
    DEFAULT_EGRESS_RING_FRAMES,
    DuplexSession,
)
from voice_ai_realtime_gateway.domain.states import SessionState
from voice_ai_realtime_gateway.telemetry.events import BargeInHandled

if TYPE_CHECKING:
    from conftest import FakeClock, FrameFactory
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")


def test_specified_constants() -> None:
    assert BARGE_IN_BUDGET_NS == 15_000_000, "the 15 ms barge-in budget"
    assert DEFAULT_EGRESS_RING_FRAMES == 256
    assert DEFAULT_EGRESS_RING_BYTES == 524_288


async def test_smallest_budget_is_valid_and_sessions_start_unfaulted(
    recorder: FlightRecorder,
) -> None:
    session = DuplexSession(egress=RecordingEgress(), sink=recorder, barge_in_budget_ns=1)
    assert not session.faulted
    await session.aclose()
    assert not session.faulted, "a graceful close is not a fault"


async def test_latency_equal_to_the_budget_is_over_budget(
    recorder: FlightRecorder, trace: TraceContext, make_frame: FrameFactory
) -> None:
    from conftest import FakeClock  # noqa: PLC0415 -- runtime import of a test double

    clock = FakeClock(start=BARGE_IN_BUDGET_NS, step=0)
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    span = session.begin_response()
    await session.push_audio(make_frame(trace=span))
    report = await session.on_user_speech_started(at_ns=0)
    assert report is not None
    assert report.detect_to_draining_ns == BARGE_IN_BUDGET_NS
    assert not report.within_budget, "the budget is a strict upper bound"
    assert not recorder.of_type(BargeInHandled)[0].within_budget
    await session.aclose()


async def test_a_replayed_turn_end_is_consumed_exactly_once(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    span = session.begin_response()
    await session.push_audio(make_frame(trace=span))
    _ = session.spawn(slow_cleanup(5), name="gen")
    await asyncio.sleep(0)
    barge = asyncio.create_task(session.on_user_speech_started())
    await until(lambda: session.state is SessionState.DRAINING)
    session.on_user_speech_stopped()  # deferred until the drain completes
    await barge
    states = [session.state]  # replayed once
    assert await session.interrupt() is not None  # a client cancel, with no new VAD onset
    states.append(session.state)  # the deferred turn end must not be replayed again
    assert states == [SessionState.PROCESSING_THOUGHT, SessionState.LISTENING]
    await session.aclose()
