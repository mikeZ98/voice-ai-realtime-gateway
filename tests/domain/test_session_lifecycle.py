"""Supervisor-facing session API: listeners, termination signal, interrupt and abort."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from duplex_support import RecordingEgress, park, until
from voice_ai_realtime_gateway.core.supervisor import JoinReport
from voice_ai_realtime_gateway.domain.duplex import DuplexSession
from voice_ai_realtime_gateway.domain.machine import SessionStateMachine, Transition
from voice_ai_realtime_gateway.domain.states import SessionState, SessionTrigger
from voice_ai_realtime_gateway.telemetry.events import StateTransitioned

if TYPE_CHECKING:
    from conftest import FakeClock, FrameFactory
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")

S = SessionState
T = SessionTrigger


def make(recorder: FlightRecorder, clock: FakeClock, trace: TraceContext) -> DuplexSession:
    return DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )


async def test_listener_sees_every_commit_in_order_after_telemetry(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    seen: list[tuple[Transition, int]] = []
    machine = SessionStateMachine(
        trace=trace,
        sink=recorder,
        on_transition=lambda t: seen.append((t, len(recorder.of_type(StateTransitioned)))),
    )
    machine.fire(T.CONNECT)
    machine.fire(T.CONNECTED)
    with pytest.raises(ValueError, match="observed"):
        machine.fire(T.CLOSE, observed=-1)
    assert [t.target for t, _ in seen] == [S.CONNECTING, S.LISTENING]
    assert [emitted for _, emitted in seen] == [1, 2], "listener runs after the event is emitted"
    assert [t for t, _ in seen] == list(machine.history)


async def test_termination_signal_on_graceful_close(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make(recorder, clock, trace)
    waiter = asyncio.create_task(session.wait_terminated())
    await asyncio.sleep(0)
    assert not waiter.done()
    observed = [session.terminated]
    await session.aclose()
    await waiter
    observed.append(session.terminated)
    assert observed == [False, True]
    assert not session.faulted


async def test_abort_faults_cancels_and_reaps(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = make(recorder, clock, trace)
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    _ = session.spawn(park(), name="llm")
    span = session.begin_response()
    await session.push_audio(make_frame(trace=span))
    report = await session.abort()
    assert report == JoinReport(finished=2, stragglers=0)
    assert session.terminated
    assert session.faulted
    assert session.ring.closed
    assert session.active_tasks == 0
    assert await session.abort() == JoinReport(finished=0, stragglers=0), "idempotent"
    assert session.faulted


async def test_interrupt_cancels_output_without_touching_turn_bookkeeping(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = make(recorder, clock, trace)
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    span = session.begin_response()
    await session.push_audio(make_frame(trace=span))
    report = await session.interrupt(at_ns=clock.now)
    assert report is not None
    assert report.resumed_listening
    assert report.cancelled_tasks == 1
    assert session.state is S.LISTENING
    assert session._speech_started_at is None, "no VAD onset was recorded"


async def test_interrupt_outside_agent_output_is_a_no_op(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make(recorder, clock, trace)
    assert await session.interrupt() is None
    session.connect()
    session.connected()
    assert await session.interrupt() is None
    assert session.state is S.LISTENING


async def test_session_fault_from_a_task_sets_the_termination_signal(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make(recorder, clock, trace)
    session.connect()
    session.connected()
    session.on_user_speech_stopped()

    async def explode() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    _ = session.spawn(explode(), name="llm")
    await asyncio.wait_for(session.wait_terminated(), timeout=1)
    await until(lambda: session.active_tasks == 0)
    assert session.faulted
