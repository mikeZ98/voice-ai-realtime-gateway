"""Duplex controller: lifecycle, turn-taking, playback, telemetry and fault handling."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

import pytest

from duplex_support import BrokenEgress, RecordingEgress, StubbornEgress, park, until
from voice_ai_realtime_gateway.core.ring import AudioRingBuffer, OverflowPolicy, RingClosedError
from voice_ai_realtime_gateway.core.supervisor import JoinReport
from voice_ai_realtime_gateway.domain.duplex import (
    DEFAULT_EGRESS_RING_BYTES,
    DEFAULT_EGRESS_RING_FRAMES,
    DuplexSession,
)
from voice_ai_realtime_gateway.domain.errors import (
    InvalidTransitionError,
    SessionStateError,
    StaleSpanError,
)
from voice_ai_realtime_gateway.domain.states import SessionState, SessionTrigger
from voice_ai_realtime_gateway.telemetry.events import (
    FirstTokenLatency,
    PacketLatency,
    StateTransitioned,
    TaskFailed,
    VadSpeechStarted,
    VadSpeechStopped,
)

if TYPE_CHECKING:
    from conftest import FakeClock, FrameFactory
    from voice_ai_realtime_gateway.domain.duplex import AudioEgress
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")

S = SessionState


def make_session(
    egress: AudioEgress,
    recorder: FlightRecorder,
    clock: FakeClock,
    trace: TraceContext,
    ring: AudioRingBuffer | None = None,
) -> DuplexSession:
    return DuplexSession(egress=egress, sink=recorder, trace=trace, clock=clock, ring=ring)


async def listening(session: DuplexSession) -> None:
    session.connect()
    session.connected()
    assert session.state is S.LISTENING


async def thinking(session: DuplexSession) -> None:
    await listening(session)
    assert await session.on_user_speech_started() is None
    assert session.on_user_speech_stopped() is not None
    assert session.state is S.PROCESSING_THOUGHT


# ------------------------------------------------------------------------ construction


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"drain_timeout_s": 0.0}, "drain_timeout_s"), ({"barge_in_budget_ns": 0}, "budget")],
)
async def test_configuration_is_validated(
    kwargs: dict[str, float], match: str, recorder: FlightRecorder
) -> None:
    with pytest.raises(ValueError, match=match):
        DuplexSession(egress=RecordingEgress(), sink=recorder, **kwargs)  # type: ignore[arg-type]


async def test_defaults_build_a_bounded_backpressure_egress_ring(recorder: FlightRecorder) -> None:
    session = DuplexSession(egress=RecordingEgress(), sink=recorder)
    ring = session.ring
    assert ring.capacity_frames == DEFAULT_EGRESS_RING_FRAMES
    assert ring.capacity_bytes == DEFAULT_EGRESS_RING_BYTES
    assert ring.policy is OverflowPolicy.BACKPRESSURE
    assert session.state is S.IDLE
    assert session.machine.state is S.IDLE
    assert session.active_tasks == 0
    assert repr(session) == "DuplexSession(state=idle, tasks=0)"


# ------------------------------------------------------------------------ happy path


async def test_full_turn_streams_audio_in_order_and_returns_to_listening(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    egress = RecordingEgress()
    session = make_session(egress, recorder, clock, trace)
    await listening(session)

    clock.now = 1_000
    assert await session.on_user_speech_started() is None
    clock.now = 501_000
    turn = session.on_user_speech_stopped()
    assert turn is not None
    assert turn.trigger is SessionTrigger.USER_TURN_ENDED
    response_trace = session.trace
    assert response_trace.session_id == trace.session_id
    assert response_trace.span_id != trace.span_id, "each response gets its own audio span"

    clock.now = 800_000
    first = session.mark_first_token()
    assert first == FirstTokenLatency(trace=response_trace, at_ns=800_000, latency_ns=299_000)
    assert session.mark_first_token() is None, "first token is reported once per turn"

    assert session.begin_response() == response_trace
    frames = [make_frame(320, trace=response_trace, origin_ns=700_000) for _ in range(5)]
    for frame in frames:
        assert await session.push_audio(frame) == 0
    session.end_response()
    await until(lambda: session.state is S.LISTENING)
    await until(lambda: session.active_tasks == 0)  # pump is reaped one loop turn later

    assert egress.sent == frames
    stopped = recorder.of_type(VadSpeechStopped)[0]
    assert stopped.speech_duration_ns == 500_000
    assert len(recorder.of_type(VadSpeechStarted)) == 1
    latencies = recorder.of_type(PacketLatency)
    assert [p.sequence for p in latencies] == [f.sequence for f in frames]
    assert all(p.trace == response_trace for p in latencies)
    assert session.glass_to_glass.window.count == 5
    assert session.first_token.window.count == 1
    targets = [event.target for event in recorder.of_type(StateTransitioned)]
    assert targets == ["connecting", "listening", "processing_thought", "speaking", "listening"]


async def test_skip_response_returns_to_listening_and_disarms_first_token(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    await thinking(session)
    transition = session.skip_response(observed=500)
    assert transition.target is S.LISTENING
    assert transition.lamport == 501
    assert not session.first_token.armed


async def test_signalling_merges_remote_lamport_timestamps(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    assert session.connect(observed=10).lamport == 11
    assert session.connected(observed=20).lamport == 21
    session.on_user_speech_stopped()
    session.begin_response(observed=40)
    assert session.machine.lamport == 41
    await session.aclose()


async def test_speech_stop_outside_a_turn_is_recorded_but_ignored(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    assert session.on_user_speech_stopped(at_ns=5) is None
    assert session.state is S.IDLE
    (event,) = recorder.of_type(VadSpeechStopped)
    assert (event.at_ns, event.speech_duration_ns) == (5, None)


async def test_speech_start_while_listening_is_not_a_barge_in(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    await listening(session)
    assert await session.on_user_speech_started(at_ns=77) is None
    assert session.state is S.LISTENING
    assert recorder.of_type(VadSpeechStarted)[0].at_ns == 77


# ------------------------------------------------------------------------ guards


async def test_operations_are_rejected_in_the_wrong_state(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    coro = park()
    with pytest.raises(SessionStateError, match="cannot spawn"):
        _ = session.spawn(coro, name="gen")
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED, "refused coroutine is closed"
    with pytest.raises(SessionStateError, match="cannot push"):
        await session.push_audio(make_frame())
    with pytest.raises(SessionStateError, match="cannot end"):
        session.end_response()
    with pytest.raises(InvalidTransitionError):
        session.begin_response()
    with pytest.raises(InvalidTransitionError):
        session.connected()


async def test_frames_from_a_stale_span_are_fenced(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    await thinking(session)
    session.begin_response()
    with pytest.raises(StaleSpanError):
        await session.push_audio(make_frame(trace=trace))  # the pre-turn span
    await session.aclose()


async def test_push_blocked_by_backpressure_is_released_by_close(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    egress = RecordingEgress(gated=True)
    ring = AudioRingBuffer(
        capacity_frames=1, capacity_bytes=4096, policy=OverflowPolicy.BACKPRESSURE
    )
    session = make_session(egress, recorder, clock, trace, ring)
    await thinking(session)
    span = session.begin_response()
    await session.push_audio(make_frame(trace=span))  # pump takes it, blocks in send
    await until(egress.in_send.is_set)
    await session.push_audio(make_frame(trace=span))  # fills the ring
    blocked = asyncio.create_task(session.push_audio(make_frame(trace=span)))
    await until(lambda: len(ring._putters) == 1)
    report = await session.aclose()
    with pytest.raises(RingClosedError):
        await blocked
    assert report == JoinReport(finished=1, stragglers=0)
    assert session.state is S.TERMINATED


# ------------------------------------------------------------------------ shutdown


async def test_close_from_idle_terminates_without_draining(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    assert await session.aclose() == JoinReport(finished=0, stragglers=0)
    assert session.state is S.TERMINATED
    targets = [event.target for event in recorder.of_type(StateTransitioned)]
    assert targets == ["terminated"]


async def test_close_drains_then_terminates_and_is_idempotent(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    await thinking(session)
    _ = session.spawn(park(), name="generation")
    report = await session.aclose()
    assert report == JoinReport(finished=1, stragglers=0)
    assert session.state is S.TERMINATED
    assert session.ring.closed
    targets = [event.target for event in recorder.of_type(StateTransitioned)]
    assert targets[-2:] == ["draining", "terminated"]
    assert await session.aclose() == JoinReport(finished=0, stragglers=0)
    with pytest.raises(SessionStateError):
        _ = session.spawn(park(), name="late")


async def test_close_with_a_straggler_faults_instead_of_hanging(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    egress = StubbornEgress()
    session = DuplexSession(
        egress=egress, sink=recorder, trace=trace, clock=clock, drain_timeout_s=0.01
    )
    await thinking(session)
    span = session.begin_response()
    await session.push_audio(make_frame(trace=span))
    await until(egress.in_send.is_set)
    report = await session.aclose()
    assert report.stragglers == 1
    assert session.state is S.TERMINATED
    assert session.machine.history[-1].trigger is SessionTrigger.FAULT
    egress.release.set()
    await until(lambda: session.active_tasks == 0)
    assert egress.swallowed >= 1


# ------------------------------------------------------------------------ faults


async def test_failed_generation_task_faults_the_session(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = make_session(RecordingEgress(), recorder, clock, trace)
    await thinking(session)

    async def generation() -> None:
        msg = "model server 503"
        raise RuntimeError(msg)

    _ = session.spawn(generation(), name="llm")
    await until(lambda: session.state is S.TERMINATED)
    (failed,) = recorder.of_type(TaskFailed)
    assert (failed.task_name, failed.error_type) == ("llm", "RuntimeError")
    assert session.ring.closed
    assert not session.first_token.armed
    await session.aclose()


async def test_transport_failure_in_pump_faults_the_session(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = make_session(BrokenEgress(), recorder, clock, trace)
    await thinking(session)
    span = session.begin_response()
    await session.push_audio(make_frame(trace=span))
    await until(lambda: session.state is S.TERMINATED)
    (failed,) = recorder.of_type(TaskFailed)
    assert failed.error_type == "ConnectionResetError"
    assert failed.task_name == f"egress-pump:{span.span_id}"
    await session.aclose()
