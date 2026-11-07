"""Barge-in: < 15 ms to DRAINING, no task leaks, no unhandled cancellations, no lost turns."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from duplex_support import RecordingEgress, StubbornEgress, park, slow_cleanup, until
from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.core.ring import AudioRingBuffer, OverflowPolicy, RingClosedError
from voice_ai_realtime_gateway.domain.duplex import BARGE_IN_BUDGET_NS, DuplexSession
from voice_ai_realtime_gateway.domain.states import SessionState, SessionTrigger
from voice_ai_realtime_gateway.telemetry.events import BargeInHandled, TaskFailed
from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

if TYPE_CHECKING:
    from conftest import FakeClock, FrameFactory
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

pytestmark = pytest.mark.usefixtures("strict_loop")

S = SessionState


async def speaking(
    session: DuplexSession, make_frame: FrameFactory, frames: int = 3
) -> TraceContext:
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    span = session.begin_response()
    for _ in range(frames):
        await session.push_audio(make_frame(trace=span))
    return span


async def test_barge_in_while_speaking_meets_budget_with_real_clock(
    recorder: FlightRecorder, make_frame: FrameFactory
) -> None:
    egress = RecordingEgress(gated=True)
    ring = AudioRingBuffer(
        capacity_frames=256, capacity_bytes=1 << 20, policy=OverflowPolicy.BACKPRESSURE
    )
    session = DuplexSession(egress=egress, sink=recorder, ring=ring)  # real monotonic clock
    span = await speaking(session, make_frame, frames=200)
    await until(egress.in_send.is_set)
    for index in range(16):
        _ = session.spawn(park(), name=f"gen-{index}")
    await asyncio.sleep(0)

    detected = time.monotonic_ns()
    report = await session.on_user_speech_started(at_ns=detected)

    assert report is not None
    assert report.detect_to_draining_ns < BARGE_IN_BUDGET_NS
    assert report.within_budget
    assert report.cancelled_tasks == 17, "16 generation tasks + the egress pump"
    assert report.dropped_frames == 199, "one frame was in flight inside send()"
    assert report.stragglers == 0
    assert report.resumed_listening
    assert session.state is S.LISTENING
    assert session.active_tasks == 0
    assert len(ring) == 0
    assert egress.sent == [], "no audio is played after the barge-in"
    (event,) = recorder.of_type(BargeInHandled)
    assert event.within_budget
    assert event.budget_ns == BARGE_IN_BUDGET_NS
    assert event.trace == span
    triggers = [t.trigger for t in session.machine.history[-3:]]
    assert triggers == [
        SessionTrigger.BARGE_IN,
        SessionTrigger.DRAIN,
        SessionTrigger.DRAIN_COMPLETE,
    ]


async def test_critical_section_reaches_draining_without_yielding(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)
    barge = asyncio.create_task(session.on_user_speech_started())
    states_seen: list[SessionState] = []
    while not barge.done():
        states_seen.append(session.state)
        await asyncio.sleep(0)
    assert S.BARGED_IN not in states_seen, "BARGED_IN -> DRAINING is atomic"
    assert S.SPEAKING not in states_seen[1:]
    assert (await barge) is not None


async def test_barge_in_while_thinking_cancels_generation(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    session = DuplexSession(egress=RecordingEgress(), sink=recorder, clock=clock, trace=trace)
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    generation = session.spawn(park(), name="llm")
    report = await session.on_user_speech_started()
    assert report is not None
    assert (report.cancelled_tasks, report.dropped_frames) == (1, 0)
    assert generation.cancelled()
    assert session.state is S.LISTENING
    assert not session.first_token.armed


async def test_budget_overrun_is_flagged(
    recorder: FlightRecorder, trace: TraceContext, make_frame: FrameFactory
) -> None:
    from conftest import FakeClock  # noqa: PLC0415 -- runtime import of a test double

    clock = FakeClock(start=0, step=BARGE_IN_BUDGET_NS)
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)
    report = await session.on_user_speech_started(at_ns=0)
    assert report is not None
    assert report.detect_to_draining_ns >= BARGE_IN_BUDGET_NS
    assert not report.within_budget
    assert not recorder.of_type(BargeInHandled)[0].within_budget


async def test_producer_blocked_on_backpressure_is_released_not_leaked(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    egress = RecordingEgress(gated=True)
    ring = AudioRingBuffer(
        capacity_frames=2, capacity_bytes=4096, policy=OverflowPolicy.BACKPRESSURE
    )
    session = DuplexSession(egress=egress, sink=recorder, clock=clock, trace=trace, ring=ring)
    span = await speaking(session, make_frame, frames=0)
    outcome: list[str] = []

    async def tts() -> None:
        try:
            for _ in range(100):
                await session.push_audio(make_frame(trace=span))
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise

    _ = session.spawn(tts(), name="tts")
    await until(lambda: len(ring._putters) == 1)
    report = await session.on_user_speech_started()
    assert report is not None
    assert outcome == ["cancelled"]
    assert session.active_tasks == 0
    assert session.state is S.LISTENING


async def test_external_producer_racing_a_barge_in_gets_ring_closed(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    egress = RecordingEgress(gated=True)
    ring = AudioRingBuffer(
        capacity_frames=1, capacity_bytes=4096, policy=OverflowPolicy.BACKPRESSURE
    )
    session = DuplexSession(egress=egress, sink=recorder, clock=clock, trace=trace, ring=ring)
    span = await speaking(session, make_frame, frames=2)
    blocked = asyncio.create_task(session.push_audio(make_frame(trace=span)))  # not supervised
    await until(lambda: len(ring._putters) == 1)
    await session.on_user_speech_started()
    with pytest.raises(RingClosedError):
        await blocked
    assert len(ring) == 0


async def test_second_onset_during_drain_does_not_re_enter(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)
    _ = session.spawn(slow_cleanup(5), name="gen")
    await asyncio.sleep(0)
    first = asyncio.create_task(session.on_user_speech_started())
    await until(lambda: session.state is S.DRAINING)
    assert await session.on_user_speech_started() is None
    report = await first
    assert report is not None
    assert len(recorder.of_type(BargeInHandled)) == 1


async def test_turn_end_during_drain_is_replayed_once_listening(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)
    _ = session.spawn(slow_cleanup(5), name="gen")
    await asyncio.sleep(0)
    barge = asyncio.create_task(session.on_user_speech_started())
    await until(lambda: session.state is S.DRAINING)
    assert session.on_user_speech_stopped() is None, "deferred, not dropped"
    report = await barge
    assert report is not None
    assert report.resumed_listening
    assert session.state is S.PROCESSING_THOUGHT, "the interrupting utterance became a turn"
    assert session.first_token.armed
    await session.aclose()


async def test_speech_restart_during_drain_cancels_the_deferred_turn_end(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)
    _ = session.spawn(slow_cleanup(5), name="gen")
    await asyncio.sleep(0)
    barge = asyncio.create_task(session.on_user_speech_started())
    await until(lambda: session.state is S.DRAINING)
    session.on_user_speech_stopped()
    assert await session.on_user_speech_started() is None
    await barge
    assert session.state is S.LISTENING, "user is still talking: no premature turn end"


async def test_straggler_during_barge_in_fails_closed(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    egress = StubbornEgress()
    session = DuplexSession(
        egress=egress, sink=recorder, clock=clock, trace=trace, drain_timeout_s=0.01
    )
    await speaking(session, make_frame, frames=1)
    await until(egress.in_send.is_set)
    report = await session.on_user_speech_started()
    assert report is not None
    assert report.stragglers == 1
    assert not report.resumed_listening
    assert session.state is S.TERMINATED
    # When the misbehaving pump finally exits it hits a terminated machine; the resulting
    # failure is captured and reported, never left unretrieved.
    egress.release.set()
    await until(lambda: session.active_tasks == 0)
    assert recorder.of_type(TaskFailed)[0].error_type == "InvalidTransitionError"
    await session.aclose()


async def test_task_failing_during_its_cancellation_faults_and_skips_drain_complete(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)

    async def bad_cleanup() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            msg = "cleanup failed"
            raise OSError(msg)

    _ = session.spawn(bad_cleanup(), name="gen")
    await asyncio.sleep(0)
    report = await session.on_user_speech_started()
    assert report is not None
    assert report.stragglers == 0
    assert not report.resumed_listening
    assert session.state is S.TERMINATED
    assert SessionTrigger.DRAIN_COMPLETE not in [t.trigger for t in session.machine.history]


async def test_cancelled_barge_in_caller_propagates_and_close_completes_the_drain(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)
    _ = session.spawn(slow_cleanup(50), name="gen")
    await asyncio.sleep(0)
    barge = asyncio.create_task(session.on_user_speech_started())
    await until(lambda: session.state is S.DRAINING)
    barge.cancel()
    with pytest.raises(asyncio.CancelledError):
        await barge
    assert session.machine.state is S.DRAINING
    await session.aclose()
    assert session.state is S.TERMINATED
    assert session.active_tasks == 0


@pytest.mark.parametrize("close_first", [True, False])
async def test_close_racing_an_in_flight_barge_in_terminates_exactly_once(
    recorder: FlightRecorder,
    clock: FakeClock,
    trace: TraceContext,
    make_frame: FrameFactory,
    *,
    close_first: bool,
) -> None:
    session = DuplexSession(
        egress=RecordingEgress(gated=True), sink=recorder, clock=clock, trace=trace
    )
    await speaking(session, make_frame)
    _ = session.spawn(slow_cleanup(5), name="gen")
    await asyncio.sleep(0)
    barge = asyncio.create_task(session.on_user_speech_started())
    await until(lambda: session.state is S.DRAINING)
    closer = asyncio.create_task(session.aclose())
    if close_first:
        await closer
        report = await barge
    else:
        report = await barge
        await closer
    assert report is not None
    assert not report.resumed_listening
    assert session.state is S.TERMINATED
    terminal = [t for t in session.machine.history if t.target is S.TERMINATED]
    assert len(terminal) == 1


async def test_natural_completion_racing_barge_in_never_double_transitions(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, make_frame: FrameFactory
) -> None:
    egress = RecordingEgress()
    session = DuplexSession(egress=egress, sink=recorder, clock=clock, trace=trace)
    await speaking(session, make_frame, frames=2)
    session.end_response()  # pump will drain and fire RESPONSE_AUDIO_FINISHED ...
    report = await session.on_user_speech_started()  # ... but the user barges in first
    assert report is not None
    assert session.state is S.LISTENING
    assert SessionTrigger.RESPONSE_AUDIO_FINISHED not in [
        t.trigger for t in session.machine.history
    ]


# ------------------------------------------------------------------------ schedule fuzzing


def _frame(trace: TraceContext, sequence: int) -> AudioFrame:
    return AudioFrame.wrap(
        bytes(64),
        codec=AudioCodec.PCM16,
        sample_rate_hz=16_000,
        channels=1,
        duration_ns=20_000_000,
        sequence=sequence,
        origin_ns=0,
        trace=trace,
    )


@settings(max_examples=75)
@given(
    frames=st.integers(min_value=0, max_value=20),
    barge_after_yields=st.integers(min_value=0, max_value=40),
    egress_yields=st.integers(min_value=0, max_value=3),
    log2_capacity=st.integers(min_value=0, max_value=3),
    end_response=st.booleans(),
)
def test_barge_in_at_any_point_of_a_response_is_safe(
    frames: int,
    barge_after_yields: int,
    egress_yields: int,
    log2_capacity: int,
    *,
    end_response: bool,
) -> None:
    errors: list[dict[str, object]] = []

    class PacedEgress:
        def __init__(self) -> None:
            self.sent: list[int] = []

        async def send(self, frame: AudioFrame, /) -> None:
            for _ in range(egress_yields):
                await asyncio.sleep(0)
            self.sent.append(frame.sequence)

    async def scenario() -> tuple[DuplexSession, PacedEgress]:
        asyncio.get_running_loop().set_exception_handler(lambda _l, ctx: errors.append(dict(ctx)))
        egress = PacedEgress()
        ring = AudioRingBuffer(
            capacity_frames=1 << log2_capacity,
            capacity_bytes=1 << 16,
            policy=OverflowPolicy.BACKPRESSURE,
        )
        session = DuplexSession(egress=egress, sink=FlightRecorder(), ring=ring)
        session.connect()
        session.connected()
        session.on_user_speech_stopped()
        session.begin_response()
        span = session.trace

        async def tts() -> None:
            for sequence in range(frames):
                await session.push_audio(_frame(span, sequence))
            if end_response:
                session.end_response()

        _ = session.spawn(tts(), name="tts")
        for _ in range(barge_after_yields):
            await asyncio.sleep(0)
        await session.on_user_speech_started()
        await session.aclose()
        for _ in range(3):
            await asyncio.sleep(0)
        leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert not leaked
        return session, egress

    session, egress = asyncio.run(scenario())
    assert errors == []
    assert session.state is S.TERMINATED
    assert session.active_tasks == 0
    assert egress.sent == list(range(len(egress.sent))), "played audio is an in-order prefix"
    history = session.machine.history
    stamps = [t.lamport for t in history]
    assert stamps == sorted(set(stamps))
    assert sum(t.target is S.TERMINATED for t in history) == 1
