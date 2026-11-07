"""End-to-end: client socket -> transport -> VAD -> session -> egress -> client socket."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Protocol

import pytest

from audio_support import LOUD, pcm_frame
from duplex_support import until
from transport_support import Client
from voice_ai_realtime_gateway.domain.states import SessionState
from voice_ai_realtime_gateway.pipeline.runner import GatewaySessionRunner
from voice_ai_realtime_gateway.telemetry.events import TaskFailed
from voice_ai_realtime_gateway.transport.base import CloseInitiator
from voice_ai_realtime_gateway.transport.control import (
    CancelResponse,
    Hangup,
    PlaybackFlush,
    VadConfigUpdate,
)
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError
from voice_ai_realtime_gateway.transport.framing import DecodedPacket
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.transport.websocket import HeartbeatConfig, WebSocketVoiceTransport
from voice_ai_realtime_gateway.vad.base import VadConfig
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from voice_ai_realtime_gateway.domain.duplex import DuplexSession
    from voice_ai_realtime_gateway.pipeline.runner import RunOutcome, TurnHandler
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")

VAD = VadConfig(min_speech_ms=40, hangover_ms=60)  # 2 loud frames start, 3 quiet frames end
QUIET = HeartbeatConfig(interval_s=3600.0, timeout_s=3600.0)
DEADLINE = 5.0


class Call:
    """A running gateway session plus the client end of its socket."""

    def __init__(
        self,
        recorder: FlightRecorder,
        trace: TraceContext,
        on_user_turn: TurnHandler | None,
        capacity: int,
    ) -> None:
        server, peer = MemoryWebSocket.pair(capacity=capacity)
        self.transport = WebSocketVoiceTransport(
            server, trace=trace, sink=recorder, heartbeat=QUIET
        )
        self.runner = GatewaySessionRunner(
            transport=self.transport,
            vad=EnergyVAD(VAD),
            sink=recorder,
            on_user_turn=on_user_turn,
        )
        self.client = Client(peer)
        self.trace = trace
        self.sequence = 0
        self.task: asyncio.Task[RunOutcome] = asyncio.create_task(self.runner.run())

    async def speak(self, frames: int, amplitude: int = LOUD) -> None:
        for _ in range(frames):
            await self.client.audio(
                pcm_frame(self.trace, amplitude=amplitude, sequence=self.sequence)
            )
            self.sequence += 1

    async def pause(self, frames: int) -> None:
        await self.speak(frames, amplitude=0)

    async def outcome(self) -> RunOutcome:
        return await asyncio.wait_for(self.task, timeout=DEADLINE)


def responder(frames: int) -> TurnHandler:
    def on_turn(session: DuplexSession) -> None:
        async def respond() -> None:
            span = session.begin_response()
            for sequence in range(frames):
                frame = pcm_frame(
                    span, amplitude=LOUD, sequence=sequence, origin_ns=time.monotonic_ns()
                )
                await session.push_audio(frame)
            session.end_response()

        _ = session.spawn(respond(), name="tts")

    return on_turn


class CallFactory(Protocol):
    def __call__(self, on_user_turn: TurnHandler | None = ..., capacity: int = ...) -> Call: ...


@pytest.fixture
async def call(recorder: FlightRecorder, trace: TraceContext) -> AsyncIterator[CallFactory]:
    calls: list[Call] = []

    def start(on_user_turn: TurnHandler | None = None, capacity: int = 64) -> Call:
        started = Call(recorder, trace, on_user_turn, capacity)
        calls.append(started)
        return started

    yield start
    for leftover in calls:  # a failed test must not cascade into leaked-task errors
        if not leftover.task.done():
            leftover.task.cancel()
        with contextlib.suppress(BaseException):
            await leftover.task


async def receive_audio(client: Client, count: int) -> list[DecodedPacket]:
    packets: list[DecodedPacket] = []
    while len(packets) < count:
        message = await asyncio.wait_for(client.receive(), timeout=DEADLINE)
        assert isinstance(message, DecodedPacket)
        packets.append(message)
    return packets


async def drain_until_closed(socket: MemoryWebSocket) -> None:
    while True:
        await socket.recv()


# ------------------------------------------------------------------------ conversation


async def test_full_turn_from_speech_to_agent_audio_to_hangup(call: CallFactory) -> None:
    c = call(on_user_turn=responder(frames=5))
    await c.pause(2)
    await c.speak(4)
    await c.pause(3)  # hangover elapses: user turn ends, agent responds
    packets = await receive_audio(c.client, 5)
    assert [p.frame.sequence for p in packets] == list(range(5))
    assert all(p.frame.trace.session_id == c.trace.session_id for p in packets)
    assert packets[0].frame.trace.span_id != c.trace.span_id, "agent audio has its own span"
    await until(lambda: c.runner.session.state is SessionState.LISTENING)
    await c.client.control(Hangup("done talking"))
    outcome = await c.outcome()
    assert (outcome.close_code, outcome.close_reason, outcome.closed_by) == (
        1000,
        "done talking",
        CloseInitiator.LOCAL,
    )
    assert (outcome.final_state, outcome.faulted) == (SessionState.TERMINATED, False)
    assert (outcome.turns, outcome.barge_ins) == (1, 0)
    assert repr(c.runner) == "GatewaySessionRunner(state=terminated, turns=1, barge_ins=0)"
    assert c.runner.transport is c.transport


async def test_turns_are_counted_without_a_handler(call: CallFactory) -> None:
    c = call()
    for _ in range(2):
        await c.speak(2)
        await c.pause(3)
        await until(lambda: c.runner.session.state is SessionState.PROCESSING_THOUGHT)
        c.runner.session.skip_response()
    await c.client.control(Hangup(""))
    outcome = await c.outcome()
    assert outcome.turns == 2
    assert outcome.close_reason == "client hangup"


async def test_each_turn_is_dispatched_exactly_once_while_thinking(call: CallFactory) -> None:
    dispatched: list[str] = []
    c = call(on_user_turn=lambda session: dispatched.append(session.trace.span_id))
    await c.speak(2)
    await c.pause(3)  # turn ends; the handler does not respond, the agent keeps thinking
    await c.pause(10)  # more audio arrives during PROCESSING_THOUGHT
    await until(lambda: c.transport.stats().frames_in == 15 and len(c.transport.ingress_ring) == 0)
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(dispatched) == 1
    assert c.runner.session.state is SessionState.PROCESSING_THOUGHT
    await c.client.control(Hangup("bye"))
    assert (await c.outcome()).turns == 1


async def test_barge_in_drains_playback_and_flushes_the_client(call: CallFactory) -> None:
    c = call(on_user_turn=responder(frames=400), capacity=8)
    await c.speak(2)
    await c.pause(3)
    first = await receive_audio(c.client, 3)  # agent is talking; client buffer stays full
    interrupted = first[0].frame.trace.span_id
    await until(lambda: c.runner.session.state is SessionState.SPEAKING)
    await c.speak(2)  # user talks over the agent
    stale = 0
    while not isinstance(message := await c.client.receive(), PlaybackFlush):
        assert isinstance(message, DecodedPacket)
        assert message.frame.trace.span_id == interrupted
        stale += 1
    assert stale < 400 - 3, "playback was cut short, not played out"
    await until(lambda: c.runner.session.state is SessionState.LISTENING)
    await c.pause(3)  # the interrupting utterance ends: a second turn starts
    (reply,) = await receive_audio(c.client, 1)
    assert reply.frame.trace.span_id != interrupted, "nothing of the old span after the flush"
    assert reply.frame.sequence == 0
    await c.client.socket.close()
    outcome = await c.outcome()
    assert (outcome.turns, outcome.barge_ins) == (2, 1)
    assert outcome.closed_by is CloseInitiator.REMOTE
    assert not outcome.faulted


async def test_client_cancel_interrupts_only_when_the_agent_is_active(call: CallFactory) -> None:
    c = call(on_user_turn=responder(frames=400), capacity=8)
    await c.client.control(CancelResponse())  # nothing to cancel: no flush
    await until(lambda: c.transport.stats().control_in == 1 and len(c.transport.control) == 0)
    for _ in range(5):
        await asyncio.sleep(0)
    assert c.runner._barge_ins == 0
    await c.speak(2)
    await c.pause(3)
    await receive_audio(c.client, 1)
    await c.client.control(CancelResponse())
    while not isinstance(await c.client.receive(), PlaybackFlush):
        pass
    await until(lambda: c.runner.session.state is SessionState.LISTENING)
    await c.client.control(Hangup("bye"))
    assert (await c.outcome()).barge_ins == 1


async def test_vad_is_reconfigurable_over_the_control_channel(call: CallFactory) -> None:
    c = call()
    tuned = VadConfig(start_threshold_dbfs=-20.0, stop_threshold_dbfs=-30.0, hangover_ms=500)
    await c.client.control(VadConfigUpdate(tuned))
    await until(lambda: c.runner.vad.config == tuned)
    await c.client.control(Hangup("bye"))
    await c.outcome()


# ------------------------------------------------------------------------ disconnects


async def test_client_disconnect_mid_response_ends_gracefully(call: CallFactory) -> None:
    c = call(on_user_turn=responder(frames=400), capacity=4)
    await c.speak(2)
    await c.pause(3)
    await receive_audio(c.client, 1)
    await c.client.socket.close(CloseCode.GOING_AWAY, "tab closed")
    outcome = await c.outcome()
    assert (outcome.close_code, outcome.closed_by) == (1001, CloseInitiator.REMOTE)
    assert outcome.final_state is SessionState.TERMINATED
    assert not outcome.faulted, "a disconnect is a CLOSE, not a FAULT"
    assert c.runner.session.active_tasks == 0


async def test_network_drop_ends_the_session(call: CallFactory) -> None:
    c = call(on_user_turn=responder(frames=10))
    await c.speak(2)
    await c.client.socket.sever()
    outcome = await c.outcome()
    assert (outcome.close_code, outcome.final_state) == (1006, SessionState.TERMINATED)
    assert not outcome.faulted


async def test_protocol_violation_closes_the_socket_and_the_session(call: CallFactory) -> None:
    c = call()
    await c.client.raw("not json")
    outcome = await c.outcome()
    assert (outcome.close_code, outcome.closed_by) == (1007, CloseInitiator.LOCAL)
    assert outcome.final_state is SessionState.TERMINATED


# ------------------------------------------------------------------------ failures


async def test_session_fault_closes_the_socket_with_internal_error(call: CallFactory) -> None:
    def failing(session: DuplexSession) -> None:
        async def crash() -> None:
            msg = "LLM backend unavailable"
            raise RuntimeError(msg)

        _ = session.spawn(crash(), name="llm")

    c = call(on_user_turn=failing)
    await c.speak(2)
    await c.pause(3)
    outcome = await c.outcome()
    assert (outcome.close_code, outcome.close_reason) == (1011, "session fault")
    assert outcome.faulted
    with pytest.raises(TransportClosedError) as excinfo:
        await drain_until_closed(c.client.socket)
    assert excinfo.value.code == 1011


async def test_runner_task_failure_aborts_closes_and_propagates(
    call: CallFactory, recorder: FlightRecorder
) -> None:
    def broken(_session: DuplexSession) -> None:
        msg = "handler bug"
        raise LookupError(msg)

    c = call(on_user_turn=broken)
    await c.speak(2)
    await c.pause(3)
    with pytest.raises(ExceptionGroup) as excinfo:
        await c.outcome()
    assert excinfo.group_contains(LookupError, match="handler bug")
    assert c.transport.close_code == CloseCode.INTERNAL_ERROR
    assert c.runner.session.faulted
    assert any(e.error_type == "LookupError" for e in recorder.of_type(TaskFailed))


async def test_server_shutdown_cancels_cleanly_with_going_away(call: CallFactory) -> None:
    c = call(on_user_turn=responder(frames=400), capacity=4)
    await c.speak(2)
    await c.pause(3)
    await receive_audio(c.client, 1)
    c.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await c.task
    assert c.transport.close_code == CloseCode.GOING_AWAY
    assert c.runner.session.state is SessionState.TERMINATED
    assert c.runner.session.active_tasks == 0


async def test_run_is_single_use(call: CallFactory) -> None:
    c = call()
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="only be called once"):
        await c.runner.run()
    await c.client.control(Hangup("bye"))
    await c.outcome()
