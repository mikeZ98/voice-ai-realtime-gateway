"""Runner extension points: egress override, ingress tap and supervised services."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from audio_support import LOUD, pcm_frame
from duplex_support import until
from transport_support import Client
from voice_ai_realtime_gateway.domain.states import SessionState
from voice_ai_realtime_gateway.pipeline.runner import GatewaySessionRunner
from voice_ai_realtime_gateway.transport.control import Hangup
from voice_ai_realtime_gateway.transport.errors import CloseCode
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.transport.websocket import HeartbeatConfig, WebSocketVoiceTransport
from voice_ai_realtime_gateway.vad.base import VadConfig, VadDecision, VadEvent
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.pipeline.runner import IngressTap, Service
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")

QUIET = HeartbeatConfig(interval_s=3600.0, timeout_s=3600.0)


class CountingEgress:
    def __init__(self, inner: WebSocketVoiceTransport) -> None:
        self.inner = inner
        self.frames = 0

    async def send(self, frame: AudioFrame, /) -> None:
        self.frames += 1
        await self.inner.send(frame)


def build(
    recorder: FlightRecorder,
    trace: TraceContext,
    *,
    on_ingress: IngressTap | None = None,
    services: tuple[Service, ...] = (),
    wrap: bool = False,
) -> tuple[GatewaySessionRunner, Client, CountingEgress]:
    server, peer = MemoryWebSocket.pair()
    transport = WebSocketVoiceTransport(server, trace=trace, sink=recorder, heartbeat=QUIET)
    egress = CountingEgress(transport)
    runner = GatewaySessionRunner(
        transport=transport,
        vad=EnergyVAD(VadConfig(min_speech_ms=20, hangover_ms=20)),
        sink=recorder,
        egress=egress if wrap else None,
        on_ingress=on_ingress,
        services=services,
    )
    return runner, Client(peer), egress


async def test_ingress_tap_sees_every_frame_with_its_decision(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    seen: list[tuple[int, VadEvent | None, SessionState]] = []
    runner: GatewaySessionRunner

    def tap(frame: AudioFrame, decision: VadDecision) -> None:
        seen.append((frame.sequence, decision.event, runner.session.state))

    runner, client, _ = build(recorder, trace, on_ingress=tap)
    task = asyncio.create_task(runner.run())
    for sequence, amplitude in enumerate([0, LOUD, 0]):
        await client.audio(pcm_frame(trace, amplitude=amplitude, sequence=sequence))
    await until(lambda: len(seen) == 3)
    assert seen == [
        (0, None, SessionState.LISTENING),
        (1, VadEvent.ONSET, SessionState.LISTENING),
        (2, VadEvent.OFFSET, SessionState.PROCESSING_THOUGHT),
    ], "the tap runs after the session applied the decision"
    await client.control(Hangup("bye"))
    await asyncio.wait_for(task, timeout=5)


async def test_egress_override_carries_session_audio(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    runner, client, egress = build(recorder, trace, wrap=True)
    task = asyncio.create_task(runner.run())
    await client.audio(pcm_frame(trace, amplitude=LOUD, sequence=0))
    await client.audio(pcm_frame(trace, amplitude=0, sequence=1))
    await until(lambda: runner.session.state is SessionState.PROCESSING_THOUGHT)
    span = runner.session.begin_response()
    await runner.session.push_audio(pcm_frame(span, amplitude=LOUD))
    await until(lambda: egress.frames == 1)
    await client.control(Hangup("bye"))
    await asyncio.wait_for(task, timeout=5)


async def test_services_run_alongside_and_are_cancelled_after_termination(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    lifecycle: list[str] = []

    async def service() -> None:
        lifecycle.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            lifecycle.append("cancelled")
            raise

    async def short_lived() -> None:
        lifecycle.append("done")

    runner, client, _ = build(recorder, trace, services=(service, short_lived))
    task = asyncio.create_task(runner.run())
    await until(lambda: "started" in lifecycle)
    await client.control(Hangup("bye"))
    outcome = await asyncio.wait_for(task, timeout=5)
    assert sorted(lifecycle) == ["cancelled", "done", "started"]
    assert outcome.close_code == CloseCode.NORMAL
    assert not outcome.faulted


async def test_failing_service_faults_the_call(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def broken() -> None:
        await asyncio.sleep(0)
        msg = "ASR backend lost"
        raise ConnectionError(msg)

    runner, client, _ = build(recorder, trace, services=(broken,))
    with pytest.raises(ExceptionGroup) as excinfo:
        await asyncio.wait_for(runner.run(), timeout=5)
    assert excinfo.group_contains(ConnectionError)
    assert runner.session.faulted
    assert runner.transport.close_code == CloseCode.INTERNAL_ERROR
    del client
