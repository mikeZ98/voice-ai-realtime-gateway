"""WebSocket voice transport: multiplexing, bounded ingress, heartbeat, disconnects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from audio_support import pcm_frame
from duplex_support import until
from transport_support import Client, ManualSleeper, ScriptedConnection
from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.core.ring import AudioRingBuffer, OverflowPolicy
from voice_ai_realtime_gateway.telemetry.events import BufferOverflow, IngressGap, TransportClosed
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.transport.base import CloseInitiator
from voice_ai_realtime_gateway.transport.control import (
    AudioFormat,
    CancelResponse,
    Hangup,
    Ping,
    PlaybackFlush,
    Pong,
    SessionConfig,
    VadConfigUpdate,
)
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError
from voice_ai_realtime_gateway.transport.framing import HEADER_SIZE, DecodedPacket, encode_packet
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.transport.websocket import (
    HeartbeatConfig,
    TransportLimits,
    WebSocketVoiceTransport,
)
from voice_ai_realtime_gateway.vad.base import VadConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from conftest import FakeClock
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")

QUIET = HeartbeatConfig(interval_s=3600.0, timeout_s=3600.0)


class Harness:
    def __init__(
        self,
        transport: WebSocketVoiceTransport,
        client: Client,
        runner: asyncio.Task[None],
    ) -> None:
        self.transport = transport
        self.client = client
        self.runner = runner


@pytest.fixture
async def harness_factory(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> AsyncIterator[type[Harness]]:
    started: list[Harness] = []

    class Factory(Harness):
        @classmethod
        def start(cls, **kwargs: object) -> Harness:
            capacity = kwargs.pop("capacity", 64)
            server, peer = MemoryWebSocket.pair(capacity=capacity)  # type: ignore[arg-type]
            options: dict[str, object] = {"heartbeat": QUIET, **kwargs}
            transport = WebSocketVoiceTransport(
                server,
                trace=trace,
                sink=recorder,
                clock=clock,
                **options,  # type: ignore[arg-type]
            )
            harness = Harness(transport, Client(peer), asyncio.create_task(transport.run()))
            started.append(harness)
            return harness

    yield Factory
    for harness in started:
        await harness.transport.close()
        await harness.runner


Factory = type[Harness]


# ------------------------------------------------------------------------ configuration


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"interval_s": 0.0}, "interval_s"),
        ({"interval_s": 10.0, "timeout_s": 5.0}, "interval_s <= timeout_s"),
        ({"timeout_s": float("inf")}, "finite"),
    ],
)
async def test_heartbeat_config_is_validated(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HeartbeatConfig(**kwargs)


async def test_limits_and_codecs_are_validated(recorder: FlightRecorder) -> None:
    with pytest.raises(ValueError, match="max_pending_control"):
        TransportLimits(max_pending_control=0)
    with pytest.raises(ValueError, match="accepted_codecs"):
        WebSocketVoiceTransport(
            ScriptedConnection(),
            trace=TraceContext.new(),
            sink=recorder,
            accepted_codecs=frozenset(),
        )
    assert HeartbeatConfig(interval_s=1.0, timeout_s=2.5).timeout_ns == 2_500_000_000


async def test_defaults_bound_ingress_with_drop_oldest(recorder: FlightRecorder) -> None:
    transport = WebSocketVoiceTransport(
        ScriptedConnection(), trace=TraceContext.new(), sink=recorder
    )
    ring = transport.ingress_ring
    assert (ring.capacity_frames, ring.policy) == (64, OverflowPolicy.DROP_OLDEST)
    assert transport.audio_format is None
    assert not transport.closed
    assert (transport.close_code, transport.close_reason, transport.closed_by) == (None, "", None)
    assert repr(transport).startswith("WebSocketVoiceTransport(session=")


async def test_run_is_single_use(harness_factory: Factory) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="only be called once"):
        await harness.transport.run()


# ------------------------------------------------------------------------ multiplexing


async def test_binary_audio_is_framed_into_ingress_with_receiver_timestamps(
    harness_factory: Factory, clock: FakeClock, trace: TraceContext
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    clock.now = 9_999
    sent = [pcm_frame(trace, amplitude=100 * i, sequence=i, origin_ns=1) for i in range(3)]
    for frame in sent:
        await harness.client.audio(frame)
    received = [await anext(harness.transport.ingress) for _ in range(3)]
    assert [f.sequence for f in received] == [0, 1, 2]
    assert [bytes(f.payload) for f in received] == [bytes(f.payload) for f in sent]
    assert all(f.origin_ns == 9_999 and f.trace == trace for f in received)
    stats = harness.transport.stats()
    assert (stats.frames_in, stats.bytes_in) == (3, 3 * (HEADER_SIZE + 640))


async def test_text_control_is_typed_and_ping_is_answered(
    harness_factory: Factory,
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    vad = VadConfigUpdate(VadConfig(min_speech_ms=40))
    for message in (Ping(41), Pong(5), vad, CancelResponse(), Hangup("bye")):
        await harness.client.control(message)
    assert await harness.client.receive() == Pong(41)
    control = harness.transport.control
    assert [await anext(control) for _ in range(3)] == [vad, CancelResponse(), Hangup("bye")]
    stats = harness.transport.stats()
    assert (stats.control_in, stats.control_out) == (5, 1)


async def test_session_config_negotiates_the_ingress_format(
    harness_factory: Factory, trace: TraceContext
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    audio = AudioFormat(AudioCodec.PCM16, 16_000, 1)
    await harness.client.control(SessionConfig(audio))
    await harness.client.audio(pcm_frame(trace))
    await anext(harness.transport.ingress)
    assert harness.transport.audio_format == audio
    assert len(harness.transport.control) == 0, "session.config is consumed by the transport"


async def test_egress_frames_and_server_control_reach_the_client(
    harness_factory: Factory, trace: TraceContext
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    frame = pcm_frame(trace.child(), amplitude=1_000, sequence=3, origin_ns=77)
    await harness.transport.send(frame)
    assert await harness.transport.control.send(PlaybackFlush())
    packet = await harness.client.receive()
    assert isinstance(packet, DecodedPacket)
    assert (packet.frame.sequence, packet.sender_timestamp_ns) == (3, 77)
    assert bytes(packet.frame.payload) == bytes(frame.payload)
    assert await harness.client.receive() == PlaybackFlush()
    stats = harness.transport.stats()
    assert (stats.frames_out, stats.bytes_out) == (1, HEADER_SIZE + frame.nbytes)


# ------------------------------------------------------------------------ disconnects


async def test_remote_close_ends_ingress_control_and_run(
    harness_factory: Factory, recorder: FlightRecorder
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    await harness.client.socket.close(CloseCode.GOING_AWAY, "tab closed")
    await harness.runner
    transport = harness.transport
    assert (transport.close_code, transport.close_reason, transport.closed_by) == (
        1001,
        "tab closed",
        CloseInitiator.REMOTE,
    )
    assert [f async for f in transport.ingress] == []
    assert [m async for m in transport.control] == []
    assert [m async for m in transport.control] == [], "an ended stream stays ended"
    (event,) = recorder.of_type(TransportClosed)
    assert (event.code, event.initiator) == (1001, "remote")


async def test_network_loss_is_an_abnormal_remote_close(harness_factory: Factory) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    await harness.client.socket.sever()
    await harness.runner
    assert harness.transport.close_code == CloseCode.ABNORMAL
    assert harness.transport.closed_by is CloseInitiator.REMOTE


async def test_local_close_is_idempotent_and_truncates_the_reason(
    harness_factory: Factory,
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    await harness.transport.close(CloseCode.GOING_AWAY, "x" * 300)
    await harness.transport.close(CloseCode.INTERNAL_ERROR, "second close loses")
    await harness.runner
    assert harness.transport.close_code == 1001
    assert len(harness.transport.close_reason) == 123
    assert harness.transport.closed_by is CloseInitiator.LOCAL
    with pytest.raises(TransportClosedError) as excinfo:
        await harness.client.socket.recv()
    assert excinfo.value.code == 1001


async def test_egress_after_disconnect_is_dropped_not_raised(
    harness_factory: Factory, trace: TraceContext
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    await harness.client.socket.close()
    await harness.transport.send(pcm_frame(trace))  # peer gone, reader not yet aware
    assert not await harness.transport.send_control(PlaybackFlush())
    await harness.runner
    await harness.transport.send(pcm_frame(trace))  # transport closed
    assert not await harness.transport.send_control(PlaybackFlush())
    assert harness.transport.stats().egress_dropped == 2


# ------------------------------------------------------------------------ violations


@pytest.mark.parametrize(
    ("message", "code"),
    [
        (b"\x00" * 10, CloseCode.INVALID_PAYLOAD),
        (b"\x00" * (HEADER_SIZE + 1025), CloseCode.MESSAGE_TOO_BIG),
        ("{", CloseCode.INVALID_PAYLOAD),
        ('{"type":"playback.flush"}', CloseCode.UNSUPPORTED_DATA),
        ("x" * 4097, CloseCode.MESSAGE_TOO_BIG),
    ],
)
async def test_wire_violations_close_with_the_matching_code(
    harness_factory: Factory, message: str | bytes, code: CloseCode
) -> None:
    harness = harness_factory.start(  # type: ignore[attr-defined]
        limits=TransportLimits(max_payload_bytes=1024)
    )
    await harness.client.raw(message)
    await harness.runner
    assert harness.transport.close_code == code
    assert harness.transport.closed_by is CloseInitiator.LOCAL


async def test_foreign_session_audio_is_a_policy_violation(harness_factory: Factory) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    await harness.client.audio(pcm_frame(TraceContext.new()))
    await harness.runner
    assert harness.transport.close_code == CloseCode.POLICY_VIOLATION
    assert harness.transport.stats().frames_in == 0


async def test_unaccepted_codec_is_unsupported_data(
    harness_factory: Factory, trace: TraceContext
) -> None:
    harness = harness_factory.start(  # type: ignore[attr-defined]
        accepted_codecs=frozenset({AudioCodec.OPUS})
    )
    await harness.client.audio(pcm_frame(trace))
    await harness.runner
    assert harness.transport.close_code == CloseCode.UNSUPPORTED_DATA


async def test_session_config_with_unaccepted_codec_is_unsupported_data(
    harness_factory: Factory,
) -> None:
    harness = harness_factory.start(  # type: ignore[attr-defined]
        accepted_codecs=frozenset({AudioCodec.PCM16})
    )
    await harness.client.control(SessionConfig(AudioFormat(AudioCodec.OPUS, 48_000, 1)))
    await harness.runner
    assert harness.transport.close_code == CloseCode.UNSUPPORTED_DATA


@pytest.mark.parametrize(
    "negotiated",
    [
        AudioFormat(AudioCodec.PCM16, 48_000, 1),
        AudioFormat(AudioCodec.PCM16, 16_000, 2),
        AudioFormat(AudioCodec.OPUS, 16_000, 1),
    ],
)
async def test_audio_not_matching_session_config_is_unsupported_data(
    harness_factory: Factory, trace: TraceContext, negotiated: AudioFormat
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    await harness.client.control(SessionConfig(negotiated))
    await harness.client.audio(pcm_frame(trace))  # PCM16 / 16 kHz / mono
    await harness.runner
    assert harness.transport.close_code == CloseCode.UNSUPPORTED_DATA


async def test_control_flood_from_a_client_is_a_policy_violation(harness_factory: Factory) -> None:
    harness = harness_factory.start(  # type: ignore[attr-defined]
        limits=TransportLimits(max_pending_control=4)
    )
    for _ in range(5):
        await harness.client.control(CancelResponse())
    await harness.runner
    assert harness.transport.close_code == CloseCode.POLICY_VIOLATION


# ------------------------------------------------------------------------ sequence hygiene


async def test_gaps_are_counted_and_late_packets_dropped(
    harness_factory: Factory, recorder: FlightRecorder, trace: TraceContext
) -> None:
    harness = harness_factory.start()  # type: ignore[attr-defined]
    for sequence in (0, 1, 4, 3, 4, 5):  # 2-3 lost, then a late 3 and a duplicate 4
        await harness.client.audio(pcm_frame(trace, sequence=sequence))
    next_span = trace.child()
    await harness.client.audio(pcm_frame(next_span, sequence=100))  # new span: no gap
    delivered = [await anext(harness.transport.ingress) for _ in range(5)]
    assert [f.sequence for f in delivered] == [0, 1, 4, 5, 100]
    stats = harness.transport.stats()
    assert (stats.lost_packets, stats.late_dropped) == (2, 2)
    (gap,) = recorder.of_type(IngressGap)
    assert (gap.expected_sequence, gap.received_sequence, gap.lost) == (2, 4, 2)


# ------------------------------------------------------------------------ backpressure


async def test_slow_consumer_with_drop_oldest_keeps_memory_bounded(
    harness_factory: Factory, trace: TraceContext
) -> None:
    ring = AudioRingBuffer(
        capacity_frames=8, capacity_bytes=64 * 1024, policy=OverflowPolicy.DROP_OLDEST
    )
    harness = harness_factory.start(ingress_ring=ring)  # type: ignore[attr-defined]
    for sequence in range(200):  # nobody consumes ingress
        await harness.client.audio(pcm_frame(trace, sequence=sequence))
    await until(lambda: harness.transport.stats().frames_in == 200)
    assert len(ring) == 8
    assert ring.stats().high_watermark_frames == 8
    assert ring.stats().evicted == 192
    newest = [await anext(harness.transport.ingress) for _ in range(8)]
    assert [f.sequence for f in newest] == list(range(192, 200)), "newest audio survives"


async def test_drop_oldest_overflow_is_observable(
    harness_factory: Factory, recorder: FlightRecorder, trace: TraceContext
) -> None:
    ring = AudioRingBuffer(
        capacity_frames=2,
        capacity_bytes=64 * 1024,
        policy=OverflowPolicy.DROP_OLDEST,
        sink=recorder,
    )
    harness = harness_factory.start(ingress_ring=ring)  # type: ignore[attr-defined]
    for sequence in range(5):
        await harness.client.audio(pcm_frame(trace, sequence=sequence))
    await until(lambda: harness.transport.stats().frames_in == 5)
    assert sum(e.evicted_frames for e in recorder.of_type(BufferOverflow)) == 3


async def test_backpressure_policy_pushes_back_on_the_client_socket(
    harness_factory: Factory, trace: TraceContext
) -> None:
    ring = AudioRingBuffer(
        capacity_frames=4, capacity_bytes=64 * 1024, policy=OverflowPolicy.BACKPRESSURE
    )
    harness = harness_factory.start(ingress_ring=ring, capacity=2)  # type: ignore[attr-defined]

    async def flood() -> None:
        for sequence in range(20):
            await harness.client.audio(pcm_frame(trace, sequence=sequence))

    flooder = asyncio.create_task(flood())
    await until(lambda: harness.transport._reader_blocked)
    for _ in range(20):
        await asyncio.sleep(0)
    assert not flooder.done(), "client is throttled by the full socket buffer"
    assert len(ring) == 4
    received = [(await anext(harness.transport.ingress)).sequence for _ in range(20)]
    await flooder
    assert received == list(range(20)), "backpressure loses nothing"
    assert ring.stats().evicted == 0


async def test_close_releases_a_reader_blocked_on_backpressure(
    harness_factory: Factory, trace: TraceContext
) -> None:
    ring = AudioRingBuffer(
        capacity_frames=1, capacity_bytes=64 * 1024, policy=OverflowPolicy.BACKPRESSURE
    )
    harness = harness_factory.start(ingress_ring=ring)  # type: ignore[attr-defined]
    await harness.client.audio(pcm_frame(trace, sequence=0))
    await harness.client.audio(pcm_frame(trace, sequence=1))
    await until(lambda: harness.transport._reader_blocked)
    await harness.transport.close()
    await harness.runner
    assert not harness.transport._reader_blocked


# ------------------------------------------------------------------------ heartbeat


def scripted(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext, **kwargs: object
) -> tuple[WebSocketVoiceTransport, ScriptedConnection, ManualSleeper]:
    connection = ScriptedConnection()
    sleeper = ManualSleeper()
    transport = WebSocketVoiceTransport(
        connection,
        trace=trace,
        sink=recorder,
        clock=clock,
        heartbeat=HeartbeatConfig(interval_s=1.0, timeout_s=3.0),
        sleep=sleeper,
        **kwargs,  # type: ignore[arg-type]
    )
    return transport, connection, sleeper


async def test_heartbeat_pings_every_interval(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    transport, connection, sleeper = scripted(recorder, clock, trace)
    runner = asyncio.create_task(transport.run())
    for expected in (1, 2):
        await until(lambda: sleeper.sleeping)
        sleeper.release()
        await until(lambda n=expected: len(connection.sent) == n)  # type: ignore[misc]
    assert connection.sent == ['{"type":"ping","nonce":1}', '{"type":"ping","nonce":2}']
    assert sleeper.delays[:2] == [1.0, 1.0]
    connection.inbound.put_nowait(TransportClosedError(1000))
    await runner


async def test_silent_peer_is_declared_dead_and_closed_going_away(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    transport, connection, sleeper = scripted(recorder, clock, trace)
    runner = asyncio.create_task(transport.run())
    await until(lambda: sleeper.sleeping)
    clock.now += 3_000_000_001
    sleeper.release()
    await until(lambda: transport.closed)
    assert connection.closes == [(1001, "heartbeat timeout")]
    connection.inbound.put_nowait(TransportClosedError(1001, "heartbeat timeout"))
    await runner
    assert transport.closed_by is CloseInitiator.LOCAL


async def test_any_inbound_message_counts_as_liveness(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    transport, connection, sleeper = scripted(recorder, clock, trace)
    runner = asyncio.create_task(transport.run())
    await until(lambda: sleeper.sleeping)
    clock.now += 2_900_000_000
    connection.inbound.put_nowait('{"type":"pong","nonce":0}')
    await until(lambda: transport.stats().control_in == 1)
    clock.now += 2_900_000_000
    sleeper.release()
    await until(lambda: len(connection.sent) == 1)
    assert not transport.closed
    connection.inbound.put_nowait(TransportClosedError(1000))
    await runner


async def test_dead_peer_detection_is_suspended_while_we_apply_backpressure(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    ring = AudioRingBuffer(
        capacity_frames=1, capacity_bytes=64 * 1024, policy=OverflowPolicy.BACKPRESSURE
    )
    transport, connection, sleeper = scripted(recorder, clock, trace, ingress_ring=ring)
    runner = asyncio.create_task(transport.run())
    for sequence in range(2):
        connection.inbound.put_nowait(encode_packet(pcm_frame(trace, sequence=sequence)))
    await until(lambda: transport._reader_blocked)
    await until(lambda: sleeper.sleeping)
    clock.now += 10_000_000_000
    sleeper.release()
    await until(lambda: len(connection.sent) == 1)
    assert not transport.closed, "our own stall is not the peer's death"
    await transport.close()
    connection.inbound.put_nowait(TransportClosedError(1000))  # scripted recv: deliver the close
    await asyncio.wait_for(runner, timeout=5)


async def test_heartbeat_stops_when_the_transport_closed_during_its_sleep(
    recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    transport, connection, sleeper = scripted(recorder, clock, trace)
    runner = asyncio.create_task(transport.run())
    await until(lambda: sleeper.sleeping)
    await transport.close()  # scripted recv stays blocked: the reader has not noticed yet
    sleeper.release()
    for _ in range(5):
        await asyncio.sleep(0)
    assert connection.sent == [], "no ping after close"
    connection.inbound.put_nowait(TransportClosedError(1000))
    await runner


async def test_control_end_is_idempotent_and_trace_is_exposed(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    transport = WebSocketVoiceTransport(ScriptedConnection(), trace=trace, sink=recorder)
    assert transport.trace == trace
    transport.control.end()
    transport.control.end()
    assert len(transport.control) == 1, "exactly one end-of-stream marker"
    assert [m async for m in transport.control] == []
