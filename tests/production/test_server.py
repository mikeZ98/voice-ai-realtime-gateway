"""Gateway server end to end: admission, bounded calls, cleanup and accounting."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from audio_support import pcm_frame
from conftest import FakeClock
from duplex_support import until
from transport_support import Client, ManualSleeper
from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.inference.agent import VoiceAgentPipeline
from voice_ai_realtime_gateway.inference.doubles import ScriptedLLM, ToneASR, ToneTTS
from voice_ai_realtime_gateway.production.admission import TokenAuthenticator, TokenSigner
from voice_ai_realtime_gateway.production.circuit import CircuitPolicy, CircuitRegistry
from voice_ai_realtime_gateway.production.observability import GatewayMetrics, SessionRegistry
from voice_ai_realtime_gateway.production.reaper import ZombieReaper
from voice_ai_realtime_gateway.production.server import (
    Admission,
    CallContext,
    GatewayServer,
    ServerConfig,
    TenantPolicy,
    _LiveCall,
)
from voice_ai_realtime_gateway.telemetry.ids import TraceContext, new_session_id
from voice_ai_realtime_gateway.transport.control import Hangup
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError
from voice_ai_realtime_gateway.transport.framing import DecodedPacket
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.transport.websocket import HeartbeatConfig, WebSocketVoiceTransport
from voice_ai_realtime_gateway.vad.base import VadConfig
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from collections.abc import Callable

    from voice_ai_realtime_gateway.production.server import CallResult

pytestmark = pytest.mark.usefixtures("strict_loop")

KEY = b"s" * 32
NOW = 1_800_000_000.0
DEADLINE = 5.0

type HeaderFactory = Callable[[Harness, TraceContext], dict[str, str]]


class Harness:
    def __init__(
        self,
        *,
        max_sessions: int = 2,
        max_duration_s: float = 600.0,
        fail: bool = False,
        circuits: CircuitRegistry | None = None,
        stubborn: asyncio.Event | None = None,
    ) -> None:
        self.sleeper = ManualSleeper()
        self.sessions = SessionRegistry()
        self.metrics = GatewayMetrics(self.sessions)
        self.closed: list[str] = []
        self.fail = fail
        self.stubborn = stubborn
        self.clock = FakeClock()
        self.server = GatewayServer(
            authenticator=TokenAuthenticator({"k1": KEY}, wall_clock=lambda: NOW),
            tenants={
                "acme": TenantPolicy(
                    max_concurrent_sessions=max_sessions, max_session_duration_s=max_duration_s
                ),
                "opus-only": TenantPolicy(
                    max_concurrent_sessions=1,
                    max_session_duration_s=60,
                    allowed_codecs=frozenset({AudioCodec.OPUS}),
                ),
            },
            metrics=self.metrics,
            sessions=self.sessions,
            factory=self.factory,
            config=ServerConfig(
                media_timeout_s=5.0, heartbeat=HeartbeatConfig(interval_s=3600, timeout_s=3600)
            ),
            sleep=self.sleeper,
            clock=self.clock,
            circuits=circuits,
            required_providers=("deepgram",),
        )

    def factory(self, context: CallContext) -> VoiceAgentPipeline:
        async def closer() -> None:
            self.closed.append(context.admission.grant.tenant)

        context.on_close(closer)
        llm = ScriptedLLM("Sure.") if not self.fail else ScriptedLLM(lambda _h, _u: 1 / 0)  # type: ignore[arg-type,return-value]
        pipeline = VoiceAgentPipeline(
            transport=context.transport,
            vad=EnergyVAD(VadConfig(min_speech_ms=40, hangover_ms=60)),
            asr=ToneASR({1_000: "hello"}),
            llm=llm,
            tts=ToneTTS(),
            sink=context.sink,
        )
        if self.stubborn is not None:
            release = self.stubborn

            async def ignores_cancellation() -> None:  # a hung upstream client
                while not release.is_set():
                    with contextlib.suppress(asyncio.CancelledError):
                        await release.wait()

            pipeline.runner._services = (ignores_cancellation,)
        return pipeline

    def token(self, **claims: Any) -> str:  # noqa: ANN401 -- arbitrary claim values
        base: dict[str, Any] = {
            "aud": "voice-gateway", "exp": NOW + 600, "tenant": "acme",
            "codecs": ["pcm16"], "max_duration_s": 300,
        }  # fmt: skip
        base.update(claims)
        return TokenSigner("k1", KEY).sign(base)

    def connect(self, headers: dict[str, str]) -> tuple[Client, asyncio.Task[CallResult]]:
        server_end, client_end = MemoryWebSocket.pair()
        return Client(client_end), asyncio.create_task(self.server.handle(server_end, headers))


def headers(token: str | None, trace: TraceContext | None = None) -> dict[str, str]:
    result = {} if token is None else {"Authorization": f"Bearer {token}"}
    if trace is not None:
        result.update(trace.to_headers())
    return result


async def speak_and_hear(client: Client, trace: TraceContext) -> None:
    for sequence, amplitude in enumerate([1_000, 1_000, 0, 0, 0]):
        await client.audio(pcm_frame(trace, amplitude=amplitude, sequence=sequence))
    while not isinstance(await asyncio.wait_for(client.receive(), DEADLINE), DecodedPacket):
        pass


async def test_admitted_call_runs_end_to_end_and_is_accounted(trace: TraceContext) -> None:
    harness = Harness()
    client, task = harness.connect(headers(harness.token(), trace))
    await speak_and_hear(client, trace)
    assert harness.server.active_calls("acme") == 1
    assert "voice_gateway_active_sessions_count 1.0" in harness.metrics.render()
    await client.control(Hangup("bye"))
    result = await asyncio.wait_for(task, DEADLINE)
    assert (result.admitted, result.tenant, result.close_code) == (True, "acme", 1000)
    assert result.outcome is not None
    assert result.outcome.turns == 1
    assert harness.closed == ["acme"], "registered closers ran"
    assert harness.server.active_calls("acme") == 0
    text = harness.metrics.render()
    assert 'voice_gateway_sessions_total{outcome="admitted"} 1.0' in text
    assert 'voice_gateway_sessions_total{outcome="completed"} 1.0' in text
    assert "voice_gateway_active_sessions_count 0.0" in text
    assert "voice_gateway_glass_to_glass_seconds_count 1" in text


async def test_token_session_binding_supplies_the_session_id() -> None:
    harness = Harness()
    sid = new_session_id()
    client, task = harness.connect(headers(harness.token(sid=sid)))
    await speak_and_hear(client, TraceContext(sid, TraceContext.new().span_id))
    await client.control(Hangup("bye"))
    assert (await asyncio.wait_for(task, DEADLINE)).admitted


@pytest.mark.parametrize(
    ("make_headers", "code", "reason"),
    [
        (lambda _h, t: headers(None, t), 1008, "missing bearer token"),
        (lambda _h, t: headers("garbage", t), 1008, "malformed token"),
        (lambda h, t: headers(h.token(tenant="ghost"), t), 1008, "unknown tenant"),
        (lambda h, t: headers(h.token(tenant="opus-only"), t), 1008, "no permitted codec"),
        (
            lambda h, t: headers(h.token(sid=new_session_id()), t),
            1008,
            "session id does not match token",
        ),
        (
            lambda h, _t: {**headers(h.token(), None), "x-session-id": "bad"},
            1008,
            "invalid session id",
        ),
    ],
)
async def test_refusals_close_with_policy_violation(
    trace: TraceContext,
    make_headers: HeaderFactory,
    code: int,
    reason: str,
) -> None:
    harness = Harness()
    client, task = harness.connect(make_headers(harness, trace))
    result = await asyncio.wait_for(task, DEADLINE)
    assert (result.admitted, result.close_code, result.close_reason) == (False, code, reason)
    with pytest.raises(TransportClosedError) as excinfo:
        await client.socket.recv()
    assert (excinfo.value.code, excinfo.value.reason) == (code, reason)
    assert 'voice_gateway_sessions_total{outcome="rejected"} 1.0' in harness.metrics.render()


async def test_capacity_and_draining_ask_clients_to_retry_later(trace: TraceContext) -> None:
    harness = Harness(max_sessions=1)
    first, running = harness.connect(headers(harness.token(), trace))
    await until(lambda: harness.server.active_calls("acme") == 1)
    _, refused = harness.connect(headers(harness.token(), TraceContext.new()))
    result = await asyncio.wait_for(refused, DEADLINE)
    assert (result.close_code, result.close_reason) == (
        CloseCode.TRY_AGAIN_LATER,
        "tenant at capacity",
    )
    harness.server.begin_drain()
    assert not harness.server.ready
    _, drained = harness.connect(headers(harness.token(), TraceContext.new()))
    assert (await asyncio.wait_for(drained, DEADLINE)).close_reason == "server draining"
    await first.control(Hangup("bye"))
    await asyncio.wait_for(running, DEADLINE)


async def test_silent_connections_are_closed_as_slow_loris(trace: TraceContext) -> None:
    harness = Harness()
    _client, task = harness.connect(headers(harness.token(), trace))
    await until(lambda: harness.sleeper.sleeping)
    assert harness.sleeper.delays == [5.0]
    harness.sleeper.release()
    result = await asyncio.wait_for(task, DEADLINE)
    assert (result.close_code, result.close_reason) == (1008, "no media received")


async def test_maximum_duration_is_enforced(trace: TraceContext) -> None:
    harness = Harness(max_duration_s=120.0)
    client, task = harness.connect(headers(harness.token(max_duration_s=60), trace))
    await client.audio(pcm_frame(trace))
    await until(lambda: harness.sleeper.sleeping)
    harness.sleeper.release()  # media check passes
    await until(lambda: len(harness.sleeper.delays) == 2)
    assert harness.sleeper.delays == [5.0, 55.0], "min(token 60 s, tenant 120 s) - media window"
    harness.sleeper.release()
    result = await asyncio.wait_for(task, DEADLINE)
    assert (result.close_code, result.close_reason) == (1008, "session duration exceeded")


async def test_codecs_from_the_grant_bound_the_transport(trace: TraceContext) -> None:
    harness = Harness()
    client, task = harness.connect(headers(harness.token(), trace))
    opus = AudioFrame.wrap(
        bytes(40), codec=AudioCodec.OPUS, sample_rate_hz=48_000, channels=1,
        duration_ns=20_000_000, sequence=0, origin_ns=0, trace=trace,
    )  # fmt: skip
    await client.audio(opus)
    result = await asyncio.wait_for(task, DEADLINE)
    assert result.close_code == CloseCode.UNSUPPORTED_DATA


async def test_pipeline_failure_is_accounted_cleaned_up_and_propagated(trace: TraceContext) -> None:
    harness = Harness(fail=True)
    client, task = harness.connect(headers(harness.token(), trace))
    for sequence, amplitude in enumerate([1_000, 1_000, 0, 0, 0]):
        await client.audio(pcm_frame(trace, amplitude=amplitude, sequence=sequence))
    result = await asyncio.wait_for(task, DEADLINE)
    assert result.outcome is not None
    assert result.outcome.faulted
    assert 'voice_gateway_sessions_total{outcome="faulted"} 1.0' in harness.metrics.render()
    assert harness.closed == ["acme"]


async def test_runner_level_failure_still_cleans_up(trace: TraceContext) -> None:
    harness = Harness()

    def broken(context: CallContext) -> VoiceAgentPipeline:
        pipeline = harness.factory(context)

        async def crash() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        pipeline.runner._services = (crash,)  # a supervised service that fails
        return pipeline

    harness.server._factory = broken
    _client, task = harness.connect(headers(harness.token(), trace))
    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(task, DEADLINE)
    assert 'voice_gateway_sessions_total{outcome="faulted"} 1.0' in harness.metrics.render()
    assert harness.server.active_calls("acme") == 0
    assert harness.closed == ["acme"]


def test_policies_are_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        TenantPolicy(max_concurrent_sessions=0, max_session_duration_s=1)
    with pytest.raises(ValueError, match="codec"):
        TenantPolicy(
            max_concurrent_sessions=1, max_session_duration_s=1, allowed_codecs=frozenset()
        )
    with pytest.raises(ValueError, match="media_timeout_s"):
        ServerConfig(media_timeout_s=0)


# ------------------------------------------------------------------------ self-healing


async def test_open_upstream_circuit_refuses_calls_before_building_a_pipeline(
    trace: TraceContext,
) -> None:
    circuits = CircuitRegistry(CircuitPolicy(min_calls=1, failure_ratio=1.0))
    harness = Harness(circuits=circuits)
    circuits._on_change = harness.metrics.record_circuit
    circuits.breaker("deepgram").record_failure()
    built = harness.closed
    _, task = harness.connect(headers(harness.token(), trace))
    result = await asyncio.wait_for(task, DEADLINE)
    assert (result.close_code, result.close_reason) == (1013, "upstream unavailable")
    assert built == [], "no pipeline was built for the refused call"
    text = harness.metrics.render()
    assert 'voice_gateway_circuit_rejections_total{provider="deepgram"} 1.0' in text
    assert 'voice_gateway_circuit_state{provider="deepgram"} 2.0' in text


async def test_close_all_ends_every_live_call(trace: TraceContext) -> None:
    harness = Harness()
    _, task = harness.connect(headers(harness.token(), trace))
    await until(lambda: harness.server.active_calls("acme") == 1)
    assert await harness.server.close_all(CloseCode.GOING_AWAY, "server shutting down") == 1
    result = await asyncio.wait_for(task, DEADLINE)
    assert (result.close_code, result.close_reason) == (1001, "server shutting down")


async def test_reaper_isolates_a_wedged_zombie_without_touching_siblings(
    trace: TraceContext,
) -> None:
    release = asyncio.Event()
    harness = Harness(stubborn=release, max_duration_s=60.0)
    reaper = ZombieReaper(harness.sessions, sink=harness.metrics, grace_s=5, clock=harness.clock)
    _, zombie = harness.connect(headers(harness.token(max_duration_s=60), trace))
    await until(lambda: harness.server.active_calls("acme") == 1)
    harness.clock.now += 30 * 1_000_000_000
    harness.stubborn = None  # the sibling is healthy
    sibling_client, sibling = harness.connect(headers(harness.token(), TraceContext.new()))
    await until(lambda: harness.server.active_calls("acme") == 2)
    assert await reaper.sweep() == 0, "nobody is past deadline + grace yet"
    harness.clock.now += 36 * 1_000_000_000  # zombie: 66 s old > 60 s + 5 s; sibling: 36 s
    assert await reaper.sweep() == 1
    assert reaper.reaped == 1
    assert harness.server.active_calls("acme") == 1, "the zombie's slot is freed even while wedged"
    assert len(harness.sessions) == 1
    assert not zombie.done(), "the zombie is still wedged in its hung service"
    assert not sibling.done(), "the sibling is untouched"
    assert "voice_gateway_sessions_reaped_total 1.0" in harness.metrics.render()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(zombie, DEADLINE)
    await sibling_client.control(Hangup("bye"))
    assert (await asyncio.wait_for(sibling, DEADLINE)).admitted


async def test_reaper_runs_periodically_and_validates(trace: TraceContext) -> None:
    sweeps: list[float] = []

    async def sleep(delay: float) -> None:
        sweeps.append(delay)
        await asyncio.sleep(0)

    reaper = ZombieReaper(
        SessionRegistry(), sink=GatewayMetrics(SessionRegistry()), interval_s=2, sleep=sleep
    )
    task = asyncio.create_task(reaper.run())
    await until(lambda: len(sweeps) >= 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(sweeps) == {2}
    with pytest.raises(ValueError, match="positive"):
        ZombieReaper(SessionRegistry(), sink=GatewayMetrics(SessionRegistry()), grace_s=0)
    del trace


async def test_healthy_circuits_admit_calls(trace: TraceContext) -> None:
    harness = Harness(circuits=CircuitRegistry())
    client, task = harness.connect(headers(harness.token(), trace))
    await until(lambda: harness.server.active_calls("acme") == 1)
    await client.control(Hangup("bye"))
    assert (await asyncio.wait_for(task, DEADLINE)).admitted


async def test_quarantining_a_call_whose_task_already_finished_only_releases_it(
    trace: TraceContext,
) -> None:
    harness = Harness()
    released: list[bool] = []
    server_end, _ = MemoryWebSocket.pair()
    grant = harness.server._authenticator.verify(harness.token())
    admission = Admission(grant=grant, trace=trace, codecs=grant.codecs, max_duration_s=60)
    transport = WebSocketVoiceTransport(server_end, trace=trace, sink=harness.metrics)
    pipeline = harness.factory(CallContext(transport, admission, harness.metrics))

    async def register() -> _LiveCall:
        return _LiveCall(
            transport, pipeline, started_ns=0, deadline_ns=1, release=lambda: released.append(True)
        )

    live = await asyncio.create_task(register())
    assert live.task is not None
    assert live.task.done()
    await live.quarantine()
    assert released == [True]
    assert pipeline.session.faulted
