"""Gateway server end to end: admission, bounded calls, cleanup and accounting."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from audio_support import pcm_frame
from duplex_support import until
from transport_support import Client, ManualSleeper
from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.inference.agent import VoiceAgentPipeline
from voice_ai_realtime_gateway.inference.doubles import ScriptedLLM, ToneASR, ToneTTS
from voice_ai_realtime_gateway.production.admission import TokenAuthenticator, TokenSigner
from voice_ai_realtime_gateway.production.observability import GatewayMetrics, SessionRegistry
from voice_ai_realtime_gateway.production.server import (
    GatewayServer,
    ServerConfig,
    TenantPolicy,
)
from voice_ai_realtime_gateway.telemetry.ids import TraceContext, new_session_id
from voice_ai_realtime_gateway.transport.control import Hangup
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError
from voice_ai_realtime_gateway.transport.framing import DecodedPacket
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.transport.websocket import HeartbeatConfig
from voice_ai_realtime_gateway.vad.base import VadConfig
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from collections.abc import Callable

    from voice_ai_realtime_gateway.production.server import CallContext, CallResult

pytestmark = pytest.mark.usefixtures("strict_loop")

KEY = b"s" * 32
NOW = 1_800_000_000.0
DEADLINE = 5.0

type HeaderFactory = Callable[[Harness, TraceContext], dict[str, str]]


class Harness:
    def __init__(
        self, *, max_sessions: int = 2, max_duration_s: float = 600.0, fail: bool = False
    ) -> None:
        self.sleeper = ManualSleeper()
        self.sessions = SessionRegistry()
        self.metrics = GatewayMetrics(self.sessions)
        self.closed: list[str] = []
        self.fail = fail
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
        )

    def factory(self, context: CallContext) -> VoiceAgentPipeline:
        async def closer() -> None:
            self.closed.append(context.admission.grant.tenant)

        context.on_close(closer)
        llm = ScriptedLLM("Sure.") if not self.fail else ScriptedLLM(lambda _h, _u: 1 / 0)  # type: ignore[arg-type,return-value]
        return VoiceAgentPipeline(
            transport=context.transport,
            vad=EnergyVAD(VadConfig(min_speech_ms=40, hangover_ms=60)),
            asr=ToneASR({1_000: "hello"}),
            llm=llm,
            tts=ToneTTS(),
            sink=context.sink,
        )

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
