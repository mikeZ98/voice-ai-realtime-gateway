"""The gateway's front door: admit, run, bound and account for every call.

``GatewayServer.handle(connection, headers)`` is what a WebSocket route calls after the
upgrade. ``admit(headers)`` can also run *before* the upgrade, to reject with an HTTP
status instead.

Admission, in order (any failure closes the socket with the error's code):

1. drain mode -> ``1013`` (try again later);
2. bearer token -> :class:`~.admission.TokenAuthenticator` -> ``1008`` on failure;
3. tenant policy lookup -> ``1008`` for an unknown tenant;
4. per-tenant concurrency -> ``1013`` at capacity;
5. ``X-Session-ID`` must match the token's ``sid`` when the token binds one -> ``1008``;
6. allowed codecs = token codecs ∩ tenant codecs (empty -> ``1008``), enforced by the
   transport's ``accepted_codecs``;
7. maximum duration = min(token, tenant).

While a call runs, a watchdog enforces two bounds with ``1008``: a client that sends no
audio within ``media_timeout_s`` (slow-loris, or a connection squatting a slot) and the
admitted maximum duration. Control-message rate limits and frame size limits are
enforced by the transport itself (``TransportLimits``).

Cleanup is deterministic: registered closers (for example the provider adapters'
persistent connections) run when the call ends, whatever the outcome.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import Counter as Tally
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, NoReturn, final

from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.production.admission import AdmissionError, bearer_token
from voice_ai_realtime_gateway.telemetry.ids import (
    SESSION_ID_HEADER,
    InvalidTraceIdError,
    TraceContext,
    new_span_id,
    parse_session_id,
)
from voice_ai_realtime_gateway.transport.errors import CloseCode
from voice_ai_realtime_gateway.transport.websocket import (
    HeartbeatConfig,
    TransportLimits,
    WebSocketVoiceTransport,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.inference.agent import VoiceAgentPipeline
    from voice_ai_realtime_gateway.pipeline.runner import RunOutcome
    from voice_ai_realtime_gateway.production.admission import AdmissionGrant, TokenAuthenticator
    from voice_ai_realtime_gateway.production.observability import (
        GatewayMetrics,
        SessionRegistry,
    )
    from voice_ai_realtime_gateway.telemetry.ids import SessionId
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import WebSocketConnection

__all__ = [
    "Admission",
    "CallContext",
    "CallResult",
    "GatewayServer",
    "PipelineFactory",
    "ServerConfig",
    "TenantPolicy",
]

type Closer = Callable[[], Awaitable[None]]
type Sleeper = Callable[[float], Awaitable[None]]
type PipelineFactory = Callable[[CallContext], VoiceAgentPipeline]
"""Builds the call's pipeline (adapters, VAD, ...) for an admitted connection."""

_ALL_CODECS: Final = frozenset(AudioCodec)


def _refuse(reason: str, *, code: int = CloseCode.POLICY_VIOLATION) -> NoReturn:
    raise AdmissionError(reason, code=code)


def _session_id(raw: str) -> SessionId:
    try:
        return parse_session_id(raw)
    except InvalidTraceIdError:
        pass
    _refuse("invalid session id")


@dataclass(frozen=True, slots=True, kw_only=True)
class TenantPolicy:
    """Per-tenant bounds, intersected with each token's grant."""

    max_concurrent_sessions: int
    max_session_duration_s: float
    allowed_codecs: frozenset[AudioCodec] = _ALL_CODECS

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: on a non-positive bound or an empty codec set.
        """
        if self.max_concurrent_sessions < 1 or self.max_session_duration_s <= 0:
            msg = "tenant bounds must be positive"
            raise ValueError(msg)
        if not self.allowed_codecs:
            msg = "a tenant must allow at least one codec"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerConfig:
    """Per-connection hardening knobs."""

    media_timeout_s: float = 10.0
    limits: TransportLimits = field(default_factory=TransportLimits)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: if ``media_timeout_s`` is not positive.
        """
        if self.media_timeout_s <= 0:
            msg = "media_timeout_s must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class Admission:
    """An admitted connection's entitlements."""

    grant: AdmissionGrant
    trace: TraceContext
    codecs: frozenset[AudioCodec]
    max_duration_s: float


@final
class CallContext:
    """What a :data:`PipelineFactory` receives; ``on_close`` registers cleanup."""

    __slots__ = ("_closers", "admission", "sink", "transport")

    def __init__(
        self, transport: WebSocketVoiceTransport, admission: Admission, sink: TelemetrySink
    ) -> None:
        """Bundle the call's transport, entitlements and telemetry sink."""
        self.transport = transport
        self.admission = admission
        self.sink = sink
        self._closers: list[Closer] = []

    def on_close(self, closer: Closer) -> None:
        """Run ``closer`` when the call ends (in registration order)."""
        self._closers.append(closer)

    async def close(self) -> None:
        """Run every registered closer."""
        for closer in self._closers:
            await closer()


@dataclass(frozen=True, slots=True, kw_only=True)
class CallResult:
    """How a connection ended."""

    admitted: bool
    close_code: int | None
    close_reason: str
    tenant: str | None
    outcome: RunOutcome | None


@final
class _LiveCall:
    __slots__ = ("pipeline", "transport")

    def __init__(self, transport: WebSocketVoiceTransport, pipeline: VoiceAgentPipeline) -> None:
        self.transport = transport
        self.pipeline = pipeline

    def ring_bytes(self) -> int:
        current = self.pipeline.capture.current
        utterance = current.ring.nbytes if current is not None else 0
        return self.transport.ingress_ring.nbytes + self.pipeline.session.ring.nbytes + utterance


@final
class GatewayServer:
    """Admits and supervises calls."""

    __slots__ = (
        "_authenticator",
        "_clock",
        "_config",
        "_draining",
        "_factory",
        "_metrics",
        "_per_tenant",
        "_sessions",
        "_sink",
        "_sleep",
        "_tenants",
    )

    def __init__(  # noqa: PLR0913 -- the composition root of the service
        self,
        *,
        authenticator: TokenAuthenticator,
        tenants: Mapping[str, TenantPolicy],
        metrics: GatewayMetrics,
        sessions: SessionRegistry,
        factory: PipelineFactory,
        sink: TelemetrySink | None = None,
        config: ServerConfig | None = None,
        clock: MonotonicClock = time.monotonic_ns,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        """Serve calls built by ``factory``; ``sink`` defaults to ``metrics``."""
        self._authenticator = authenticator
        self._tenants = dict(tenants)
        self._metrics = metrics
        self._sessions = sessions
        self._factory = factory
        self._sink: TelemetrySink = metrics if sink is None else sink
        self._config = ServerConfig() if config is None else config
        self._clock = clock
        self._sleep = sleep
        self._draining = False
        self._per_tenant: Tally[str] = Tally()

    @property
    def ready(self) -> bool:
        """Whether new calls are accepted (``/readyz``)."""
        return not self._draining

    def begin_drain(self) -> None:
        """Refuse new calls (``1013``) while live calls finish."""
        self._draining = True

    def active_calls(self, tenant: str) -> int:
        """Live calls of ``tenant``."""
        return self._per_tenant[tenant]

    def admit(self, headers: Mapping[str, str]) -> Admission:
        """Decide whether a connection with ``headers`` may start a call.

        Raises:
            AdmissionError: with the close code and public reason for the refusal.
        """
        if self._draining:
            _refuse("server draining", code=CloseCode.TRY_AGAIN_LATER)
        grant = self._authenticator.verify(bearer_token(headers))
        policy = self._tenants.get(grant.tenant)
        if policy is None:
            _refuse("unknown tenant")
        if self._per_tenant[grant.tenant] >= policy.max_concurrent_sessions:
            _refuse("tenant at capacity", code=CloseCode.TRY_AGAIN_LATER)
        codecs = grant.codecs & policy.allowed_codecs
        if not codecs:
            _refuse("no permitted codec")
        return Admission(
            grant=grant,
            trace=self._trace(headers, grant),
            codecs=codecs,
            max_duration_s=min(grant.max_duration_s, policy.max_session_duration_s),
        )

    @staticmethod
    def _trace(headers: Mapping[str, str], grant: AdmissionGrant) -> TraceContext:
        raw = next((v for k, v in headers.items() if k.lower() == SESSION_ID_HEADER.lower()), None)
        session = None if raw is None else _session_id(raw)
        if grant.session_id is not None and session not in {None, grant.session_id}:
            _refuse("session id does not match token")
        if session is None:
            return (
                TraceContext.new()
                if grant.session_id is None
                else TraceContext(grant.session_id, new_span_id())
            )
        return TraceContext(session, new_span_id())

    async def handle(
        self, connection: WebSocketConnection, headers: Mapping[str, str]
    ) -> CallResult:
        """Admit and run one call to completion.

        Raises:
            ExceptionGroup: if the call's pipeline failed internally (after cleanup and
                accounting), so the hosting server logs it.
        """
        try:
            admission = self.admit(headers)
        except AdmissionError as refusal:
            self._metrics.record_session("rejected")
            await connection.close(refusal.code, refusal.reason)
            return CallResult(
                admitted=False,
                close_code=refusal.code,
                close_reason=refusal.reason,
                tenant=None,
                outcome=None,
            )
        tenant = admission.grant.tenant
        self._metrics.record_session("admitted")
        transport = WebSocketVoiceTransport(
            connection,
            trace=admission.trace,
            sink=self._sink,
            clock=self._clock,
            accepted_codecs=admission.codecs,
            heartbeat=self._config.heartbeat,
            limits=self._config.limits,
        )
        context = CallContext(transport, admission, self._sink)
        pipeline = self._factory(context)
        live = _LiveCall(transport, pipeline)
        self._sessions.add(live)
        self._per_tenant[tenant] += 1
        watchdog = asyncio.create_task(self._watchdog(transport, admission.max_duration_s))
        outcome: RunOutcome | None = None
        try:
            outcome = await pipeline.run()
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
            self._sessions.discard(live)
            self._per_tenant[tenant] -= 1
            await context.close()
            self._metrics.record_session(
                "faulted" if outcome is None or outcome.faulted else "completed"
            )
        return CallResult(
            admitted=True,
            close_code=transport.close_code,
            close_reason=transport.close_reason,
            tenant=tenant,
            outcome=outcome,
        )

    async def _watchdog(self, transport: WebSocketVoiceTransport, max_duration_s: float) -> None:
        first = min(self._config.media_timeout_s, max_duration_s)
        await self._sleep(first)
        if transport.stats().frames_in == 0:
            await transport.close(CloseCode.POLICY_VIOLATION, "no media received")
            return
        await self._sleep(max(0.0, max_duration_s - first))
        await transport.close(CloseCode.POLICY_VIOLATION, "session duration exceeded")
