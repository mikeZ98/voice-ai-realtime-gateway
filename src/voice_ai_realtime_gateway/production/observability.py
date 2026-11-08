"""Gateway-wide Prometheus metrics fed by the telemetry event stream.

:class:`GatewayMetrics` is a ``TelemetrySink``: pass it (or a ``FanoutSink`` containing
it) to every call and it maintains:

=====================================================  ==========  ======================
metric                                                 type        source
=====================================================  ==========  ======================
``voice_gateway_active_sessions_count``                gauge       :class:`SessionRegistry`
``voice_gateway_active_audio_rings_bytes``             gauge       :class:`SessionRegistry`
``voice_gateway_glass_to_glass_seconds``               histogram   ``ResponseTimeline``
``voice_gateway_first_token_latency_seconds``          summary     ``FirstTokenLatency``
``voice_gateway_tts_barge_in_halt_microseconds``       histogram   ``TtsHalted``
``voice_gateway_barge_in_total``                       counter     ``BargeInHandled``
``voice_gateway_packet_drops_ingress_total{reason}``   counter     ``BufferOverflow``,
                                                                   ``IngressGap``
``voice_gateway_packet_drops_egress_total``            counter     ``EgressDropped``
``voice_gateway_upstream_provider_errors_total``       counter     ``ProviderError``
``{provider,kind,fatal}``
``voice_gateway_transport_closed_total{code}``         counter     ``TransportClosed``
``voice_gateway_sessions_total{outcome}``              counter     :meth:`record_session`
``voice_gateway_circuit_state{provider}``              gauge       :meth:`record_circuit`
``voice_gateway_circuit_rejections_total{provider}``   counter     :meth:`record_circuit`
``voice_gateway_sessions_reaped_total``                counter     ``SessionReaped``
=====================================================  ==========  ======================

Circuit state is exported as 0 (closed), 1 (half-open) or 2 (open).

Live gauges are computed at scrape time from the registry, so they are never stale and
cost nothing between scrapes. Label values are bounded enumerations, so cardinality is
fixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, final

from voice_ai_realtime_gateway.telemetry.events import (
    BargeInHandled,
    BufferOverflow,
    EgressDropped,
    FirstTokenLatency,
    IngressGap,
    ProviderError,
    ResponseTimeline,
    SessionReaped,
    TransportClosed,
    TtsHalted,
)
from voice_ai_realtime_gateway.telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    Summary,
)

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.production.circuit import CircuitState
    from voice_ai_realtime_gateway.telemetry.events import TelemetryEvent
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

__all__ = ["GatewayMetrics", "SessionRegistry", "TrackedCall"]

_NS_PER_S: Final = 1e9
_NS_PER_US: Final = 1e3
GLASS_TO_GLASS_BUCKETS_S: Final = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
_CIRCUIT_LEVELS: Final = {"closed": 0.0, "half_open": 1.0, "open": 2.0}
HALT_BUCKETS_US: Final = (50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0, 50_000.0)


class TrackedCall(Protocol):
    """A live call as seen by the registry, the drain controller and the reaper."""

    @property
    def trace(self) -> TraceContext:
        """The call's correlation context."""

    @property
    def started_ns(self) -> int:
        """When the call started (monotonic)."""

    @property
    def deadline_ns(self) -> int:
        """When the call's admitted duration ends (monotonic)."""

    def ring_bytes(self) -> int:
        """Audio payload bytes currently queued in the call's rings."""

    async def terminate(self, code: int, reason: str, /) -> None:
        """Close the client socket and let the call wind down."""

    async def quarantine(self) -> None:
        """Force-isolate the call (zombie reaping)."""


@final
class SessionRegistry:
    """The set of live calls."""

    __slots__ = ("_calls",)

    def __init__(self) -> None:
        """Start empty."""
        self._calls: set[TrackedCall] = set()

    def add(self, call: TrackedCall) -> None:
        """Track ``call``."""
        self._calls.add(call)

    def discard(self, call: TrackedCall) -> None:
        """Stop tracking ``call``."""
        self._calls.discard(call)

    def __len__(self) -> int:
        """Live calls."""
        return len(self._calls)

    def calls(self) -> tuple[TrackedCall, ...]:
        """Snapshot of the live calls."""
        return tuple(self._calls)

    def ring_bytes(self) -> int:
        """Audio bytes queued across every live call."""
        return sum(call.ring_bytes() for call in self._calls)


@final
class GatewayMetrics:
    """Metric families for the whole gateway, updated from telemetry events."""

    __slots__ = (
        "barge_ins",
        "circuit_rejections",
        "circuit_state",
        "egress_drops",
        "first_token",
        "glass_to_glass",
        "halt",
        "ingress_drops",
        "provider_errors",
        "reaped",
        "registry",
        "sessions",
        "transport_closed",
    )

    def __init__(self, sessions: SessionRegistry) -> None:
        """Register every family; live gauges read from ``sessions``."""
        registry = MetricsRegistry()
        self.registry = registry
        registry.register(
            Gauge(
                "voice_gateway_active_sessions_count",
                "Calls currently connected.",
                function=lambda: len(sessions),
            )
        )
        registry.register(
            Gauge(
                "voice_gateway_active_audio_rings_bytes",
                "Audio payload bytes queued in live ingress, utterance and egress rings.",
                function=sessions.ring_bytes,
            )
        )
        self.glass_to_glass = registry.register(
            Histogram(
                "voice_gateway_glass_to_glass_seconds",
                "End of user speech (VAD offset) to first agent audio packet on the socket.",
                GLASS_TO_GLASS_BUCKETS_S,
            )
        )
        self.first_token = registry.register(
            Summary(
                "voice_gateway_first_token_latency_seconds",
                "End of user turn to first LLM token.",
            )
        )
        self.halt = registry.register(
            Histogram(
                "voice_gateway_tts_barge_in_halt_microseconds",
                "Barge-in VAD onset to TTS stream closed.",
                HALT_BUCKETS_US,
            )
        )
        self.barge_ins = registry.register(
            Counter("voice_gateway_barge_in_total", "Barge-ins handled.")
        )
        self.ingress_drops = registry.register(
            Counter(
                "voice_gateway_packet_drops_ingress_total",
                "Ingress audio frames lost (network) or evicted (overflow).",
                ("reason",),
            )
        )
        self.egress_drops = registry.register(
            Counter(
                "voice_gateway_packet_drops_egress_total",
                "Agent audio frames discarded because the client was gone.",
            )
        )
        self.provider_errors = registry.register(
            Counter(
                "voice_gateway_upstream_provider_errors_total",
                "Upstream model provider failures.",
                ("provider", "kind", "fatal"),
            )
        )
        self.transport_closed = registry.register(
            Counter(
                "voice_gateway_transport_closed_total",
                "Client connections closed, by close code.",
                ("code",),
            )
        )
        self.sessions = registry.register(
            Counter(
                "voice_gateway_sessions_total",
                "Session outcomes: admitted, rejected, completed, faulted.",
                ("outcome",),
            )
        )

        self.circuit_state = registry.register(
            Gauge(
                "voice_gateway_circuit_state",
                "Upstream circuit breaker state: 0 closed, 1 half-open, 2 open.",
                ("provider",),
            )
        )
        self.circuit_rejections = registry.register(
            Counter(
                "voice_gateway_circuit_rejections_total",
                "Calls refused at admission because an upstream circuit was open.",
                ("provider",),
            )
        )
        self.reaped = registry.register(
            Counter("voice_gateway_sessions_reaped_total", "Zombie sessions force-isolated.")
        )

    def record_circuit(self, provider: str, state: CircuitState) -> None:
        """Export a breaker transition (wire as ``CircuitRegistry(on_change=...)``)."""
        self.circuit_state.labels(provider).set(_CIRCUIT_LEVELS[state])

    def record_circuit_rejection(self, provider: str) -> None:
        """Count a call refused because ``provider``'s circuit was open."""
        self.circuit_rejections.labels(provider).inc()

    def record_session(self, outcome: str) -> None:
        """Count a session outcome (``admitted``, ``rejected``, ``completed``, ``faulted``)."""
        self.sessions.labels(outcome).inc()

    def render(self) -> str:
        """Prometheus text exposition of every family."""
        return self.registry.render()

    def emit(self, event: TelemetryEvent, /) -> None:  # noqa: C901 -- one arm per event type
        """Fold one telemetry event into the metrics."""
        if isinstance(event, ResponseTimeline):
            if event.glass_to_glass_ns is not None:
                self.glass_to_glass.observe(event.glass_to_glass_ns / _NS_PER_S)
        elif isinstance(event, FirstTokenLatency):
            self.first_token.observe(event.latency_ns / _NS_PER_S)
        elif isinstance(event, TtsHalted):
            self.halt.observe(event.halt_ns / _NS_PER_US)
        elif isinstance(event, BargeInHandled):
            self.barge_ins.inc()
        elif isinstance(event, BufferOverflow):
            self.ingress_drops.labels("overflow").inc(event.evicted_frames)
        elif isinstance(event, IngressGap):
            self.ingress_drops.labels("network_loss").inc(event.lost)
        elif isinstance(event, EgressDropped):
            self.egress_drops.inc()
        elif isinstance(event, ProviderError):
            fatal = "true" if event.fatal else "false"
            self.provider_errors.labels(event.provider, event.kind, fatal).inc()
        elif isinstance(event, TransportClosed):
            self.transport_closed.labels(str(event.code)).inc()
        elif isinstance(event, SessionReaped):
            self.reaped.inc()
