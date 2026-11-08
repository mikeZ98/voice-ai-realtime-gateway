"""Composition root: settings from the environment, wiring, and a runnable demo app.

Environment (all prefixed ``GATEWAY_``):

=================================  ========  ==============================================
variable                           required  meaning
=================================  ========  ==============================================
``GATEWAY_KEYS``                   yes       ``kid=base64url-secret[,kid=...]`` (>= 32 bytes)
``GATEWAY_TENANTS``                yes       JSON ``{"tenant": {"max_concurrent_sessions": n,
                                             "max_session_duration_s": s, "codecs": [...]}}``
``GATEWAY_AUDIENCE``               no        token audience (default ``voice-gateway``)
``GATEWAY_WEBSOCKET_PATH``         no        media route (default ``/v1/voice``)
``GATEWAY_DRAIN_TIMEOUT_S``        no        phase-1 drain budget (default 30)
``GATEWAY_FINAL_SCRAPE_GRACE_S``   no        hold after drain for a last scrape (default 15)
``GATEWAY_MEDIA_TIMEOUT_S``        no        slow-loris bound (default 10)
``GATEWAY_REAPER_GRACE_S``         no        zombie grace past deadline (default 30)
``GATEWAY_REQUIRED_PROVIDERS``     no        comma list gated by circuit breakers
=================================  ========  ==============================================

``build_app(settings, factory)`` wires metrics, circuits, admission, the server, the
drain controller, the reaper and the status endpoint into one :class:`GatewayAsgiApp`.
Production deployments supply their own ``factory`` (real provider adapters behind
their connectors). ``create_demo_app`` serves the deterministic in-memory doubles, so a
freshly built image can be smoke-tested without provider credentials.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, final

from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.deploy.asgi import GatewayAsgiApp
from voice_ai_realtime_gateway.deploy.lifecycle import DrainController
from voice_ai_realtime_gateway.inference.agent import VoiceAgentPipeline
from voice_ai_realtime_gateway.inference.doubles import ScriptedLLM, ToneASR, ToneTTS
from voice_ai_realtime_gateway.production.admission import DEFAULT_AUDIENCE, TokenAuthenticator
from voice_ai_realtime_gateway.production.circuit import CircuitRegistry
from voice_ai_realtime_gateway.production.observability import GatewayMetrics, SessionRegistry
from voice_ai_realtime_gateway.production.reaper import ZombieReaper
from voice_ai_realtime_gateway.production.server import GatewayServer, ServerConfig, TenantPolicy
from voice_ai_realtime_gateway.production.status import StatusApp
from voice_ai_realtime_gateway.telemetry.sink import FanoutSink
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from collections.abc import Mapping

    from voice_ai_realtime_gateway.production.server import CallContext, PipelineFactory

__all__ = ["GatewaySettings", "build_app", "create_demo_app", "demo_pipeline"]


def _keys(raw: str) -> dict[str, bytes]:
    keys: dict[str, bytes] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        key_id, separator, secret = entry.partition("=")
        if not separator or not key_id:
            msg = "GATEWAY_KEYS entries must look like kid=base64url-secret"
            raise ValueError(msg)
        try:
            padded = secret + "=" * (-len(secret) % 4)
            keys[key_id] = base64.b64decode(padded, altchars=b"-_", validate=True)
        except (binascii.Error, ValueError) as exc:
            msg = f"GATEWAY_KEYS secret for {key_id!r} is not base64url"
            raise ValueError(msg) from exc
    if not keys:
        msg = "GATEWAY_KEYS must define at least one key"
        raise ValueError(msg)
    return keys


def _tenants(raw: str) -> dict[str, TenantPolicy]:
    try:
        document = json.loads(raw)
    except ValueError as exc:
        msg = "GATEWAY_TENANTS must be JSON"
        raise ValueError(msg) from exc
    if not isinstance(document, dict) or not document:
        msg = "GATEWAY_TENANTS must be a non-empty JSON object"
        raise ValueError(msg)
    tenants: dict[str, TenantPolicy] = {}
    for name, spec in document.items():
        try:
            codecs = frozenset(
                AudioCodec(c) for c in spec.get("codecs", [c.value for c in AudioCodec])
            )
            tenants[name] = TenantPolicy(
                max_concurrent_sessions=int(spec["max_concurrent_sessions"]),
                max_session_duration_s=float(spec["max_session_duration_s"]),
                allowed_codecs=codecs,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            msg = f"tenant {name!r} is invalid: {exc}"
            raise ValueError(msg) from exc
    return tenants


def _seconds(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        msg = f"{name} must be a non-negative number of seconds, got {raw!r}"
        raise ValueError(msg)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class GatewaySettings:
    """Deployment configuration."""

    keys: Mapping[str, bytes]
    tenants: Mapping[str, TenantPolicy]
    audience: str = DEFAULT_AUDIENCE
    websocket_path: str = "/v1/voice"
    drain_timeout_s: float = 30.0
    final_scrape_grace_s: float = 15.0
    media_timeout_s: float = 10.0
    reaper_grace_s: float = 30.0
    required_providers: tuple[str, ...] = field(default=())

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Self:
        """Read ``GATEWAY_*`` variables.

        Raises:
            ValueError: on a missing or malformed variable (message names it).
        """
        for required in ("GATEWAY_KEYS", "GATEWAY_TENANTS"):
            if not environ.get(required):
                msg = f"{required} is required"
                raise ValueError(msg)
        providers = environ.get("GATEWAY_REQUIRED_PROVIDERS", "")
        return cls(
            keys=_keys(environ["GATEWAY_KEYS"]),
            tenants=_tenants(environ["GATEWAY_TENANTS"]),
            audience=environ.get("GATEWAY_AUDIENCE", DEFAULT_AUDIENCE),
            websocket_path=environ.get("GATEWAY_WEBSOCKET_PATH", "/v1/voice"),
            drain_timeout_s=_seconds(environ, "GATEWAY_DRAIN_TIMEOUT_S", 30.0),
            final_scrape_grace_s=_seconds(environ, "GATEWAY_FINAL_SCRAPE_GRACE_S", 15.0),
            media_timeout_s=_seconds(environ, "GATEWAY_MEDIA_TIMEOUT_S", 10.0) or 10.0,
            reaper_grace_s=_seconds(environ, "GATEWAY_REAPER_GRACE_S", 30.0) or 30.0,
            required_providers=tuple(p.strip() for p in providers.split(",") if p.strip()),
        )


@final
class _Wiring:
    """Shared objects of one process (kept for introspection in tests and ops)."""

    __slots__ = ("circuits", "metrics", "sessions")

    def __init__(self) -> None:
        self.sessions = SessionRegistry()
        self.metrics = GatewayMetrics(self.sessions)
        self.circuits = CircuitRegistry(on_change=self.metrics.record_circuit)


def build_app(settings: GatewaySettings, factory: PipelineFactory) -> GatewayAsgiApp:
    """Wire one deployable gateway process."""
    wiring = _Wiring()
    sink = FanoutSink(wiring.metrics, wiring.circuits)
    server = GatewayServer(
        authenticator=TokenAuthenticator(settings.keys, audience=settings.audience),
        tenants=settings.tenants,
        metrics=wiring.metrics,
        sessions=wiring.sessions,
        factory=factory,
        sink=sink,
        config=ServerConfig(media_timeout_s=settings.media_timeout_s),
        circuits=wiring.circuits,
        required_providers=settings.required_providers,
    )
    drain = DrainController(
        server=server,
        sessions=wiring.sessions,
        drain_timeout_s=settings.drain_timeout_s,
        final_scrape_grace_s=settings.final_scrape_grace_s,
    )
    status = StatusApp(
        metrics=wiring.metrics.render, ready=lambda: server.ready, drained=lambda: drain.drained
    )
    reaper = ZombieReaper(wiring.sessions, sink=sink, grace_s=settings.reaper_grace_s)
    return GatewayAsgiApp(
        server=server,
        status=status,
        drain=drain,
        reaper=reaper,
        websocket_path=settings.websocket_path,
    )


def demo_pipeline(context: CallContext) -> VoiceAgentPipeline:
    """Deterministic in-memory voice agent (tone ASR, scripted LLM, tone TTS)."""
    return VoiceAgentPipeline(
        transport=context.transport,
        vad=EnergyVAD(),
        asr=ToneASR({1_000: "hello", 2_000: "world"}),
        llm=ScriptedLLM("Hello from the voice gateway demo."),
        tts=ToneTTS(),
        sink=context.sink,
    )


def create_demo_app() -> GatewayAsgiApp:
    """ASGI factory for the demo image: ``uvicorn --factory ...app:create_demo_app``."""
    return build_app(GatewaySettings.from_env(os.environ), demo_pipeline)
