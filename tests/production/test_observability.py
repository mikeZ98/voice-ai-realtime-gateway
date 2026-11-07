"""Gateway metrics: every telemetry event lands in the right Prometheus family."""

from __future__ import annotations

from typing import TYPE_CHECKING

from voice_ai_realtime_gateway.production.observability import GatewayMetrics, SessionRegistry
from voice_ai_realtime_gateway.telemetry.events import (
    BargeInHandled,
    BufferOverflow,
    EgressDropped,
    FirstTokenLatency,
    IngressGap,
    ProviderError,
    ResponseTimeline,
    TransportClosed,
    TtsHalted,
    VadSpeechStarted,
)

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext


class Call:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes

    def ring_bytes(self) -> int:
        return self.nbytes


def timeline(trace: TraceContext, egress: int | None) -> ResponseTimeline:
    return ResponseTimeline(
        trace=trace, at_ns=0, user_span_id="00000000000000aa", speech_end_ns=1_000_000_000,
        transcript_ns=None, first_token_ns=None, first_audio_ns=None, first_egress_ns=egress,
        completed=True, generated_chars=0, spoken_chars=0,
    )  # fmt: skip


def test_events_fold_into_the_documented_families(trace: TraceContext) -> None:
    sessions = SessionRegistry()
    metrics = GatewayMetrics(sessions)
    for event in (
        timeline(trace, 1_250_000_000),
        timeline(trace, None),
        FirstTokenLatency(trace=trace, at_ns=0, latency_ns=400_000_000),
        TtsHalted(trace=trace, at_ns=0, halt_ns=240_000),
        BargeInHandled(
            trace=trace,
            at_ns=0,
            detect_to_draining_ns=1,
            budget_ns=2,
            within_budget=True,
            cancelled_tasks=1,
            dropped_frames=1,
            stragglers=0,
        ),
        BufferOverflow(trace=trace, at_ns=0, evicted_frames=3, evicted_bytes=9),
        IngressGap(trace=trace, at_ns=0, expected_sequence=2, received_sequence=7),
        EgressDropped(trace=trace, at_ns=0, sequence=1),
        ProviderError(
            trace=trace,
            at_ns=0,
            provider="deepgram",
            kind="transient",
            fatal=False,
            attempt=1,
            detail="",
        ),
        ProviderError(
            trace=trace, at_ns=0, provider="openai", kind="quota", fatal=True, attempt=1, detail=""
        ),
        TransportClosed(trace=trace, at_ns=0, code=1000, reason="", initiator="local"),
        VadSpeechStarted(trace=trace, at_ns=0),  # not a metric: ignored
    ):
        metrics.emit(event)
    metrics.record_session("admitted")
    text = metrics.render()
    for line in (
        'voice_gateway_glass_to_glass_seconds_bucket{le="1.5"} 1',
        "voice_gateway_glass_to_glass_seconds_count 1",
        'voice_gateway_first_token_latency_seconds{quantile="0.5"} 0.4',
        'voice_gateway_tts_barge_in_halt_microseconds_bucket{le="250.0"} 1',
        "voice_gateway_barge_in_total 1.0",
        'voice_gateway_packet_drops_ingress_total{reason="overflow"} 3.0',
        'voice_gateway_packet_drops_ingress_total{reason="network_loss"} 5.0',
        "voice_gateway_packet_drops_egress_total 1.0",
        (
            "voice_gateway_upstream_provider_errors_total"
            '{provider="deepgram",kind="transient",fatal="false"} 1.0'
        ),
        (
            "voice_gateway_upstream_provider_errors_total"
            '{provider="openai",kind="quota",fatal="true"} 1.0'
        ),
        'voice_gateway_transport_closed_total{code="1000"} 1.0',
        'voice_gateway_sessions_total{outcome="admitted"} 1.0',
        "voice_gateway_active_sessions_count 0.0",
    ):
        assert line in text, line


def test_live_gauges_read_the_registry_at_scrape_time() -> None:
    sessions = SessionRegistry()
    metrics = GatewayMetrics(sessions)
    calls = [Call(100), Call(28)]
    for call in calls:
        sessions.add(call)
    text = metrics.render()
    assert "voice_gateway_active_sessions_count 2.0" in text
    assert "voice_gateway_active_audio_rings_bytes 128.0" in text
    sessions.discard(calls[0])
    assert "voice_gateway_active_audio_rings_bytes 28.0" in metrics.render()
