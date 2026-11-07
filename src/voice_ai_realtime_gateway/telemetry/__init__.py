"""Telemetry leaf layer: correlation IDs, typed events, sinks and latency windows."""

from voice_ai_realtime_gateway.telemetry.events import (
    BargeInHandled,
    BufferOverflow,
    FirstTokenLatency,
    PacketLatency,
    StateTransitioned,
    TaskFailed,
    TelemetryEvent,
    TransitionRejected,
    VadSpeechStarted,
    VadSpeechStopped,
)
from voice_ai_realtime_gateway.telemetry.ids import (
    AUDIO_SPAN_ID_HEADER,
    SESSION_ID_HEADER,
    AudioSpanId,
    InvalidTraceIdError,
    SessionId,
    TraceContext,
    new_session_id,
    new_span_id,
    parse_session_id,
    parse_span_id,
)
from voice_ai_realtime_gateway.telemetry.latency import (
    FirstTokenTracker,
    GlassToGlassTracker,
    LatencySummary,
    LatencyWindow,
)
from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder, NullSink, TelemetrySink

__all__ = [
    "AUDIO_SPAN_ID_HEADER",
    "SESSION_ID_HEADER",
    "AudioSpanId",
    "BargeInHandled",
    "BufferOverflow",
    "FirstTokenLatency",
    "FirstTokenTracker",
    "FlightRecorder",
    "GlassToGlassTracker",
    "InvalidTraceIdError",
    "LatencySummary",
    "LatencyWindow",
    "NullSink",
    "PacketLatency",
    "SessionId",
    "StateTransitioned",
    "TaskFailed",
    "TelemetryEvent",
    "TelemetrySink",
    "TraceContext",
    "TransitionRejected",
    "VadSpeechStarted",
    "VadSpeechStopped",
    "new_session_id",
    "new_span_id",
    "parse_session_id",
    "parse_span_id",
]
