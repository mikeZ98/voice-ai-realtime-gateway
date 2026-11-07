"""Typed, immutable, slot-only telemetry events.

Every event is a frozen ``slots`` dataclass (no ``__dict__``, ~1 allocation each) that
carries the :class:`~voice_ai_realtime_gateway.telemetry.ids.TraceContext` it belongs to
and a monotonic nanosecond timestamp ``at_ns``. Telemetry is a leaf layer: domain
enums are serialised as their ``str`` values so this module imports nothing upward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

__all__ = [
    "BargeInHandled",
    "BufferOverflow",
    "FirstTokenLatency",
    "IngressGap",
    "PacketLatency",
    "ResponseTimeline",
    "StateTransitioned",
    "TaskFailed",
    "TelemetryEvent",
    "TranscriptReceived",
    "TransitionRejected",
    "TransportClosed",
    "VadSpeechStarted",
    "VadSpeechStopped",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class VadSpeechStarted:
    """Voice activity detector reported the onset of user speech."""

    trace: TraceContext
    at_ns: int


@dataclass(frozen=True, slots=True, kw_only=True)
class VadSpeechStopped:
    """Voice activity detector reported the end of user speech."""

    trace: TraceContext
    at_ns: int
    speech_duration_ns: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class FirstTokenLatency:
    """End of user turn -> first generated token of the agent response."""

    trace: TraceContext
    at_ns: int
    latency_ns: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PacketLatency:
    """Glass-to-glass latency of one audio packet: origin timestamp -> egress."""

    trace: TraceContext
    at_ns: int
    sequence: int
    latency_ns: int


@dataclass(frozen=True, slots=True, kw_only=True)
class StateTransitioned:
    """The session state machine committed a transition."""

    trace: TraceContext
    at_ns: int
    lamport: int
    source: str
    target: str
    trigger: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TransitionRejected:
    """The session state machine refused an invalid ``(state, trigger)`` pair."""

    trace: TraceContext
    at_ns: int
    lamport: int
    state: str
    trigger: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeInHandled:
    """User speech interrupted agent output; playback was cancelled and drained."""

    trace: TraceContext
    at_ns: int
    detect_to_draining_ns: int
    budget_ns: int
    within_budget: bool
    cancelled_tasks: int
    dropped_frames: int
    stragglers: int


@dataclass(frozen=True, slots=True, kw_only=True)
class BufferOverflow:
    """A drop-oldest ring evicted queued frames to admit a newer one."""

    trace: TraceContext
    at_ns: int
    evicted_frames: int
    evicted_bytes: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskFailed:
    """A supervised session task terminated with an exception."""

    trace: TraceContext
    at_ns: int
    task_name: str
    error_type: str


@dataclass(frozen=True, slots=True, kw_only=True)
class IngressGap:
    """Ingress sequence numbers jumped within a span: packets were lost in the network."""

    trace: TraceContext
    at_ns: int
    expected_sequence: int
    received_sequence: int

    @property
    def lost(self) -> int:
        """Number of packets missing between the expected and the received sequence."""
        return self.received_sequence - self.expected_sequence


@dataclass(frozen=True, slots=True, kw_only=True)
class TransportClosed:
    """The client transport closed (``initiator`` is ``"local"`` or ``"remote"``)."""

    trace: TraceContext
    at_ns: int
    code: int
    reason: str
    initiator: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptReceived:
    """The ASR produced an interim or final transcript for a user utterance span."""

    trace: TraceContext
    at_ns: int
    is_final: bool
    text_chars: int
    confidence: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseTimeline:
    """End-to-end timeline of one agent response, from user speech end to egress.

    ``trace`` is the response span; ``user_span_id`` links the user utterance whose
    packets triggered it. Stage timestamps are ``None`` if the stage never happened
    (for example, a barge-in before any audio was synthesised).
    """

    trace: TraceContext
    at_ns: int
    user_span_id: str
    speech_end_ns: int
    transcript_ns: int | None
    first_token_ns: int | None
    first_audio_ns: int | None
    first_egress_ns: int | None
    completed: bool
    generated_chars: int
    spoken_chars: int

    @property
    def glass_to_glass_ns(self) -> int | None:
        """End of user speech to the first agent audio packet leaving the gateway."""
        if self.first_egress_ns is None:
            return None
        return self.first_egress_ns - self.speech_end_ns


type TelemetryEvent = (
    VadSpeechStarted
    | VadSpeechStopped
    | FirstTokenLatency
    | PacketLatency
    | StateTransitioned
    | TransitionRejected
    | BargeInHandled
    | BufferOverflow
    | TaskFailed
    | IngressGap
    | TransportClosed
    | TranscriptReceived
    | ResponseTimeline
)
