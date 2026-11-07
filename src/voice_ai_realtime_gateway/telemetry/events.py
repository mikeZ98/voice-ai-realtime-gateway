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
    "PacketLatency",
    "StateTransitioned",
    "TaskFailed",
    "TelemetryEvent",
    "TransitionRejected",
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
)
