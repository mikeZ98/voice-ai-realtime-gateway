"""Telemetry sinks.

Contract: :meth:`TelemetrySink.emit` is synchronous, non-blocking, allocation-light and
**must not raise** -- it is invoked on the realtime hot path between ``await`` points.
Exporters that do I/O must buffer internally and flush from their own task.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Protocol, final, override

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.events import TelemetryEvent

__all__ = ["FanoutSink", "FlightRecorder", "NullSink", "TelemetrySink"]


class TelemetrySink(Protocol):
    """Destination for typed telemetry events."""

    def emit(self, event: TelemetryEvent, /) -> None:
        """Accept one event. Must be O(1), non-blocking and never raise."""


@final
class NullSink:
    """Discards every event (zero-cost default)."""

    __slots__ = ()

    def emit(self, event: TelemetryEvent, /) -> None:
        """Drop ``event``."""


@final
class FanoutSink:
    """Delivers every event to each of several sinks, in order."""

    __slots__ = ("_sinks",)

    def __init__(self, *sinks: TelemetrySink) -> None:
        """Fan out to ``sinks``."""
        self._sinks = sinks

    def emit(self, event: TelemetryEvent, /) -> None:
        """Forward ``event`` to every sink."""
        for sink in self._sinks:
            sink.emit(event)


@final
class FlightRecorder:
    """Bounded in-memory recorder keeping the most recent ``capacity`` events.

    Memory is O(capacity) regardless of event rate; older events are evicted and
    counted in :attr:`evicted`. Suitable as a post-mortem black box per session.
    """

    __slots__ = ("_events", "_evicted")

    def __init__(self, capacity: int = 1024) -> None:
        """Create a recorder holding at most ``capacity`` events."""
        if capacity < 1:
            msg = f"capacity must be >= 1, got {capacity}"
            raise ValueError(msg)
        self._events: deque[TelemetryEvent] = deque(maxlen=capacity)
        self._evicted = 0

    def emit(self, event: TelemetryEvent, /) -> None:
        """Record ``event``, evicting the oldest one when full."""
        if len(self._events) == self._events.maxlen:
            self._evicted += 1
        self._events.append(event)

    @property
    def events(self) -> tuple[TelemetryEvent, ...]:
        """Snapshot of retained events, oldest first."""
        return tuple(self._events)

    @property
    def evicted(self) -> int:
        """Number of events evicted because the recorder was full."""
        return self._evicted

    def of_type[E](self, kind: type[E]) -> list[E]:
        """Return retained events that are instances of ``kind``, oldest first."""
        return [event for event in self._events if isinstance(event, kind)]

    def clear(self) -> None:
        """Forget all retained events (the eviction counter is preserved)."""
        self._events.clear()

    @override
    def __repr__(self) -> str:
        return f"FlightRecorder(len={len(self._events)}, evicted={self._evicted})"
