"""Low-overhead latency tracking.

* :class:`LatencyWindow` -- O(1) record into a preallocated ``array('q')`` ring of the
  last *N* samples; percentiles are computed on demand (O(N log N), off the hot path).
* :class:`GlassToGlassTracker` -- per-packet origin -> egress latency.
* :class:`FirstTokenTracker` -- end-of-user-turn -> first agent token, once per turn.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from voice_ai_realtime_gateway.telemetry.events import FirstTokenLatency, PacketLatency

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink

__all__ = [
    "FirstTokenTracker",
    "GlassToGlassTracker",
    "LatencySummary",
    "LatencyWindow",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class LatencySummary:
    """Point-in-time distribution summary of a :class:`LatencyWindow` (nanoseconds)."""

    count: int
    window: int
    min_ns: int
    max_ns: int
    mean_ns: float
    p50_ns: int
    p95_ns: int
    p99_ns: int


@final
class LatencyWindow:
    """Sliding window over the most recent ``capacity`` latency samples."""

    __slots__ = ("_capacity", "_count", "_cursor", "_max", "_min", "_samples", "_total")

    def __init__(self, capacity: int = 1024) -> None:
        """Preallocate storage for ``capacity`` int64 samples."""
        if capacity < 1:
            msg = f"capacity must be >= 1, got {capacity}"
            raise ValueError(msg)
        self._capacity = capacity
        self._samples = array("q", bytes(8 * capacity))
        self._cursor = 0
        self._count = 0
        self._total = 0
        self._min = 0
        self._max = 0

    def record(self, latency_ns: int) -> None:
        """Record one non-negative sample in O(1) without allocating."""
        if latency_ns < 0:
            msg = f"latency must be non-negative, got {latency_ns}"
            raise ValueError(msg)
        self._samples[self._cursor] = latency_ns
        self._cursor = (self._cursor + 1) % self._capacity
        if self._count == 0:
            self._min = self._max = latency_ns
        else:
            self._min = min(self._min, latency_ns)
            self._max = max(self._max, latency_ns)
        self._count += 1
        self._total += latency_ns

    @property
    def count(self) -> int:
        """Lifetime number of recorded samples."""
        return self._count

    def percentile(self, quantile: float) -> int:
        """Nearest-rank percentile over the retained window.

        Args:
            quantile: in ``(0, 1]``.

        Raises:
            ValueError: on an out-of-range quantile or an empty window.
        """
        if not 0.0 < quantile <= 1.0:
            msg = f"quantile must be in (0, 1], got {quantile}"
            raise ValueError(msg)
        retained = min(self._count, self._capacity)
        if retained == 0:
            msg = "percentile of an empty window"
            raise ValueError(msg)
        ordered = sorted(self._samples[:retained])
        return ordered[math.ceil(quantile * retained) - 1]

    def summary(self) -> LatencySummary:
        """Lifetime min/max/mean plus windowed p50/p95/p99.

        Raises:
            ValueError: if nothing has been recorded yet.
        """
        if self._count == 0:
            msg = "summary of an empty window"
            raise ValueError(msg)
        return LatencySummary(
            count=self._count,
            window=min(self._count, self._capacity),
            min_ns=self._min,
            max_ns=self._max,
            mean_ns=self._total / self._count,
            p50_ns=self.percentile(0.50),
            p95_ns=self.percentile(0.95),
            p99_ns=self.percentile(0.99),
        )


@final
class GlassToGlassTracker:
    """Tracks per-packet latency from its origin timestamp to egress."""

    __slots__ = ("_sink", "_window")

    def __init__(self, sink: TelemetrySink, window: LatencyWindow | None = None) -> None:
        """Emit :class:`PacketLatency` events to ``sink`` and aggregate in ``window``."""
        self._sink = sink
        self._window = LatencyWindow() if window is None else window

    @property
    def window(self) -> LatencyWindow:
        """The aggregation window."""
        return self._window

    def record(
        self, *, trace: TraceContext, sequence: int, origin_ns: int, egress_ns: int
    ) -> PacketLatency:
        """Record one packet's ``origin_ns -> egress_ns`` latency.

        Raises:
            ValueError: if ``egress_ns`` precedes ``origin_ns`` (non-monotonic clock).
        """
        latency = egress_ns - origin_ns
        self._window.record(latency)
        event = PacketLatency(trace=trace, at_ns=egress_ns, sequence=sequence, latency_ns=latency)
        self._sink.emit(event)
        return event


@final
class FirstTokenTracker:
    """Measures end-of-user-turn -> first agent token, reporting at most once per turn."""

    __slots__ = ("_sink", "_turn_ended_at", "_window")

    def __init__(self, sink: TelemetrySink, window: LatencyWindow | None = None) -> None:
        """Emit :class:`FirstTokenLatency` events to ``sink`` and aggregate in ``window``."""
        self._sink = sink
        self._window = LatencyWindow() if window is None else window
        self._turn_ended_at: int | None = None

    @property
    def window(self) -> LatencyWindow:
        """The aggregation window."""
        return self._window

    @property
    def armed(self) -> bool:
        """Whether a user turn ended and its first token has not been reported yet."""
        return self._turn_ended_at is not None

    def user_turn_ended(self, at_ns: int) -> None:
        """Arm the tracker at the end of a user turn (re-arming restarts the clock)."""
        self._turn_ended_at = at_ns

    def disarm(self) -> None:
        """Abandon the current measurement (e.g. the turn was interrupted)."""
        self._turn_ended_at = None

    def first_token(self, *, trace: TraceContext, at_ns: int) -> FirstTokenLatency | None:
        """Report the first token of the response; later calls in the turn return ``None``."""
        started = self._turn_ended_at
        if started is None:
            return None
        self._turn_ended_at = None
        latency = at_ns - started
        self._window.record(latency)
        event = FirstTokenLatency(trace=trace, at_ns=at_ns, latency_ns=latency)
        self._sink.emit(event)
        return event
