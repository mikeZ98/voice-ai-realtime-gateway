"""Per-provider circuit breakers: stop spending pipeline compute on a failing upstream.

State machine (one breaker per provider)::

    CLOSED ──(calls >= min_calls and failure ratio >= threshold in the rolling window)──► OPEN
    OPEN ──(cooldown elapsed)──► HALF_OPEN
    HALF_OPEN ──(probes consecutive successes)──► CLOSED     (window and backoff reset)
    HALF_OPEN ──(any failure)──► OPEN                        (cooldown grows exponentially)

* **Rolling window.** ``window_s`` is split into ``buckets`` time buckets; recording is
  O(1) and memory is O(buckets), whatever the event rate. Old buckets age out lazily.
* **Exponential recovery.** The n-th consecutive opening waits
  ``min(max_open_s, open_s * backoff ** (n - 1))`` before probing, so a flapping
  provider is probed less and less often. A clean recovery resets the backoff.
* **Bounded probing.** While ``HALF_OPEN`` at most ``probes`` calls are let through per
  ``open_s`` period. Probe slots expire, so a probe that never reports back cannot wedge
  the breaker.

:class:`CircuitRegistry` is a ``TelemetrySink``: ``ProviderConnected`` counts as a
success and ``ProviderError`` as a failure. Admission asks :meth:`CircuitRegistry.blocking`
before building a pipeline, and calls whose required providers are open are refused with
``1013`` (try again later). :meth:`CircuitRegistry.select` lets a pipeline factory fall
back to the first healthy alternative.

Confined to one event loop, like the rest of the gateway.
"""

from __future__ import annotations

import math
import time
from array import array
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Final, final, override

from voice_ai_realtime_gateway.telemetry.events import ProviderConnected, ProviderError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.telemetry.events import TelemetryEvent

__all__ = ["CircuitBreaker", "CircuitPolicy", "CircuitRegistry", "CircuitState"]

_NS_PER_S: Final = 1_000_000_000


@unique
class CircuitState(StrEnum):
    """Breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


type ChangeListener = Callable[[str, CircuitState], None]


@dataclass(frozen=True, slots=True, kw_only=True)
class CircuitPolicy:
    """Breaker tuning.

    Attributes:
        window_s: rolling window over which the failure ratio is computed.
        buckets: time buckets in the window (resolution ``window_s / buckets``).
        min_calls: outcomes required in the window before the breaker may open.
        failure_ratio: failure share in ``(0, 1]`` that opens the breaker.
        open_s: first cooldown, and the half-open probe period.
        max_open_s: cooldown cap.
        backoff: cooldown growth per consecutive opening (``>= 1``).
        probes: half-open successes needed to close, and probes allowed per period.
    """

    window_s: float = 30.0
    buckets: int = 10
    min_calls: int = 20
    failure_ratio: float = 0.5
    open_s: float = 5.0
    max_open_s: float = 120.0
    backoff: float = 2.0
    probes: int = 3

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: on any out-of-range parameter.
        """
        durations = (self.window_s, self.open_s, self.max_open_s)
        if not all(math.isfinite(value) and value > 0 for value in durations):
            msg = "window_s, open_s and max_open_s must be positive and finite"
            raise ValueError(msg)
        if self.open_s > self.max_open_s or not (math.isfinite(self.backoff) and self.backoff >= 1):
            msg = "need open_s <= max_open_s and a finite backoff >= 1"
            raise ValueError(msg)
        if self.buckets < 1 or self.min_calls < 1 or self.probes < 1:
            msg = "buckets, min_calls and probes must be >= 1"
            raise ValueError(msg)
        if not 0 < self.failure_ratio <= 1:
            msg = f"failure_ratio must be in (0, 1], got {self.failure_ratio}"
            raise ValueError(msg)


@final
class _Window:
    """Time-bucketed success / failure counts."""

    __slots__ = ("_bucket_ns", "_failures", "_ids", "_successes")

    def __init__(self, window_ns: int, buckets: int) -> None:
        self._bucket_ns = max(1, window_ns // buckets)
        self._ids = array("q", [-1] * buckets)
        self._successes = array("q", bytes(8 * buckets))
        self._failures = array("q", bytes(8 * buckets))

    def record(self, now_ns: int, *, ok: bool) -> None:
        bucket = now_ns // self._bucket_ns
        slot = bucket % len(self._ids)
        if self._ids[slot] != bucket:
            self._ids[slot] = bucket
            self._successes[slot] = 0
            self._failures[slot] = 0
        if ok:
            self._successes[slot] += 1
        else:
            self._failures[slot] += 1

    def totals(self, now_ns: int) -> tuple[int, int]:
        """``(calls, failures)`` within the window ending at ``now_ns``."""
        current = now_ns // self._bucket_ns
        oldest = current - len(self._ids)
        calls = failures = 0
        for slot, bucket in enumerate(self._ids):
            if oldest < bucket <= current:
                calls += self._successes[slot] + self._failures[slot]
                failures += self._failures[slot]
        return calls, failures

    def clear(self) -> None:
        for slot in range(len(self._ids)):
            self._ids[slot] = -1


@final
class CircuitBreaker:
    """Rolling-window breaker for one upstream provider."""

    __slots__ = (
        "_clock",
        "_cooldown_s",
        "_on_change",
        "_opened",
        "_policy",
        "_probe_period_start",
        "_probes_admitted",
        "_reopen_at",
        "_state",
        "_successes",
        "_window",
        "name",
    )

    def __init__(
        self,
        name: str,
        policy: CircuitPolicy | None = None,
        *,
        clock: MonotonicClock = time.monotonic_ns,
        on_change: ChangeListener | None = None,
    ) -> None:
        """Start ``CLOSED``."""
        self.name = name
        self._policy = CircuitPolicy() if policy is None else policy
        self._clock = clock
        self._on_change = on_change
        self._window = _Window(int(self._policy.window_s * _NS_PER_S), self._policy.buckets)
        self._state = CircuitState.CLOSED
        self._opened = 0
        self._reopen_at = 0
        self._cooldown_s = 0.0
        self._successes = 0
        self._probes_admitted = 0
        self._probe_period_start = 0

    @property
    def state(self) -> CircuitState:
        """Current state (an elapsed cooldown turns ``OPEN`` into ``HALF_OPEN``)."""
        if self._state is CircuitState.OPEN and self._clock() >= self._reopen_at:
            self._successes = 0
            self._probes_admitted = 0
            self._probe_period_start = self._clock()
            self._move(CircuitState.HALF_OPEN)
        return self._state

    @property
    def consecutive_openings(self) -> int:
        """Openings since the last clean recovery (drives the backoff)."""
        return self._opened

    @property
    def current_cooldown_s(self) -> float:
        """Cooldown of the most recent opening (0 before the first)."""
        return self._cooldown_s

    def cooldown_s(self) -> float:
        """Cooldown the next opening will apply."""
        policy = self._policy
        return min(policy.max_open_s, policy.open_s * policy.backoff**self._opened)

    def available(self) -> bool:
        """Whether a call could be admitted now, without claiming a probe slot."""
        state = self.state
        if state is CircuitState.HALF_OPEN:
            return self._probe_slot_free()
        return state is CircuitState.CLOSED

    def allow(self) -> bool:
        """Admit one call (claims a probe slot while ``HALF_OPEN``)."""
        if not self.available():
            return False
        if self._state is CircuitState.HALF_OPEN:
            self._probes_admitted += 1
        return True

    def record_success(self) -> None:
        """Count one successful upstream interaction."""
        state = self.state
        if state is CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self._policy.probes:
                self._opened = 0
                self._window.clear()
                self._move(CircuitState.CLOSED)
        elif state is CircuitState.CLOSED:
            self._window.record(self._clock(), ok=True)

    def record_failure(self) -> None:
        """Count one failed upstream interaction."""
        state = self.state
        if state is CircuitState.HALF_OPEN:
            self._open()
        elif state is CircuitState.CLOSED:
            now = self._clock()
            self._window.record(now, ok=False)
            calls, failures = self._window.totals(now)
            if calls >= self._policy.min_calls and failures >= self._policy.failure_ratio * calls:
                self._open()

    @override
    def __repr__(self) -> str:
        return f"CircuitBreaker(name={self.name}, state={self._state.value}, opened={self._opened})"

    def _probe_slot_free(self) -> bool:
        now = self._clock()
        if now - self._probe_period_start >= self._policy.open_s * _NS_PER_S:
            self._probe_period_start = now
            self._probes_admitted = 0
        return self._probes_admitted < self._policy.probes

    def _open(self) -> None:
        self._cooldown_s = self.cooldown_s()
        self._reopen_at = self._clock() + int(self._cooldown_s * _NS_PER_S)
        self._opened += 1
        self._window.clear()
        self._move(CircuitState.OPEN)

    def _move(self, state: CircuitState) -> None:
        self._state = state
        if self._on_change is not None:
            self._on_change(self.name, state)


@final
class CircuitRegistry:
    """One breaker per provider, fed by telemetry."""

    __slots__ = ("_breakers", "_clock", "_on_change", "_policy")

    def __init__(
        self,
        policy: CircuitPolicy | None = None,
        *,
        clock: MonotonicClock = time.monotonic_ns,
        on_change: ChangeListener | None = None,
    ) -> None:
        """Create breakers lazily with ``policy``."""
        self._policy = policy
        self._clock = clock
        self._on_change = on_change
        self._breakers: dict[str, CircuitBreaker] = {}

    def breaker(self, provider: str) -> CircuitBreaker:
        """The breaker of ``provider`` (created ``CLOSED`` on first use)."""
        breaker = self._breakers.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(
                provider, self._policy, clock=self._clock, on_change=self._on_change
            )
            self._breakers[provider] = breaker
        return breaker

    def states(self) -> dict[str, CircuitState]:
        """Current state of every known provider."""
        return {name: breaker.state for name, breaker in self._breakers.items()}

    def blocking(self, providers: Sequence[str]) -> str | None:
        """The first of ``providers`` that cannot take a call, else ``None``.

        Probe slots are claimed only if every provider is available, so a refused
        call never consumes another provider's probe.
        """
        for provider in providers:
            if not self.breaker(provider).available():
                return provider
        for provider in providers:
            self.breaker(provider).allow()
        return None

    def select(self, *candidates: str) -> str | None:
        """The first candidate that admits a call (fallback selection), else ``None``."""
        for candidate in candidates:
            if self.breaker(candidate).allow():
                return candidate
        return None

    def emit(self, event: TelemetryEvent, /) -> None:
        """Feed provider outcomes into the breakers."""
        if isinstance(event, ProviderConnected):
            self.breaker(event.provider).record_success()
        elif isinstance(event, ProviderError):
            self.breaker(event.provider).record_failure()
