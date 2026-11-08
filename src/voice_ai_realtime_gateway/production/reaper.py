"""Zombie session reaper: isolate calls that outlive their admitted bounds.

Normally the per-call watchdog closes the socket when the admitted duration ends and the
call winds down on its own. A call still registered ``grace_s`` after its deadline is a
**zombie**: some task ignored cancellation, a provider stream hung, or a pipeline
deadlocked. The reaper quarantines exactly that call:

1. close the client socket with ``1011``;
2. abort the session (``FAULT``: cancel and reap turn tasks with a bounded join);
3. cancel the call's own task;
4. release its tenant slot and registry entry, even if the task stays wedged, so one
   zombie can never exhaust a tenant's concurrency or skew the gauges.

Sibling calls are never touched: the reaper acts only through the zombie's own handles.
Each reaped call emits ``SessionReaped`` (``voice_gateway_sessions_reaped_total``).
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, Final, final

from voice_ai_realtime_gateway.telemetry.events import SessionReaped

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.production.observability import SessionRegistry
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink

__all__ = ["ZombieReaper"]

_NS_PER_S: Final = 1_000_000_000


@final
class ZombieReaper:
    """Periodically sweeps the session registry for calls past their deadline."""

    __slots__ = ("_clock", "_grace_ns", "_interval_s", "_sessions", "_sink", "_sleep", "reaped")

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        sessions: SessionRegistry,
        *,
        sink: TelemetrySink,
        grace_s: float = 30.0,
        interval_s: float = 5.0,
        clock: MonotonicClock = time.monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Reap calls more than ``grace_s`` past their deadline, every ``interval_s``.

        Raises:
            ValueError: unless both durations are positive and finite.
        """
        if not all(math.isfinite(v) and v > 0 for v in (grace_s, interval_s)):
            msg = "grace_s and interval_s must be positive and finite"
            raise ValueError(msg)
        self._sessions = sessions
        self._sink = sink
        self._grace_ns = int(grace_s * _NS_PER_S)
        self._interval_s = interval_s
        self._clock = clock
        self._sleep = sleep
        self.reaped = 0

    async def sweep(self) -> int:
        """Quarantine every zombie now; returns how many were reaped."""
        now = self._clock()
        zombies = [c for c in self._sessions.calls() if now > c.deadline_ns + self._grace_ns]
        for call in zombies:
            self._sink.emit(
                SessionReaped(
                    trace=call.trace,
                    at_ns=now,
                    age_ns=now - call.started_ns,
                    reason="exceeded admitted duration",
                )
            )
            await call.quarantine()
            self._sessions.discard(call)
        self.reaped += len(zombies)
        return len(zombies)

    async def run(self) -> None:
        """Sweep forever (cancel the task to stop)."""
        while True:
            await self._sleep(self._interval_s)
            await self.sweep()
