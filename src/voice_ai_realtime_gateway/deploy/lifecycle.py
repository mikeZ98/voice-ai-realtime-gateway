"""Two-phase graceful drain for zero-downtime rollouts.

State machine::

    SERVING ──trigger()──► DRAINING ──(no live calls, or drain_timeout_s)──► FLUSHING ──► DRAINED

* **Phase 1 (DRAINING).** The server stops admitting: ``/readyz`` turns 503, so the
  load balancer stops routing here, and connections that still arrive are accepted
  and closed with ``1013`` (try again later), so clients reconnect elsewhere. Live calls
  continue for up to ``drain_timeout_s``.
* **Phase 2 (FLUSHING).** Calls still alive at the deadline are closed with ``1001``
  (going away). The ``flush`` hook runs (for example to push final metrics), then the
  controller holds for ``final_scrape_grace_s`` so Prometheus can scrape the final
  counters before the process exits.

Why the trigger is a signal *before* ``SIGTERM``: ASGI servers such as uvicorn close
every open WebSocket with 1012 as soon as they receive ``SIGTERM``, which would cut
live calls mid-sentence. The Kubernetes ``preStop`` hook therefore sends ``SIGUSR1``
(see :mod:`.drain`); the drain completes, ``/drainz`` turns 200, and only then does
``SIGTERM`` arrive. When the gateway owns its event loop, the same controller can be
bound to ``SIGTERM`` directly.
"""

from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Final, Protocol, final

from voice_ai_realtime_gateway._validation import is_non_negative_finite, is_positive_finite
from voice_ai_realtime_gateway.transport.errors import CloseCode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from voice_ai_realtime_gateway.production.observability import SessionRegistry

__all__ = [
    "DEFAULT_DRAIN_SIGNALS",
    "DrainController",
    "DrainPhase",
    "DrainReport",
    "DrainableServer",
]

DEFAULT_DRAIN_SIGNALS: Final = (signal.SIGUSR1,)
_SHUTDOWN_GRACE_S: Final = 5.0


class DrainableServer(Protocol):
    """What the controller needs from the gateway server."""

    def begin_drain(self) -> None:
        """Stop admitting new calls."""

    async def close_all(self, code: int, reason: str) -> int:
        """Close every live call; returns how many."""


class SignalLoop(Protocol):
    """The part of an event loop used for signal handling."""

    def add_signal_handler(self, sig: int, callback: Callable[[], object], /) -> None:
        """Run ``callback`` in the loop when ``sig`` arrives."""

    def remove_signal_handler(self, sig: int, /) -> bool:
        """Stop handling ``sig``."""


@unique
class DrainPhase(StrEnum):
    """Drain lifecycle."""

    SERVING = "serving"
    DRAINING = "draining"
    FLUSHING = "flushing"
    DRAINED = "drained"


@dataclass(frozen=True, slots=True, kw_only=True)
class DrainReport:
    """How a drain went."""

    waited_s: float
    forced: int
    leftover: int


@final
class DrainController:
    """Owns the drain of one gateway process."""

    __slots__ = (
        "_clock",
        "_drain_timeout_s",
        "_final_grace_s",
        "_flush",
        "_phase",
        "_poll_s",
        "_server",
        "_sessions",
        "_signals",
        "_sleep",
        "_task",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        *,
        server: DrainableServer,
        sessions: SessionRegistry,
        drain_timeout_s: float = 30.0,
        final_scrape_grace_s: float = 0.0,
        poll_interval_s: float = 0.25,
        flush: Callable[[], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Drain ``server`` within ``drain_timeout_s``.

        Raises:
            ValueError: on a negative or non-finite duration, or a poll interval that
                is not positive and finite.
        """
        if not (
            is_non_negative_finite(drain_timeout_s)
            and is_non_negative_finite(final_scrape_grace_s)
            and is_positive_finite(poll_interval_s)
        ):
            msg = "durations must be finite and non-negative, poll_interval_s positive"
            raise ValueError(msg)
        self._server = server
        self._sessions = sessions
        self._drain_timeout_s = drain_timeout_s
        self._final_grace_s = final_scrape_grace_s
        self._poll_s = poll_interval_s
        self._flush = flush
        self._clock = clock
        self._sleep = sleep
        self._phase = DrainPhase.SERVING
        self._task: asyncio.Task[DrainReport] | None = None
        self._signals: list[int] = []

    @property
    def phase(self) -> DrainPhase:
        """Current phase."""
        return self._phase

    @property
    def drained(self) -> bool:
        """Whether the drain has finished (``/drainz``)."""
        return self._phase is DrainPhase.DRAINED

    def trigger(self) -> asyncio.Task[DrainReport]:
        """Start draining in the background (idempotent; safe from a signal handler)."""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run(), name="gateway-drain")
        return self._task

    async def drain(self) -> DrainReport:
        """Start (or join) the drain and wait for it to finish.

        A caller cancelled while waiting does not abort the drain itself.
        """
        return await asyncio.shield(self.trigger())

    def install(self, loop: SignalLoop, signals: Sequence[int] = DEFAULT_DRAIN_SIGNALS) -> None:
        """Start draining when any of ``signals`` arrives."""
        for sig in signals:
            loop.add_signal_handler(sig, self.trigger)
            self._signals.append(sig)

    def uninstall(self, loop: SignalLoop) -> None:
        """Remove the handlers registered by :meth:`install`."""
        while self._signals:
            loop.remove_signal_handler(self._signals.pop())

    async def _run(self) -> DrainReport:
        self._phase = DrainPhase.DRAINING
        self._server.begin_drain()
        started = self._clock()
        await self._wait_for_calls(until=started + self._drain_timeout_s)
        forced = 0
        if len(self._sessions):
            forced = await self._server.close_all(CloseCode.GOING_AWAY, "server shutting down")
            await self._wait_for_calls(until=self._clock() + _SHUTDOWN_GRACE_S)
        self._phase = DrainPhase.FLUSHING
        if self._flush is not None:
            await self._flush()
        if self._final_grace_s:
            await self._sleep(self._final_grace_s)
        self._phase = DrainPhase.DRAINED
        return DrainReport(
            waited_s=self._clock() - started, forced=forced, leftover=len(self._sessions)
        )

    async def _wait_for_calls(self, *, until: float) -> None:
        while len(self._sessions) and self._clock() < until:
            await self._sleep(self._poll_s)
