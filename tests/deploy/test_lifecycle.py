"""Graceful drain state machine, on a deterministic clock."""

from __future__ import annotations

import asyncio
import signal

import pytest

from duplex_support import until
from voice_ai_realtime_gateway.deploy.lifecycle import DrainController, DrainPhase
from voice_ai_realtime_gateway.production.observability import SessionRegistry
from voice_ai_realtime_gateway.telemetry.ids import TraceContext

pytestmark = pytest.mark.usefixtures("strict_loop")


class Call:
    def __init__(self, sessions: SessionRegistry, *, obeys_close: bool = True) -> None:
        self.sessions = sessions
        self.obeys_close = obeys_close
        self.trace = TraceContext.new()
        self.started_ns = 0
        self.deadline_ns = 0
        self.closed_with: tuple[int, str] | None = None

    def ring_bytes(self) -> int:
        return 0

    async def terminate(self, code: int, reason: str, /) -> None:
        self.closed_with = (code, reason)
        if self.obeys_close:
            self.sessions.discard(self)

    async def quarantine(self) -> None:
        return None


class Server:
    def __init__(self, sessions: SessionRegistry) -> None:
        self.sessions = sessions
        self.draining = False

    def begin_drain(self) -> None:
        self.draining = True

    async def close_all(self, code: int, reason: str) -> int:
        calls = self.sessions.calls()
        for call in calls:
            await call.terminate(code, reason)
        return len(calls)


class Clock:
    """Virtual time advanced only by the controller's sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.on_sleep: list[object] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


def controller(
    sessions: SessionRegistry, clock: Clock, **kwargs: object
) -> tuple[DrainController, Server]:
    server = Server(sessions)
    drain = DrainController(
        server=server,
        sessions=sessions,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=1.0,
        **kwargs,  # type: ignore[arg-type]
    )
    return drain, server


async def test_drain_waits_for_live_calls_then_flushes() -> None:
    sessions, clock = SessionRegistry(), Clock()
    call = Call(sessions)
    sessions.add(call)
    flushed: list[float] = []

    async def flush() -> None:
        flushed.append(clock.now)

    drain, server = controller(
        sessions, clock, drain_timeout_s=30, final_scrape_grace_s=15, flush=flush
    )
    assert (drain.phase, drain.drained) == (DrainPhase.SERVING, False)
    task = asyncio.create_task(drain.drain())
    await until(lambda: server.draining)  # phase 1: admissions stop immediately
    assert drain.phase is DrainPhase.DRAINING
    for _ in range(3):
        await asyncio.sleep(0)
    sessions.discard(call)  # the call ends on its own after ~3 s
    report = await task
    assert (report.forced, report.leftover) == (0, 0)
    assert call.closed_with is None, "a call that finished in time is never cut"
    assert drain.drained
    assert flushed
    assert clock.sleeps[-1] == 15, "final scrape grace after the flush"


async def test_calls_still_live_at_the_deadline_are_closed_going_away() -> None:
    sessions, clock = SessionRegistry(), Clock()
    calls = [Call(sessions), Call(sessions)]
    for call in calls:
        sessions.add(call)
    drain, _ = controller(sessions, clock, drain_timeout_s=5)
    report = await drain.drain()
    assert (report.forced, report.leftover) == (2, 0)
    assert all(call.closed_with == (1001, "server shutting down") for call in calls)
    assert report.waited_s == 5.0


async def test_a_call_ignoring_the_close_is_reported_as_leftover() -> None:
    sessions, clock = SessionRegistry(), Clock()
    sessions.add(Call(sessions, obeys_close=False))
    drain, _ = controller(sessions, clock, drain_timeout_s=2)
    report = await drain.drain()
    assert (report.forced, report.leftover) == (1, 1)
    assert report.waited_s == 7.0, "2 s budget + 5 s shutdown grace"
    assert drain.drained


async def test_drain_is_idempotent_and_survives_a_cancelled_waiter() -> None:
    sessions, clock = SessionRegistry(), Clock()
    sessions.add(Call(sessions))  # keeps the drain running while the waiter is cancelled
    drain, _ = controller(sessions, clock, drain_timeout_s=3)
    first = drain.trigger()
    assert drain.trigger() is first
    waiter = asyncio.create_task(drain.drain())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert (await drain.drain()).forced == 1, "the drain itself kept going"


class Loop:
    def __init__(self) -> None:
        self.handlers: dict[int, object] = {}

    def add_signal_handler(self, sig: int, callback: object, /) -> None:
        self.handlers[sig] = callback

    def remove_signal_handler(self, sig: int, /) -> bool:
        return self.handlers.pop(sig, None) is not None


async def test_signal_handlers_trigger_the_drain() -> None:
    sessions, clock = SessionRegistry(), Clock()
    drain, server = controller(sessions, clock)
    loop = Loop()
    drain.install(loop, (signal.SIGUSR1, signal.SIGTERM))
    assert set(loop.handlers) == {signal.SIGUSR1, signal.SIGTERM}
    loop.handlers[signal.SIGUSR1]()  # type: ignore[operator]
    await drain.drain()
    assert server.draining
    drain.uninstall(loop)
    assert loop.handlers == {}


@pytest.mark.parametrize(
    "kwargs", [{"drain_timeout_s": -1}, {"final_scrape_grace_s": -1}, {"poll_interval_s": 0}]
)
async def test_configuration_is_validated(kwargs: dict[str, float]) -> None:
    sessions = SessionRegistry()
    with pytest.raises(ValueError, match="durations"):
        DrainController(server=Server(sessions), sessions=sessions, **kwargs)  # type: ignore[arg-type]
