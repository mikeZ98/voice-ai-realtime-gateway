"""Test doubles for the duplex controller (transport ports and scheduling helpers)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from voice_ai_realtime_gateway.core.frames import AudioFrame


class RecordingEgress:
    """Egress port that records frames; ``gate`` (when set) paces every send."""

    def __init__(self, *, gated: bool = False) -> None:
        self.sent: list[AudioFrame] = []
        self.gate: asyncio.Event | None = asyncio.Event() if gated else None
        self.in_send = asyncio.Event()

    async def send(self, frame: AudioFrame, /) -> None:
        self.in_send.set()
        if self.gate is not None:
            await self.gate.wait()
        self.sent.append(frame)


class StubbornEgress:
    """Misbehaving port that swallows cancellation until released (contract violation)."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.in_send = asyncio.Event()
        self.swallowed = 0

    async def send(self, _frame: AudioFrame, /) -> None:
        self.in_send.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.swallowed += 1


class BrokenEgress:
    """Transport whose send always fails."""

    async def send(self, _frame: AudioFrame, /) -> None:
        msg = "peer connection reset"
        raise ConnectionResetError(msg)


async def until(predicate: Callable[[], bool], *, spins: int = 10_000) -> None:
    """Yield to the loop until ``predicate()`` holds (bounded, deterministic)."""
    for _ in range(spins):
        if predicate():
            return
        await asyncio.sleep(0)
    msg = "condition not reached"
    raise AssertionError(msg)


async def park() -> None:
    """A generation task that waits forever (until cancelled)."""
    await asyncio.Event().wait()


async def slow_cleanup(steps: int) -> None:
    """A generation task whose cancellation cleanup takes ``steps`` loop iterations."""
    try:
        await asyncio.Event().wait()
    finally:
        for _ in range(steps):
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
