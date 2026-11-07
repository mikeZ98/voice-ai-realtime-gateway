"""Simulated upstream providers over in-memory sockets (no network)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from voice_ai_realtime_gateway.transport.errors import TransportClosedError
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from voice_ai_realtime_gateway.transport.base import WebSocketConnection

type Handler = Callable[[MemoryWebSocket, int], Awaitable[None]]


class FakeProvider:
    """Connector that pairs each connection with a scripted server handler.

    ``refusals`` are raised by successive connection attempts before any succeed.
    """

    def __init__(self, handler: Handler, *refusals: BaseException) -> None:
        self.handler = handler
        self.refusals = list(refusals)
        self.connections = 0
        self.attempts = 0
        self.tasks: list[asyncio.Task[None]] = []
        self.servers: list[MemoryWebSocket] = []

    async def __call__(self) -> WebSocketConnection:
        self.attempts += 1
        if self.refusals:
            raise self.refusals.pop(0)
        client, server = MemoryWebSocket.pair(capacity=256)
        index = self.connections
        self.connections += 1
        self.servers.append(server)
        self.tasks.append(asyncio.create_task(self._run(server, index)))
        return client

    async def _run(self, server: MemoryWebSocket, index: int) -> None:
        with contextlib.suppress(TransportClosedError):
            await self.handler(server, index)

    async def finish(self) -> None:
        """Close every server end and reap the handlers (test teardown)."""
        for server in self.servers:
            await server.close()
        for task in self.tasks:
            await task


async def instant_sleep(_delay: float) -> None:
    """Backoff sleeper that only yields (keeps chaos tests fast and deterministic)."""
    await asyncio.sleep(0)
