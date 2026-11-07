"""Transport test doubles: deterministic sleeper, scripted connection, client helper."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

from voice_ai_realtime_gateway.transport.control import (
    ClientMessage,
    ServerMessage,
    decode_server_message,
    encode_control,
)
from voice_ai_realtime_gateway.transport.errors import TransportClosedError
from voice_ai_realtime_gateway.transport.framing import DecodedPacket, decode_packet, encode_packet

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket


class ManualSleeper:
    """Heartbeat sleeper that only wakes when the test says so."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self._waiters: deque[asyncio.Future[None]] = deque()

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        await waiter

    @property
    def sleeping(self) -> bool:
        return any(not w.done() for w in self._waiters)

    def release(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return
        msg = "nobody is sleeping"
        raise AssertionError(msg)


class ScriptedConnection:
    """Connection double: the test feeds inbound items and inspects outbound ones."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[str | bytes | TransportClosedError] = asyncio.Queue()
        self.sent: list[str | bytes] = []
        self.closes: list[tuple[int, str]] = []
        self.send_error: TransportClosedError | None = None

    async def recv(self) -> str | bytes:
        item = await self.inbound.get()
        if isinstance(item, TransportClosedError):
            raise item
        return item

    async def send(self, message: str | bytes, /) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closes.append((code, reason))


class Client:
    """The remote peer, speaking the gateway wire protocol over a memory socket."""

    def __init__(self, socket: MemoryWebSocket) -> None:
        self.socket = socket

    async def audio(self, frame: AudioFrame) -> None:
        await self.socket.send(encode_packet(frame))

    async def control(self, message: ClientMessage) -> None:
        await self.socket.send(encode_control(message))

    async def raw(self, message: str | bytes) -> None:
        await self.socket.send(message)

    async def receive(self) -> ServerMessage | DecodedPacket:
        message = await self.socket.recv()
        if isinstance(message, str):
            return decode_server_message(message)
        return decode_packet(message, received_at_ns=0)
