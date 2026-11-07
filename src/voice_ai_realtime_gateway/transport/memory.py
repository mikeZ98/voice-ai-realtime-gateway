"""In-memory WebSocket connection pairs (loopback transport for tests and local tools).

Each direction is a bounded message buffer, so a slow reader really does suspend the
writer, exactly like TCP flow control behind a WebSocket. Closing follows WebSocket
semantics: messages already sent are still delivered to the peer before it observes
the close, while the closing side discards its own unread input. :meth:`sever`
simulates network loss: both sides observe code ``1006`` and in-flight data is lost.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Final, Self, final, override

from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError

__all__ = ["DEFAULT_MEMORY_CAPACITY", "MemoryWebSocket"]

DEFAULT_MEMORY_CAPACITY: Final = 64


@final
class _Inbox:
    __slots__ = ("_capacity", "_closed", "_condition", "_items")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: deque[str | bytes] = deque()
        self._closed: tuple[int, str] | None = None
        self._condition = asyncio.Condition()

    def __len__(self) -> int:
        return len(self._items)

    async def get(self) -> str | bytes:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._items) or self._closed is not None)
            if self._items:
                self._condition.notify_all()
                return self._items.popleft()
            code, reason = self._closed or (CloseCode.ABNORMAL, "")  # predicate: closed here
            raise TransportClosedError(code, reason)

    async def put(self, item: str | bytes) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self._items) < self._capacity or self._closed is not None
            )
            if self._closed is not None:
                raise TransportClosedError(*self._closed)
            self._items.append(item)
            self._condition.notify_all()

    async def close(self, code: int, reason: str, *, discard: bool) -> None:
        async with self._condition:
            if self._closed is None:
                self._closed = (code, reason)
            if discard:
                self._items.clear()
            self._condition.notify_all()


@final
class MemoryWebSocket:
    """One end of an in-memory WebSocket connection. Create ends with :meth:`pair`."""

    __slots__ = ("_closed", "_inbox", "_peer")

    def __init__(self, inbox: _Inbox) -> None:
        """Use :meth:`pair` instead; an unpaired end is unusable."""
        self._inbox = inbox
        self._closed: tuple[int, str] | None = None
        self._peer: MemoryWebSocket

    @classmethod
    def pair(cls, *, capacity: int = DEFAULT_MEMORY_CAPACITY) -> tuple[Self, Self]:
        """Two connected ends; each direction buffers at most ``capacity`` messages.

        Raises:
            ValueError: if ``capacity`` is not positive.
        """
        if capacity < 1:
            msg = f"capacity must be positive, got {capacity}"
            raise ValueError(msg)
        left, right = cls(_Inbox(capacity)), cls(_Inbox(capacity))
        left._peer, right._peer = right, left
        return left, right

    @property
    def closed(self) -> bool:
        """Whether this end observed or initiated a close."""
        return self._closed is not None

    @property
    def close_code(self) -> int | None:
        """Close code, once closed."""
        return None if self._closed is None else self._closed[0]

    @property
    def close_reason(self) -> str:
        """Close reason, once closed."""
        return "" if self._closed is None else self._closed[1]

    @property
    def pending(self) -> int:
        """Messages received but not yet read on this end."""
        return len(self._inbox)

    async def recv(self) -> str | bytes:
        """Next message from the peer.

        Raises:
            TransportClosedError: once closed and drained.
        """
        return await self._inbox.get()

    async def send(self, message: str | bytes, /) -> None:
        """Queue ``message`` for the peer, suspending while its buffer is full.

        Raises:
            TransportClosedError: if either side closed.
        """
        if self._closed is not None:
            raise TransportClosedError(*self._closed)
        await self._peer._inbox.put(message)  # noqa: SLF001 -- same class, other end

    async def close(self, code: int = CloseCode.NORMAL, reason: str = "") -> None:
        """Close this end; the peer drains what was already sent, then sees the close."""
        if self._closed is not None:
            return
        code = int(code)
        self._closed = (code, reason)
        await self._inbox.close(code, reason, discard=True)
        await self._peer._observe_close(code, reason)  # noqa: SLF001 -- same class, other end

    async def sever(self) -> None:
        """Simulate network loss: both ends close abnormally (1006) and lose buffers."""
        abnormal = int(CloseCode.ABNORMAL)
        for end in (self, self._peer):
            if end._closed is None:  # noqa: SLF001 -- same class
                end._closed = (abnormal, "")  # noqa: SLF001 -- same class
            await end._inbox.close(abnormal, "", discard=True)  # noqa: SLF001

    async def _observe_close(self, code: int, reason: str) -> None:
        # Only reached from the peer's first close(); every path that closes this end
        # (close, sever) also closes the peer, so this end is necessarily still open.
        self._closed = (code, reason)
        await self._inbox.close(code, reason, discard=False)

    @override
    def __repr__(self) -> str:
        return f"MemoryWebSocket(closed={self._closed}, pending={len(self._inbox)})"
