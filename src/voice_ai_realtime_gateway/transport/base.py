"""Protocol-agnostic transport ports.

``WebSocketConnection`` is the only thing a concrete WebSocket library must provide. Its
shape matches the ``websockets`` asyncio API (``recv`` / ``send`` / ``close``); a shim
of a few lines translates that library's ``ConnectionClosed`` into
:class:`~.errors.TransportClosedError`. Everything above it -- framing, multiplexing,
backpressure, heartbeat -- is ours and is tested against in-memory connections.
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.transport.control import ApplicationMessage, ServerMessage

__all__ = [
    "AudioIngress",
    "CloseInitiator",
    "ControlChannel",
    "VoiceTransport",
    "WebSocketConnection",
]


@unique
class CloseInitiator(StrEnum):
    """Which side ended the connection."""

    LOCAL = "local"
    REMOTE = "remote"


class WebSocketConnection(Protocol):
    """Message-oriented, bidirectional connection (text = control, binary = audio).

    Contract:
        * ``recv`` returns the next message; once the connection is closed (by either
          side, or by network loss) it raises ``TransportClosedError`` carrying the
          close code and reason. Pending ``recv`` calls are woken by ``close``.
        * ``send`` may suspend for flow control; on a closed connection it raises
          ``TransportClosedError``.
        * ``close`` is idempotent and never raises.
    """

    async def recv(self) -> str | bytes:
        """Receive one message."""

    async def send(self, message: str | bytes, /) -> None:
        """Send one message."""

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Start (or complete) the closing handshake."""


class AudioIngress(Protocol):
    """Stream of client audio frames; ends when the transport closes."""

    def __aiter__(self) -> AsyncIterator[AudioFrame]:
        """Iterate over ingress frames in arrival order."""


class ControlChannel(Protocol):
    """Typed control signalling: client application messages in, server messages out."""

    def __aiter__(self) -> AsyncIterator[ApplicationMessage]:
        """Iterate over client application messages; ends when the transport closes."""

    async def send(self, message: ServerMessage, /) -> bool:
        """Send a server message; returns whether it was handed to the connection."""


class VoiceTransport(Protocol):
    """A client connection carrying multiplexed audio and control.

    It also satisfies the domain's ``AudioEgress`` port through :meth:`send`.
    """

    @property
    def trace(self) -> TraceContext:
        """Session correlation context negotiated at connection time."""

    @property
    def ingress(self) -> AudioIngress:
        """Client audio."""

    @property
    def control(self) -> ControlChannel:
        """Client application control messages and server control output."""

    @property
    def closed(self) -> bool:
        """Whether the connection is closed."""

    @property
    def close_code(self) -> int | None:
        """Close code once closed."""

    @property
    def close_reason(self) -> str:
        """Close reason once closed (empty before)."""

    @property
    def closed_by(self) -> CloseInitiator | None:
        """Which side closed, once closed."""

    async def send(self, frame: AudioFrame, /) -> None:
        """Deliver one agent audio frame (best-effort once the peer is gone)."""

    async def run(self) -> None:
        """Pump the connection until it closes."""

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection. Idempotent."""
