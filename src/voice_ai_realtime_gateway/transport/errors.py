"""Transport error taxonomy and RFC 6455 close codes."""

from __future__ import annotations

from enum import IntEnum, unique
from typing import Final

__all__ = [
    "MAX_CLOSE_REASON_BYTES",
    "CloseCode",
    "ProtocolError",
    "TransportClosedError",
    "TransportError",
    "truncate_reason",
]

MAX_CLOSE_REASON_BYTES: Final = 123
"""A close frame's reason must fit in 123 UTF-8 bytes (RFC 6455, section 5.5)."""


@unique
class CloseCode(IntEnum):
    """WebSocket close status codes used by the gateway (RFC 6455, section 7.4.1)."""

    NORMAL = 1000
    GOING_AWAY = 1001
    PROTOCOL_ERROR = 1002
    UNSUPPORTED_DATA = 1003
    ABNORMAL = 1006
    INVALID_PAYLOAD = 1007
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    INTERNAL_ERROR = 1011
    TRY_AGAIN_LATER = 1013


def truncate_reason(reason: str) -> str:
    """Clip ``reason`` to the close-frame limit without splitting a UTF-8 sequence."""
    encoded = reason.encode()
    if len(encoded) <= MAX_CLOSE_REASON_BYTES:
        return reason
    return encoded[:MAX_CLOSE_REASON_BYTES].decode(errors="ignore")


class TransportError(Exception):
    """Base class for transport failures."""


class ProtocolError(TransportError):
    """The peer violated the wire protocol; the connection must close with ``code``."""

    def __init__(self, code: CloseCode, reason: str) -> None:
        """Record the close code and a close-frame-safe reason."""
        self.code = code
        self.reason = truncate_reason(reason)
        super().__init__(f"{code.name} ({code.value}): {self.reason}")


class TransportClosedError(TransportError):
    """The connection is closed; carries the close code and reason that ended it."""

    def __init__(self, code: int, reason: str = "") -> None:
        """Record how the connection ended."""
        self.code = code
        self.reason = reason
        super().__init__(f"connection closed ({code}): {reason}")
