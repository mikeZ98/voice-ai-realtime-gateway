"""Trace correlation identifiers carried by every audio packet and state transition.

Wire format (hex, lowercase, fixed width -- mirrors W3C Trace Context sizing):

* ``X-Session-ID``    -- 128-bit session identifier, 32 hex chars.
* ``X-Audio-Span-ID`` -- 64-bit audio span identifier, 16 hex chars.

All-zero identifiers are reserved as invalid, exactly like ``traceparent``.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NewType, Self

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "AUDIO_SPAN_ID_HEADER",
    "SESSION_ID_HEADER",
    "AudioSpanId",
    "InvalidTraceIdError",
    "SessionId",
    "TraceContext",
    "new_session_id",
    "new_span_id",
    "parse_session_id",
    "parse_span_id",
]

SESSION_ID_HEADER: Final = "X-Session-ID"
AUDIO_SPAN_ID_HEADER: Final = "X-Audio-Span-ID"

SessionId = NewType("SessionId", str)
AudioSpanId = NewType("AudioSpanId", str)

_SESSION_ID_BYTES: Final = 16
_SPAN_ID_BYTES: Final = 8
_SESSION_ID_RE: Final = re.compile(rf"[0-9a-f]{{{_SESSION_ID_BYTES * 2}}}")
_SPAN_ID_RE: Final = re.compile(rf"[0-9a-f]{{{_SPAN_ID_BYTES * 2}}}")


class InvalidTraceIdError(ValueError):
    """Raised when a correlation identifier violates the wire format."""


def _parse(raw: str, pattern: re.Pattern[str], kind: str) -> str:
    if pattern.fullmatch(raw) is None:
        msg = f"{kind} must be {pattern.pattern!r} lowercase hex, got {raw!r}"
        raise InvalidTraceIdError(msg)
    if raw.count("0") == len(raw):
        msg = f"{kind} must not be all zeros"
        raise InvalidTraceIdError(msg)
    return raw


def parse_session_id(raw: str) -> SessionId:
    """Validate ``raw`` as a session identifier.

    Raises:
        InvalidTraceIdError: if ``raw`` is not 32 lowercase hex chars or is all zeros.
    """
    return SessionId(_parse(raw, _SESSION_ID_RE, SESSION_ID_HEADER))


def parse_span_id(raw: str) -> AudioSpanId:
    """Validate ``raw`` as an audio span identifier.

    Raises:
        InvalidTraceIdError: if ``raw`` is not 16 lowercase hex chars or is all zeros.
    """
    return AudioSpanId(_parse(raw, _SPAN_ID_RE, AUDIO_SPAN_ID_HEADER))


def _random_hex(nbytes: int) -> str:
    while True:
        candidate = secrets.token_hex(nbytes)
        if candidate.count("0") != len(candidate):
            return candidate


def new_session_id() -> SessionId:
    """Generate a cryptographically random, non-zero session identifier."""
    return SessionId(_random_hex(_SESSION_ID_BYTES))


def new_span_id() -> AudioSpanId:
    """Generate a cryptographically random, non-zero audio span identifier."""
    return AudioSpanId(_random_hex(_SPAN_ID_BYTES))


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Immutable ``(session, span)`` correlation pair.

    A *span* is one contiguous audio stream inside a session (a user utterance or
    one agent response). Every packet and every state transition carries one.
    """

    session_id: SessionId
    span_id: AudioSpanId

    @classmethod
    def new(cls) -> Self:
        """Start a brand-new session with its first span."""
        return cls(new_session_id(), new_span_id())

    def child(self) -> Self:
        """Open a fresh span inside the same session."""
        return type(self)(self.session_id, new_span_id())

    def to_headers(self) -> dict[str, str]:
        """Render as transport headers."""
        return {SESSION_ID_HEADER: self.session_id, AUDIO_SPAN_ID_HEADER: self.span_id}

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> Self:
        """Parse from transport headers (header names are case-insensitive).

        Raises:
            InvalidTraceIdError: if a header is missing, duplicated or malformed.
        """
        folded: dict[str, str] = {}
        for name, value in headers.items():
            key = name.lower()
            if key in folded:
                msg = f"duplicate correlation header {name!r}"
                raise InvalidTraceIdError(msg)
            folded[key] = value
        try:
            session_raw = folded[SESSION_ID_HEADER.lower()]
            span_raw = folded[AUDIO_SPAN_ID_HEADER.lower()]
        except KeyError as exc:
            msg = f"missing correlation header {exc.args[0]!r}"
            raise InvalidTraceIdError(msg) from exc
        return cls(parse_session_id(session_raw), parse_span_id(span_raw))
