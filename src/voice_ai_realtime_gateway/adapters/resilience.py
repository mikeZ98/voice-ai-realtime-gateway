"""Upstream provider resilience: error taxonomy, backoff policy, connection state machine.

Provider state machine (one :class:`ProviderLink` per upstream stream)::

    IDLE ──open()──► CONNECTING ──ok──► STREAMING ──drop──► (reopen) ──► CONNECTING
                        │   ▲                                  │
              transient │   │ backoff elapsed                  │ fatal / budget spent
                        ▼   │                                  ▼
                      BACKOFF ── attempts exhausted ──────► FAILED
                                                   close() ─► CLOSED

* **Retryable:** transient network failures, provider 5xx / abnormal closes, rate limits
  (honouring ``retry_after``), and protocol garbage (often a single bad frame).
* **Fatal:** authentication and quota failures fail immediately; retrying cannot help.
  Exhausting ``max_attempts`` while connecting, or ``max_reconnects`` mid-stream, is
  also fatal.

The adapters convert a fatal :class:`UpstreamError` into a stage failure, which the
session supervisor turns into ``SessionTrigger.FAULT``. Every failure is reported as a
``ProviderError`` telemetry event.

Backoff: ``min(max_backoff, initial * multiplier ** (attempt - 1))``, scaled by a
jitter factor uniform in ``[1 - jitter, 1 + jitter)`` to avoid reconnect storms.
"""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from voice_ai_realtime_gateway.telemetry.events import ProviderError
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import WebSocketConnection

__all__ = [
    "GENERIC_CLOSE_KINDS",
    "Connector",
    "ErrorKind",
    "ProviderLink",
    "ProviderState",
    "RetryPolicy",
    "UpstreamError",
    "classify",
]

type Connector = Callable[[], Awaitable[WebSocketConnection]]
"""Opens one upstream connection; raises :class:`UpstreamError` for handshake refusals."""

type Sleeper = Callable[[float], Awaitable[None]]


@unique
class ErrorKind(StrEnum):
    """Why an upstream call failed."""

    AUTH = "auth"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    PROTOCOL = "protocol"


_FATAL: Final = frozenset({ErrorKind.AUTH, ErrorKind.QUOTA})


class UpstreamError(Exception):
    """A classified upstream failure."""

    def __init__(
        self,
        kind: ErrorKind,
        detail: str,
        *,
        provider: str = "upstream",
        retry_after_s: float | None = None,
        exhausted: bool = False,
    ) -> None:
        """Classify a failure; ``exhausted`` marks a retryable error that ran out of retries."""
        self.kind = kind
        self.detail = detail
        self.provider = provider
        self.retry_after_s = retry_after_s
        self.exhausted = exhausted
        super().__init__(f"{provider} {kind.value}: {detail}")

    @property
    def fatal(self) -> bool:
        """Whether retrying is pointless (auth, quota, or retries exhausted)."""
        return self.kind in _FATAL or self.exhausted


GENERIC_CLOSE_KINDS: Final[Mapping[int, ErrorKind]] = MappingProxyType(
    {
        CloseCode.PROTOCOL_ERROR: ErrorKind.PROTOCOL,
        CloseCode.UNSUPPORTED_DATA: ErrorKind.PROTOCOL,
        CloseCode.INVALID_PAYLOAD: ErrorKind.PROTOCOL,
        CloseCode.POLICY_VIOLATION: ErrorKind.PROTOCOL,
        CloseCode.MESSAGE_TOO_BIG: ErrorKind.PROTOCOL,
        CloseCode.TRY_AGAIN_LATER: ErrorKind.RATE_LIMIT,
    }
)
"""Close codes with a specific meaning; any other code is treated as transient."""


def classify(
    error: BaseException,
    *,
    provider: str,
    close_kinds: Mapping[int, ErrorKind] = GENERIC_CLOSE_KINDS,
) -> UpstreamError:
    """Map a failure to an :class:`UpstreamError` (already classified errors pass through)."""
    if isinstance(error, UpstreamError):
        return error
    if isinstance(error, TransportClosedError):
        kind = close_kinds.get(error.code, ErrorKind.TRANSIENT)
        return UpstreamError(kind, f"closed {error.code} {error.reason}".strip(), provider=provider)
    return UpstreamError(ErrorKind.TRANSIENT, f"{type(error).__name__}: {error}", provider=provider)


@unique
class ProviderState(StrEnum):
    """Lifecycle of one upstream stream."""

    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    BACKOFF = "backoff"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryPolicy:
    """Reconnection policy.

    Attributes:
        max_attempts: connection attempts per (re)connect before giving up.
        initial_backoff_s: delay after the first failed attempt.
        max_backoff_s: cap on any single delay.
        multiplier: exponential growth factor.
        jitter: relative jitter in ``[0, 1)``.
        max_reconnects: mid-stream reconnects allowed per stream.
    """

    max_attempts: int = 4
    initial_backoff_s: float = 0.05
    max_backoff_s: float = 2.0
    multiplier: float = 2.0
    jitter: float = 0.2
    max_reconnects: int = 3

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: on a non-positive count or delay, a multiplier below 1, or
                jitter outside ``[0, 1)``.
        """
        if self.max_attempts < 1 or self.max_reconnects < 0:
            msg = "max_attempts must be >= 1 and max_reconnects >= 0"
            raise ValueError(msg)
        delays = (self.initial_backoff_s, self.max_backoff_s, self.multiplier)
        if not all(math.isfinite(value) and value > 0 for value in delays):
            msg = "backoff delays and multiplier must be positive and finite"
            raise ValueError(msg)
        if self.multiplier < 1 or self.initial_backoff_s > self.max_backoff_s:
            msg = "need multiplier >= 1 and initial_backoff_s <= max_backoff_s"
            raise ValueError(msg)
        if not 0 <= self.jitter < 1:
            msg = f"jitter must be in [0, 1), got {self.jitter}"
            raise ValueError(msg)

    def backoff_s(self, attempt: int, unit: float) -> float:
        """Delay after failed ``attempt`` (1-based), given ``unit`` uniform in ``[0, 1)``."""
        base = min(self.max_backoff_s, self.initial_backoff_s * self.multiplier ** (attempt - 1))
        return base * (1 - self.jitter + 2 * self.jitter * unit)


_RETRYABLE_CONNECT_ERRORS: Final = (UpstreamError, TransportClosedError, OSError, TimeoutError)
_SYSTEM_RANDOM: Final = secrets.SystemRandom()


@final
class ProviderLink:
    """Connects (and reconnects) one upstream stream under a :class:`RetryPolicy`."""

    __slots__ = (
        "_clock",
        "_close_kinds",
        "_connector",
        "_jitter",
        "_policy",
        "_provider",
        "_reconnects",
        "_sink",
        "_sleep",
        "_state",
        "_trace",
        "attempts",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        *,
        provider: str,
        connector: Connector,
        sink: TelemetrySink,
        trace: TraceContext,
        policy: RetryPolicy | None = None,
        close_kinds: Mapping[int, ErrorKind] = GENERIC_CLOSE_KINDS,
        clock: MonotonicClock = time.monotonic_ns,
        sleep: Sleeper = asyncio.sleep,
        jitter: Callable[[], float] = _SYSTEM_RANDOM.random,
    ) -> None:
        """Create an ``IDLE`` link."""
        self._provider = provider
        self._connector = connector
        self._sink = sink
        self._trace = trace
        self._policy = RetryPolicy() if policy is None else policy
        self._close_kinds = close_kinds
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._state = ProviderState.IDLE
        self._reconnects = 0
        self.attempts = 0

    @property
    def state(self) -> ProviderState:
        """Current lifecycle state."""
        return self._state

    @property
    def reconnects(self) -> int:
        """Mid-stream reconnects performed so far."""
        return self._reconnects

    @override
    def __repr__(self) -> str:
        return f"ProviderLink(provider={self._provider}, state={self._state.value})"

    def classify(self, error: BaseException) -> UpstreamError:
        """Classify ``error`` with this provider's close-code mapping."""
        return classify(error, provider=self._provider, close_kinds=self._close_kinds)

    async def open(self) -> WebSocketConnection:
        """Establish the stream, retrying transient failures.

        Raises:
            UpstreamError: fatal failure, or retries exhausted (``exhausted=True``).
        """
        return await self._connect()

    async def reopen(self, cause: BaseException) -> WebSocketConnection:
        """Re-establish a stream that dropped with ``cause``.

        Raises:
            UpstreamError: if ``cause`` is fatal, the reconnect budget is spent, or
                reconnecting fails.
        """
        error = self.classify(cause)
        if error.fatal or self._reconnects >= self._policy.max_reconnects:
            final_error = (
                error
                if error.fatal
                else UpstreamError(
                    error.kind,
                    f"reconnect budget spent: {error.detail}",
                    provider=self._provider,
                    exhausted=True,
                )
            )
            self._report(final_error, attempt=self.attempts)
            self._state = ProviderState.FAILED
            raise final_error from cause
        self._report(error, attempt=self.attempts)
        self._reconnects += 1
        return await self._connect()

    def fail(self, cause: BaseException) -> UpstreamError:
        """Mark the stream failed without retrying; returns the classified error."""
        error = self.classify(cause)
        self._report(error, attempt=self.attempts, fatal=True)
        self._state = ProviderState.FAILED
        return error

    async def close(self, connection: WebSocketConnection | None) -> None:
        """Close ``connection`` (best effort) and mark the link ``CLOSED``."""
        if connection is not None:
            await connection.close()
        if self._state is not ProviderState.FAILED:
            self._state = ProviderState.CLOSED

    async def _connect(self) -> WebSocketConnection:
        policy = self._policy
        attempt = 0
        while True:
            attempt += 1
            self._state = ProviderState.CONNECTING
            self.attempts += 1
            try:
                connection = await self._connector()
            except _RETRYABLE_CONNECT_ERRORS as exc:
                error = self.classify(exc)
                if error.fatal:
                    self._report(error, attempt=attempt)
                    self._state = ProviderState.FAILED
                    raise error from exc
                if attempt == policy.max_attempts:
                    exhausted = UpstreamError(
                        error.kind,
                        f"gave up after {attempt} attempts: {error.detail}",
                        provider=self._provider,
                        exhausted=True,
                    )
                    self._report(exhausted, attempt=attempt)
                    self._state = ProviderState.FAILED
                    raise exhausted from exc
                self._report(error, attempt=attempt)
                self._state = ProviderState.BACKOFF
                delay = policy.backoff_s(attempt, self._jitter())
                await self._sleep(max(delay, error.retry_after_s or 0.0))
            else:
                self._state = ProviderState.STREAMING
                return connection

    def _report(self, error: UpstreamError, *, attempt: int, fatal: bool | None = None) -> None:
        self._sink.emit(
            ProviderError(
                trace=self._trace,
                at_ns=self._clock(),
                provider=self._provider,
                kind=error.kind,
                fatal=error.fatal if fatal is None else fatal,
                attempt=attempt,
                detail=error.detail,
            )
        )
