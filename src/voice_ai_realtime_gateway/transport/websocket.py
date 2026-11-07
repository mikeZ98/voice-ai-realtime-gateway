"""Asyncio-native WebSocket voice transport.

Multiplexing: binary messages carry framed audio packets (:mod:`.framing`); text
messages carry typed JSON control (:mod:`.control`).

Ingress path (single reader task)::

    recv() -> size check -> decode (zero-copy) -> session / format / sequence checks
           -> ingress AudioRingBuffer -> IngressStream consumer (VAD, ASR, ...)

* **Bounded ingress.** The ring bounds frames and bytes. ``DROP_OLDEST`` (the default)
  keeps the newest audio when downstream is slow; ``BACKPRESSURE`` stops reading the
  socket instead, so the client's TCP window closes. Either way a client cannot drive
  gateway memory beyond the ring budget plus one message.
* **Sequence hygiene.** Per span, gaps are counted as network loss (``IngressGap``) and
  late or duplicate packets are dropped. Realtime audio is never replayed out of order.
* **Isolation.** A packet whose ``X-Session-ID`` differs from the connection's is a
  policy violation (``1008``), which blocks cross-session audio injection.
* **Heartbeat.** Every ``interval_s`` the transport sends a ``ping``; any inbound
  message counts as liveness. Silence longer than ``timeout_s`` closes with ``1001``.
  While the reader is deliberately stalled by backpressure, the stall is ours, not the
  peer's, so dead-peer detection is suspended.
* **Abuse guards.** Text and binary messages above their size limits, and control
  messages above the per-connection rate (token bucket: ping floods, slow-drip
  control spam), close the connection immediately with ``1008`` (policy violation).
* **Protocol violations** close the connection with the matching RFC 6455 code.

Egress is best-effort once the peer is gone: frames sent after close are counted and
discarded, so a disconnect ends a session through ``CLOSE`` rather than ``FAULT``.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self, final, override

from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.core.ratelimit import TokenBucket
from voice_ai_realtime_gateway.core.ring import AudioRingBuffer, OverflowPolicy, RingClosedError
from voice_ai_realtime_gateway.telemetry.events import EgressDropped, IngressGap, TransportClosed
from voice_ai_realtime_gateway.transport.base import CloseInitiator
from voice_ai_realtime_gateway.transport.control import (
    Ping,
    Pong,
    SessionConfig,
    decode_client_message,
    encode_control,
)
from voice_ai_realtime_gateway.transport.errors import (
    CloseCode,
    ProtocolError,
    TransportClosedError,
    truncate_reason,
)
from voice_ai_realtime_gateway.transport.framing import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    HEADER_SIZE,
    decode_packet,
    encode_packet,
)

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.telemetry.ids import AudioSpanId, TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import WebSocketConnection
    from voice_ai_realtime_gateway.transport.control import (
        ApplicationMessage,
        AudioFormat,
        ServerMessage,
    )

__all__ = [
    "DEFAULT_INGRESS_RING_BYTES",
    "DEFAULT_INGRESS_RING_FRAMES",
    "ControlStream",
    "HeartbeatConfig",
    "IngressStream",
    "Sleeper",
    "TransportLimits",
    "TransportStats",
    "WebSocketVoiceTransport",
]

DEFAULT_INGRESS_RING_FRAMES: Final = 64
"""1.28 s of 20 ms frames."""
DEFAULT_INGRESS_RING_BYTES: Final = 256 * 1024

_NS_PER_S: Final = 1_000_000_000

type Sleeper = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True, kw_only=True)
class HeartbeatConfig:
    """Dead-connection detection.

    Attributes:
        interval_s: period between pings (and between liveness checks).
        timeout_s: inbound silence after which the peer is declared dead. Detection
            completes within ``timeout_s + interval_s``.
    """

    interval_s: float = 5.0
    timeout_s: float = 15.0

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: unless ``0 < interval_s <= timeout_s`` and both are finite.
        """
        if not (math.isfinite(self.interval_s) and math.isfinite(self.timeout_s)):
            msg = "heartbeat durations must be finite"
            raise ValueError(msg)
        if not 0 < self.interval_s <= self.timeout_s:
            msg = f"need 0 < interval_s <= timeout_s, got {self.interval_s}, {self.timeout_s}"
            raise ValueError(msg)

    @property
    def timeout_ns(self) -> int:
        """``timeout_s`` in nanoseconds."""
        return int(self.timeout_s * _NS_PER_S)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransportLimits:
    """Per-connection resource bounds.

    Attributes:
        max_payload_bytes: largest audio payload; bigger binary messages close with 1008.
        max_control_chars: largest control message; bigger text messages close with 1008.
        max_pending_control: control messages the consumer may lag behind.
        control_rate_per_s: sustained control messages per second (pings included).
        control_burst: control messages allowed in a burst.
    """

    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_control_chars: int = 4096
    max_pending_control: int = 32
    control_rate_per_s: float = 20.0
    control_burst: int = 40

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: if any limit is not positive.
        """
        for name in (
            "max_payload_bytes",
            "max_control_chars",
            "max_pending_control",
            "control_rate_per_s",
            "control_burst",
        ):
            if getattr(self, name) < 1:
                msg = f"{name} must be positive"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransportStats:
    """Lifetime counters of one connection."""

    frames_in: int
    bytes_in: int
    frames_out: int
    bytes_out: int
    egress_dropped: int
    late_dropped: int
    lost_packets: int
    control_in: int
    control_out: int


@final
class IngressStream:
    """Async iterator over ingress frames; ends when the transport closes."""

    __slots__ = ("_ring",)

    def __init__(self, ring: AudioRingBuffer) -> None:
        """Consume from ``ring``."""
        self._ring = ring

    def __aiter__(self) -> Self:
        """Return the iterator itself."""
        return self

    async def __anext__(self) -> AudioFrame:
        """Next frame in arrival order."""
        frame = await self._ring.get()
        if frame is None:
            raise StopAsyncIteration
        return frame


@final
class ControlStream:
    """Bounded queue of client application messages plus the server control sender."""

    __slots__ = ("_closed", "_ended", "_limit", "_queue", "_sender")

    def __init__(self, sender: Callable[[ServerMessage], Awaitable[bool]], limit: int) -> None:
        """Hold at most ``limit`` undelivered messages; send through ``sender``."""
        self._sender = sender
        self._limit = limit
        self._queue: asyncio.Queue[ApplicationMessage | None] = asyncio.Queue()
        self._closed = False
        self._ended = False

    def __len__(self) -> int:
        """Messages waiting for the consumer."""
        return self._queue.qsize()

    def __aiter__(self) -> Self:
        """Return the iterator itself."""
        return self

    async def __anext__(self) -> ApplicationMessage:
        """Next client application message."""
        if self._ended:
            raise StopAsyncIteration
        message = await self._queue.get()
        if message is None:
            self._ended = True
            raise StopAsyncIteration
        return message

    async def send(self, message: ServerMessage, /) -> bool:
        """Send a server control message; ``False`` if the connection is gone."""
        return await self._sender(message)

    def push(self, message: ApplicationMessage) -> None:
        """Enqueue a client message (transport-internal).

        Raises:
            ProtocolError: ``POLICY_VIOLATION`` when the consumer is too far behind.
        """
        if self._queue.qsize() >= self._limit:
            raise ProtocolError(CloseCode.POLICY_VIOLATION, "control message backlog exceeded")
        self._queue.put_nowait(message)

    def end(self) -> None:
        """Signal end-of-stream to the consumer (transport-internal). Idempotent."""
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)


@final
class WebSocketVoiceTransport:
    """Voice transport over a :class:`~.base.WebSocketConnection`."""

    __slots__ = (
        "_accepted",
        "_clock",
        "_close_code",
        "_close_reason",
        "_closed_by",
        "_connection",
        "_control",
        "_control_bucket",
        "_counters",
        "_format",
        "_heartbeat",
        "_ingress",
        "_ingress_span",
        "_last_seen",
        "_limits",
        "_next_sequence",
        "_reader_blocked",
        "_ring",
        "_sink",
        "_sleep",
        "_started",
        "_trace",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        connection: WebSocketConnection,
        *,
        trace: TraceContext,
        sink: TelemetrySink,
        clock: MonotonicClock = time.monotonic_ns,
        ingress_ring: AudioRingBuffer | None = None,
        accepted_codecs: frozenset[AudioCodec] = frozenset(AudioCodec),
        heartbeat: HeartbeatConfig | None = None,
        limits: TransportLimits | None = None,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        """Wrap ``connection`` for the session identified by ``trace``.

        ``heartbeat=None`` uses :class:`HeartbeatConfig` defaults; pass a config with a
        very long interval to effectively disable it.

        Raises:
            ValueError: if ``accepted_codecs`` is empty.
        """
        if not accepted_codecs:
            msg = "accepted_codecs must not be empty"
            raise ValueError(msg)
        self._connection = connection
        self._trace = trace
        self._sink = sink
        self._clock = clock
        self._sleep = sleep
        self._accepted = accepted_codecs
        self._heartbeat = HeartbeatConfig() if heartbeat is None else heartbeat
        self._limits = TransportLimits() if limits is None else limits
        self._ring = (
            AudioRingBuffer(
                capacity_frames=DEFAULT_INGRESS_RING_FRAMES,
                capacity_bytes=DEFAULT_INGRESS_RING_BYTES,
                policy=OverflowPolicy.DROP_OLDEST,
                sink=sink,
            )
            if ingress_ring is None
            else ingress_ring
        )
        self._ingress = IngressStream(self._ring)
        self._control = ControlStream(self.send_control, self._limits.max_pending_control)
        self._control_bucket = TokenBucket(
            rate_per_s=self._limits.control_rate_per_s,
            burst=self._limits.control_burst,
            clock=clock,
        )
        self._format: AudioFormat | None = None
        self._ingress_span: AudioSpanId | None = None
        self._next_sequence = 0
        self._last_seen = clock()
        self._reader_blocked = False
        self._started = False
        self._close_code: int | None = None
        self._close_reason = ""
        self._closed_by: CloseInitiator | None = None
        self._counters = dict.fromkeys(
            (
                "frames_in",
                "bytes_in",
                "frames_out",
                "bytes_out",
                "egress_dropped",
                "late_dropped",
                "lost_packets",
                "control_in",
                "control_out",
            ),
            0,
        )

    # ------------------------------------------------------------------ introspection

    @property
    def trace(self) -> TraceContext:
        """Session correlation context of this connection."""
        return self._trace

    @property
    def ingress(self) -> IngressStream:
        """Client audio frames."""
        return self._ingress

    @property
    def control(self) -> ControlStream:
        """Client application messages in, server control out."""
        return self._control

    @property
    def ingress_ring(self) -> AudioRingBuffer:
        """The bounded ingress buffer."""
        return self._ring

    @property
    def audio_format(self) -> AudioFormat | None:
        """Format declared by the client's ``session.config``, if any."""
        return self._format

    @property
    def closed(self) -> bool:
        """Whether the connection is closed."""
        return self._close_code is not None

    @property
    def close_code(self) -> int | None:
        """Close code, once closed."""
        return self._close_code

    @property
    def close_reason(self) -> str:
        """Close reason, once closed."""
        return self._close_reason

    @property
    def closed_by(self) -> CloseInitiator | None:
        """Which side closed, once closed."""
        return self._closed_by

    def stats(self) -> TransportStats:
        """Snapshot of the connection counters."""
        return TransportStats(**self._counters)

    @override
    def __repr__(self) -> str:
        return (
            f"WebSocketVoiceTransport(session={self._trace.session_id}, "
            f"closed={self._close_code}, ingress={len(self._ring)})"
        )

    # ------------------------------------------------------------------ egress

    async def send(self, frame: AudioFrame, /) -> None:
        """Deliver one agent frame; silently counted as dropped once closed."""
        if self.closed:
            self._drop(frame)
            return
        packet = encode_packet(frame)
        if await self._deliver(packet):
            self._counters["frames_out"] += 1
            self._counters["bytes_out"] += len(packet)
        else:
            self._drop(frame)

    def _drop(self, frame: AudioFrame) -> None:
        self._counters["egress_dropped"] += 1
        self._sink.emit(
            EgressDropped(trace=frame.trace, at_ns=self._clock(), sequence=frame.sequence)
        )

    async def send_control(self, message: ServerMessage, /) -> bool:
        """Send a server control message; ``False`` if the connection is gone."""
        if self.closed:
            return False
        delivered = await self._deliver(encode_control(message))
        self._counters["control_out"] += delivered
        return delivered

    # ------------------------------------------------------------------ lifecycle

    async def run(self) -> None:
        """Read and dispatch until the connection closes; runs the heartbeat alongside.

        Raises:
            RuntimeError: if called more than once.
        """
        if self._started:
            msg = "WebSocketVoiceTransport.run() may only be called once"
            raise RuntimeError(msg)
        self._started = True
        self._last_seen = self._clock()
        async with asyncio.TaskGroup() as group:
            heartbeat = group.create_task(self._heartbeat_loop(), name="transport-heartbeat")
            try:
                await self._read_loop()
            finally:
                heartbeat.cancel()

    async def close(self, code: int = CloseCode.NORMAL, reason: str = "") -> None:
        """Close the connection with ``code``. Idempotent; the first close wins."""
        reason = truncate_reason(reason)
        if self._finalize(int(code), reason, CloseInitiator.LOCAL):
            await self._connection.close(int(code), reason)

    # ------------------------------------------------------------------ internals

    def _finalize(self, code: int, reason: str, initiator: CloseInitiator) -> bool:
        if self._close_code is not None:
            return False
        self._close_code, self._close_reason, self._closed_by = code, reason, initiator
        self._ring.abort()
        self._control.end()
        self._sink.emit(
            TransportClosed(
                trace=self._trace,
                at_ns=self._clock(),
                code=code,
                reason=reason,
                initiator=initiator,
            )
        )
        return True

    async def _deliver(self, message: str | bytes) -> bool:
        try:
            await self._connection.send(message)
        except TransportClosedError:
            return False
        return True

    async def _read_loop(self) -> None:
        connection = self._connection
        while True:
            try:
                message = await connection.recv()
            except TransportClosedError as closed:
                self._finalize(closed.code, closed.reason, CloseInitiator.REMOTE)
                return
            self._last_seen = self._clock()
            try:
                if isinstance(message, str):
                    await self._on_control(message)
                else:
                    await self._on_audio(message)
            except ProtocolError as violation:
                await self.close(violation.code, violation.reason)
                return

    async def _on_control(self, text: str) -> None:
        if len(text) > self._limits.max_control_chars:
            raise ProtocolError(CloseCode.POLICY_VIOLATION, "control message too large")
        if not self._control_bucket.try_acquire():
            raise ProtocolError(CloseCode.POLICY_VIOLATION, "control message rate exceeded")
        message = decode_client_message(text)
        self._counters["control_in"] += 1
        if isinstance(message, Ping):
            await self.send_control(Pong(message.nonce))
        elif isinstance(message, Pong):
            return
        elif isinstance(message, SessionConfig):
            if message.audio.codec not in self._accepted:
                reason = f"codec {message.audio.codec.value} is not accepted"
                raise ProtocolError(CloseCode.UNSUPPORTED_DATA, reason)
            self._format = message.audio
        else:
            self._control.push(message)

    async def _on_audio(self, data: bytes) -> None:
        if len(data) > HEADER_SIZE + self._limits.max_payload_bytes:
            raise ProtocolError(CloseCode.POLICY_VIOLATION, "audio message too large")
        frame = decode_packet(
            data, received_at_ns=self._last_seen, max_payload=self._limits.max_payload_bytes
        ).frame
        if frame.trace.session_id != self._trace.session_id:
            raise ProtocolError(CloseCode.POLICY_VIOLATION, "packet belongs to another session")
        self._check_format(frame)
        if not self._admit_sequence(frame):
            self._counters["late_dropped"] += 1
            return
        self._counters["frames_in"] += 1
        self._counters["bytes_in"] += len(data)
        self._reader_blocked = True
        try:
            await self._ring.put(frame)
        except RingClosedError:
            return
        finally:
            self._reader_blocked = False

    def _check_format(self, frame: AudioFrame) -> None:
        if frame.codec not in self._accepted:
            reason = f"codec {frame.codec.value} is not accepted"
            raise ProtocolError(CloseCode.UNSUPPORTED_DATA, reason)
        negotiated = self._format
        if negotiated is not None and (
            frame.codec is not negotiated.codec
            or frame.sample_rate_hz != negotiated.sample_rate_hz
            or frame.channels != negotiated.channels
        ):
            reason = "packet format differs from session.config"
            raise ProtocolError(CloseCode.UNSUPPORTED_DATA, reason)

    def _admit_sequence(self, frame: AudioFrame) -> bool:
        sequence = frame.sequence
        if frame.trace.span_id != self._ingress_span:
            self._ingress_span = frame.trace.span_id
        elif sequence < self._next_sequence:
            return False
        elif sequence > self._next_sequence:
            self._counters["lost_packets"] += sequence - self._next_sequence
            self._sink.emit(
                IngressGap(
                    trace=frame.trace,
                    at_ns=frame.origin_ns,
                    expected_sequence=self._next_sequence,
                    received_sequence=sequence,
                )
            )
        self._next_sequence = sequence + 1
        return True

    async def _heartbeat_loop(self) -> None:
        config = self._heartbeat
        nonce = 0
        while True:
            await self._sleep(config.interval_s)
            if self.closed:
                return
            idle = self._clock() - self._last_seen
            if not self._reader_blocked and idle > config.timeout_ns:
                await self.close(CloseCode.GOING_AWAY, "heartbeat timeout")
                return
            nonce += 1
            await self.send_control(Ping(nonce))
