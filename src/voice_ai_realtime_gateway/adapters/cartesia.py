"""Cartesia streaming TTS adapter (``StreamingTTS``) over the TTS WebSocket.

Wire protocol (one *context* per text unit)::

    -> {"model_id": ..., "transcript": "<unit>", "voice": {"mode": "id", "id": ...},
        "output_format": {"container": "raw", "encoding": "pcm_s16le",
                          "sample_rate": <rate>},
        "context_id": "<ctx>", "continue": false, "add_timestamps": true}
    <- {"type": "chunk", "context_id": "<ctx>", "data": "<base64 pcm_s16le>", ...}
    <- {... "context_id": "<ctx>", "word_timestamps": {"words": [...], "end": [...]}}
    <- {"type": "done", "context_id": "<ctx>", "done": true}
    <- {"type": "error", "context_id": "<ctx>", "status_code": 4xx/5xx, "error": "..."}
    -> {"context_id": "<ctx>", "cancel": true}            (barge-in)

The timestamp message type has been spelled differently across API versions, so any
message carrying ``word_timestamps`` is accepted. Phoneme timestamps and unknown types
are ignored. Messages of other contexts on the shared connection are ignored.

Framing: decoded PCM is cut into exact ``frame_ms`` frames. Full frames are zero-copy
slices of the decoded buffer; only a frame straddling two chunks is assembled by copy.
The remainder at ``done`` becomes one shorter final frame.

Alignment: word end times map to character offsets; trailing punctuation attached to a
word counts as heard with it. Offsets are monotone and never claim a word before its
end time, so barge-in truncation stays conservative. Timestamps that arrive late only
delay the offset, never inflate it.

Resilience follows the other adapters. A failure before any frame retries the unit
(fresh context id); after frames were yielded, the unit ends gracefully. 401/403 and
402 are fatal; 429 and 5xx are retried. Closing the stream early cancels the context.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import bisect
import contextlib
import json
import math
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NoReturn, final

from voice_ai_realtime_gateway.adapters.resilience import (
    ErrorKind,
    ProviderLink,
    RetryPolicy,
    UpstreamError,
)
from voice_ai_realtime_gateway.inference.ports import SynthesizedAudio
from voice_ai_realtime_gateway.transport.errors import TransportClosedError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from voice_ai_realtime_gateway.adapters.resilience import Connector, Sleeper
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.inference.ports import FrameStamper
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import WebSocketConnection

__all__ = [
    "PROVIDER",
    "AudioChunk",
    "CartesiaStreamingTTS",
    "ContextDone",
    "ContextFailed",
    "WordTimestamps",
    "build_tts_request",
    "classify_status",
    "parse_cartesia_message",
]

PROVIDER: Final = "cartesia"
_NS_PER_S: Final = 1_000_000_000
_SAMPLE_BYTES: Final = 2
_HTTP_UNAUTHORIZED: Final = 401
_HTTP_PAYMENT_REQUIRED: Final = 402
_HTTP_FORBIDDEN: Final = 403
_HTTP_TOO_MANY_REQUESTS: Final = 429
_HTTP_SERVER_ERROR: Final = 500


def build_tts_request(
    text: str, *, context_id: str, model_id: str, voice_id: str, sample_rate_hz: int
) -> str:
    """Generation request for one text unit (compact JSON)."""
    request = {
        "model_id": model_id,
        "transcript": text,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate_hz,
        },
        "context_id": context_id,
        "continue": False,
        "add_timestamps": True,
    }
    return json.dumps(request, separators=(",", ":"))


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioChunk:
    """Decoded PCM for a context."""

    context_id: str
    data: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class WordTimestamps:
    """Word end times (seconds) for a context."""

    context_id: str
    words: tuple[str, ...]
    ends_s: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextDone:
    """All audio for a context has been sent."""

    context_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextFailed:
    """The provider failed a context."""

    context_id: str | None
    status_code: int | None
    message: str


type CartesiaMessage = AudioChunk | WordTimestamps | ContextDone | ContextFailed


def classify_status(status_code: int | None) -> ErrorKind:
    """Map a provider HTTP-style status onto the upstream taxonomy."""
    if status_code in {_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN}:
        return ErrorKind.AUTH
    if status_code == _HTTP_PAYMENT_REQUIRED:
        return ErrorKind.QUOTA
    if status_code == _HTTP_TOO_MANY_REQUESTS:
        return ErrorKind.RATE_LIMIT
    if status_code is not None and status_code >= _HTTP_SERVER_ERROR:
        return ErrorKind.TRANSIENT
    return ErrorKind.PROTOCOL


def _malformed(detail: str) -> NoReturn:
    raise UpstreamError(ErrorKind.PROTOCOL, f"malformed message: {detail}", provider=PROVIDER)


def _context(message: dict[str, object], *, required: bool = True) -> str | None:
    value = message.get("context_id")
    if value is None and not required:
        return None
    if not isinstance(value, str):
        _malformed("context_id must be a string")
    return value


def _timestamps(message: dict[str, object], context_id: str) -> WordTimestamps:
    stamps = message.get("word_timestamps")
    if not isinstance(stamps, dict):
        _malformed("word_timestamps must be an object")
    words, ends = stamps.get("words"), stamps.get("end")
    if not isinstance(words, list) or not isinstance(ends, list) or len(words) != len(ends):
        _malformed("word_timestamps needs equally long 'words' and 'end' lists")
    if not all(isinstance(word, str) for word in words):
        _malformed("timestamp words must be strings")
    if not all(
        not isinstance(end, bool)
        and isinstance(end, int | float)
        and math.isfinite(end)
        and end >= 0
        for end in ends
    ):
        _malformed("timestamp ends must be finite non-negative numbers")
    return WordTimestamps(
        context_id=context_id, words=tuple(words), ends_s=tuple(float(end) for end in ends)
    )


def parse_cartesia_message(text: str) -> CartesiaMessage | None:
    """Parse one server message; ``None`` for message types the adapter ignores.

    Raises:
        UpstreamError: ``PROTOCOL`` for anything malformed (including bad base64).
    """
    try:
        document = json.loads(text)
    except (ValueError, RecursionError):
        _malformed("not JSON")
    if not isinstance(document, dict):
        _malformed("message must be an object")
    kind = document.get("type")
    if kind == "error":
        status = document.get("status_code")
        message = document.get("error", document.get("message", ""))
        return ContextFailed(
            context_id=_context(document, required=False),
            status_code=status
            if isinstance(status, int) and not isinstance(status, bool)
            else None,
            message=message if isinstance(message, str) else "",
        )
    if "word_timestamps" in document:
        return _timestamps(document, _context(document) or "")
    if kind == "chunk":
        context_id = _context(document) or ""
        data = document.get("data")
        if not isinstance(data, str):
            _malformed("chunk data must be a base64 string")
        try:
            pcm = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            _malformed("chunk data is not valid base64")
        return AudioChunk(context_id=context_id, data=pcm)
    if kind == "done":
        return ContextDone(context_id=_context(document) or "")
    return None


@final
class _Aligner:
    """Maps synthesized media time to characters of the unit voiced so far."""

    __slots__ = ("_chars", "_cursor", "_ends_ns", "_text")

    def __init__(self, text: str) -> None:
        self._text = text
        self._cursor = 0
        self._ends_ns: list[int] = []
        self._chars: list[int] = []

    def add(self, stamps: WordTimestamps) -> None:
        text = self._text
        for word, end_s in zip(stamps.words, stamps.ends_s, strict=True):
            start = text.find(word, self._cursor)
            if not word or start < 0:
                continue
            end = start + len(word)
            while end < len(text) and not text[end].isspace():
                end += 1  # punctuation attached to the word is heard with it
            self._cursor = end
            end_ns = max(round(end_s * _NS_PER_S), self._ends_ns[-1] if self._ends_ns else 0)
            self._ends_ns.append(end_ns)
            self._chars.append(end)

    def offset_at(self, media_ns: int) -> int:
        index = bisect.bisect_right(self._ends_ns, media_ns)
        return self._chars[index - 1] if index else 0


@final
class _Framer:
    """Cuts a PCM byte stream into exact frames, slicing without copies when possible."""

    __slots__ = ("_frame_bytes", "_pending")

    def __init__(self, frame_bytes: int) -> None:
        self._frame_bytes = frame_bytes
        self._pending = bytearray()

    def feed(self, data: bytes) -> list[memoryview]:
        size = self._frame_bytes
        view = memoryview(data)
        frames: list[memoryview] = []
        position = 0
        if self._pending:
            position = min(size - len(self._pending), len(view))
            self._pending += view[:position]
            if len(self._pending) == size:
                frames.append(memoryview(bytes(self._pending)))
                self._pending.clear()
        while len(view) - position >= size:
            frames.append(view[position : position + size])
            position += size
        self._pending += view[position:]
        return frames

    def flush(self) -> memoryview | None:
        whole = len(self._pending) - len(self._pending) % _SAMPLE_BYTES
        remainder = bytes(self._pending[:whole])
        self._pending.clear()
        return memoryview(remainder) if remainder else None


@final
class CartesiaStreamingTTS:
    """Synthesizes text units through Cartesia's TTS WebSocket over a reused connection."""

    __slots__ = (
        "_clock",
        "_connection",
        "_connector",
        "_frame_ms",
        "_jitter",
        "_model_id",
        "_policy",
        "_sink",
        "_sleep",
        "_trace",
        "_voice_id",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        connector: Connector,
        *,
        sink: TelemetrySink,
        trace: TraceContext,
        voice_id: str,
        model_id: str = "sonic-2",
        frame_ms: int = 20,
        policy: RetryPolicy | None = None,
        clock: MonotonicClock = time.monotonic_ns,
        sleep: Sleeper = asyncio.sleep,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        """Bind to ``connector`` with a voice and model."""
        self._connector = connector
        self._sink = sink
        self._trace = trace
        self._voice_id = voice_id
        self._model_id = model_id
        self._frame_ms = frame_ms
        self._policy = policy
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._connection: WebSocketConnection | None = None

    async def aclose(self) -> None:
        """Close the persistent connection (end of call). Idempotent."""
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def synthesize(
        self, text: str, stamper: FrameStamper, /
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        """Yield aligned frames for ``text``."""
        link = ProviderLink(
            provider=PROVIDER,
            connector=self._connector,
            sink=self._sink,
            trace=self._trace,
            policy=self._policy,
            clock=self._clock,
            sleep=self._sleep,
            jitter=self._jitter,
        )
        connection = self._connection if self._connection is not None else await link.open()
        self._connection = connection
        context_id: str | None = None
        yielded = 0
        try:
            while True:
                context_id = secrets.token_hex(8)
                try:
                    async with contextlib.aclosing(
                        self._unit(connection, text, context_id, stamper)
                    ) as audio:
                        async for synthesized in audio:
                            yielded += 1
                            yield synthesized
                except (TransportClosedError, UpstreamError) as failure:
                    error = link.classify(failure)
                    self._connection = None
                    await connection.close()
                    if error.fatal or yielded:
                        link.fail(failure)
                        if error.fatal:
                            raise error from failure
                        context_id = None
                        return  # already speaking: end this unit gracefully
                    connection = await link.reopen(failure)
                    self._connection = connection
                    continue
                context_id = None
                return
        finally:
            if context_id is not None and self._connection is not None:
                await self._cancel(self._connection, context_id)

    async def _unit(
        self, connection: WebSocketConnection, text: str, context_id: str, stamper: FrameStamper
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        rate = stamper.sample_rate_hz
        framer = _Framer(rate * self._frame_ms // 1000 * _SAMPLE_BYTES * stamper.channels)
        aligner = _Aligner(text)
        frame_ns = self._frame_ms * 1_000_000
        media_ns = 0
        await connection.send(
            build_tts_request(
                text,
                context_id=context_id,
                model_id=self._model_id,
                voice_id=self._voice_id,
                sample_rate_hz=rate,
            )
        )
        while True:
            message = await connection.recv()
            event = None if isinstance(message, bytes) else parse_cartesia_message(message)
            if event is None or (event.context_id not in {context_id, None}):
                continue
            if isinstance(event, AudioChunk):
                for payload in framer.feed(event.data):
                    media_ns += frame_ns
                    frame = stamper.stamp(payload, duration_ns=frame_ns)
                    yield SynthesizedAudio(frame=frame, text_offset=aligner.offset_at(media_ns))
            elif isinstance(event, WordTimestamps):
                aligner.add(event)
            elif isinstance(event, ContextDone):
                remainder = framer.flush()
                if remainder is not None:
                    samples = remainder.nbytes // (_SAMPLE_BYTES * stamper.channels)
                    duration = max(1, samples * _NS_PER_S // rate)
                    frame = stamper.stamp(remainder, duration_ns=duration)
                    yield SynthesizedAudio(frame=frame, text_offset=len(text))
                return
            else:
                kind = classify_status(event.status_code)
                raise UpstreamError(
                    kind, event.message or f"status {event.status_code}", provider=PROVIDER
                )

    @staticmethod
    async def _cancel(connection: WebSocketConnection, context_id: str) -> None:
        try:
            await connection.send(json.dumps({"context_id": context_id, "cancel": True}))
        except TransportClosedError:
            return
