"""Deepgram live transcription adapter (``StreamingASR``).

Wire protocol (Deepgram ``/v1/listen`` WebSocket):

* client -> server: binary messages of raw audio in the encoding declared in the
  connection URL (for example ``encoding=linear16&sample_rate=16000&channels=1`` with
  ``interim_results=true``), then a text ``{"type":"CloseStream"}`` to flush.
* server -> client: text JSON. ``Results`` carries
  ``channel.alternatives[0].{transcript, confidence, words[]}`` plus ``is_final``,
  ``start`` and ``duration`` (seconds from the start of the connection's audio). Other
  types (``SpeechStarted``, ``UtteranceEnd``, future types) are ignored. After
  ``CloseStream`` the server flushes final results, sends ``Metadata`` and closes with
  1000.

Building the URL, authenticating (``Authorization: Token ...``) and choosing the
WebSocket library belong to the injected :data:`~.resilience.Connector`. A handshake
refused with HTTP 401/403 or 402 should raise ``UpstreamError(AUTH / QUOTA)``.

Resilience: audio not yet covered by a final result is kept in a bounded replay buffer.
When the connection drops, or the server sends a malformed message, the adapter
reconnects under its :class:`~.resilience.RetryPolicy`, replays that audio and shifts
word timings by the replay offset. The utterance continues without losing or
duplicating finalised words. A normal close counts as end of stream only if it is
complete: ``Metadata`` arrived after ``CloseStream``, or no audio remains unfinalised.
A premature normal close is treated as a drop. Fatal errors propagate and fault the
session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NoReturn, final

from voice_ai_realtime_gateway.adapters.resilience import (
    ErrorKind,
    ProviderLink,
    RetryPolicy,
    UpstreamError,
)
from voice_ai_realtime_gateway.inference.ports import TranscriptChunk, WordTiming
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from voice_ai_realtime_gateway.adapters.resilience import Connector, Sleeper
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import WebSocketConnection

__all__ = [
    "CLOSE_STREAM",
    "PROVIDER",
    "DeepgramMetadata",
    "DeepgramResult",
    "DeepgramStreamingASR",
    "DeepgramWord",
    "parse_deepgram_message",
]

PROVIDER: Final = "deepgram"
CLOSE_STREAM: Final = '{"type":"CloseStream"}'
_NS_PER_S: Final = 1_000_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepgramWord:
    """One recognised word; times in seconds from the connection's audio start."""

    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepgramResult:
    """A parsed ``Results`` message."""

    transcript: str
    is_final: bool
    confidence: float
    words: tuple[DeepgramWord, ...]
    start_s: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class DeepgramMetadata:
    """The stream summary Deepgram sends when a stream ends."""


def _malformed(detail: str) -> NoReturn:
    raise UpstreamError(ErrorKind.PROTOCOL, f"malformed message: {detail}", provider=PROVIDER)


def _number(document: dict[str, object], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        _malformed(f"{key!r} must be a finite number")
    return float(value)


def _dict(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _malformed(f"{what} must be an object")
    return value


def _word(raw: object) -> DeepgramWord:
    item = _dict(raw, "word")
    text = item.get("punctuated_word", item.get("word"))
    if not isinstance(text, str):
        _malformed("word text must be a string")
    start, end = _number(item, "start"), _number(item, "end")
    if not 0 <= start <= end:
        _malformed("word times must satisfy 0 <= start <= end")
    return DeepgramWord(word=text, start_s=start, end_s=end)


def parse_deepgram_message(text: str) -> DeepgramResult | DeepgramMetadata | None:
    """Parse one server message; ``None`` for message types the adapter ignores.

    Raises:
        UpstreamError: ``PROTOCOL`` for anything malformed.
    """
    try:
        document = json.loads(text)
    except (ValueError, RecursionError):
        _malformed("not JSON")
    message = _dict(document, "message")
    kind = message.get("type")
    if kind == "Metadata":
        return DeepgramMetadata()
    if kind != "Results":
        return None
    alternatives = _dict(message.get("channel"), "channel").get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        _malformed("channel.alternatives must be a non-empty list")
    best = _dict(alternatives[0], "alternative")
    transcript = best.get("transcript")
    is_final = message.get("is_final")
    words = best.get("words", [])
    if not isinstance(transcript, str):
        _malformed("transcript must be a string")
    if not isinstance(is_final, bool):
        _malformed("is_final must be a boolean")
    if not isinstance(words, list):
        _malformed("words must be a list")
    start, duration = _number(message, "start"), _number(message, "duration")
    if start < 0 or duration < 0:
        _malformed("start and duration must be non-negative")
    confidence = _number(best, "confidence")
    if not 0.0 <= confidence <= 1.0:
        _malformed("confidence must be in [0, 1]")
    return DeepgramResult(
        transcript=transcript,
        is_final=is_final,
        confidence=confidence,
        words=tuple(_word(word) for word in words),
        start_s=start,
        duration_s=duration,
    )


@final
class _Replay:
    """Audio sent but not yet covered by a final result."""

    __slots__ = (
        "_frames",
        "_limit",
        "audio_done",
        "connection_offset_ns",
        "flushed",
        "media_ns",
        "trace",
    )

    def __init__(self, limit: int, trace: TraceContext) -> None:
        self._frames: deque[tuple[AudioFrame, int]] = deque()
        self._limit = limit
        self.media_ns = 0
        self.connection_offset_ns = 0
        self.audio_done = False
        self.flushed = False
        self.trace = trace

    @property
    def pending(self) -> bool:
        """Whether some sent audio is not covered by a final result yet."""
        return bool(self._frames)

    def remember(self, frame: AudioFrame) -> None:
        self._frames.append((frame, self.media_ns))
        self.media_ns += frame.duration_ns
        self.trace = frame.trace
        if len(self._frames) > self._limit:
            self._frames.popleft()

    def commit_through(self, end_ns: int) -> None:
        frames = self._frames
        while frames and frames[0][1] + frames[0][0].duration_ns <= end_ns:
            frames.popleft()

    def rewind(self) -> list[AudioFrame]:
        """Frames to resend on a new connection, whose audio clock restarts at zero."""
        self.connection_offset_ns = self._frames[0][1] if self._frames else self.media_ns
        return [frame for frame, _ in self._frames]

    def absolute_ns(self, seconds: float) -> int:
        return self.connection_offset_ns + round(seconds * _NS_PER_S)


@final
class DeepgramStreamingASR:
    """Streams one utterance per :meth:`transcribe` call to Deepgram."""

    __slots__ = (
        "_clock",
        "_connector",
        "_jitter",
        "_policy",
        "_replay_limit",
        "_sink",
        "_sleep",
        "_trace",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        connector: Connector,
        *,
        sink: TelemetrySink,
        trace: TraceContext,
        policy: RetryPolicy | None = None,
        replay_limit_frames: int = 1024,
        clock: MonotonicClock = time.monotonic_ns,
        sleep: Sleeper = asyncio.sleep,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        """Bind to ``connector``; ``replay_limit_frames`` bounds replay memory (~20 s)."""
        self._connector = connector
        self._sink = sink
        self._trace = trace
        self._policy = policy
        self._replay_limit = replay_limit_frames
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter

    def _link(self) -> ProviderLink:
        return ProviderLink(
            provider=PROVIDER,
            connector=self._connector,
            sink=self._sink,
            trace=self._trace,
            policy=self._policy,
            clock=self._clock,
            sleep=self._sleep,
            jitter=self._jitter,
        )

    async def transcribe(
        self, audio: AsyncIterator[AudioFrame], /
    ) -> AsyncGenerator[TranscriptChunk, None]:
        """Stream ``audio`` and yield interim and final transcript chunks."""
        link = self._link()
        replay = _Replay(self._replay_limit, self._trace)
        connection = await link.open()
        sender = asyncio.create_task(self._send(connection, audio, replay, []))
        try:
            while True:
                try:
                    result = await self._next_result(connection)
                except TransportClosedError as closed:
                    if self._complete(closed, replay, sender):
                        return
                    cause: BaseException = closed
                except UpstreamError as malformed:
                    cause = malformed
                else:
                    if isinstance(result, DeepgramResult):
                        yield self._chunk(result, replay)
                    elif result is not None:
                        replay.flushed = replay.audio_done
                    continue
                await self._stop(sender)
                await connection.close()
                connection = await link.reopen(cause)
                sender = asyncio.create_task(self._send(connection, audio, replay, replay.rewind()))
        finally:
            await self._stop(sender)
            await link.close(connection)

    @staticmethod
    def _complete(
        closed: TransportClosedError, replay: _Replay, sender: asyncio.Task[None]
    ) -> bool:
        return (
            closed.code == CloseCode.NORMAL
            and replay.audio_done
            and sender.done()
            and (replay.flushed or not replay.pending)
        )

    @staticmethod
    async def _next_result(
        connection: WebSocketConnection,
    ) -> DeepgramResult | DeepgramMetadata | None:
        message = await connection.recv()
        if isinstance(message, bytes):
            return None  # Deepgram sends no binary messages; ignore rather than guess
        return parse_deepgram_message(message)

    @staticmethod
    async def _stop(sender: asyncio.Task[None]) -> None:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender

    @staticmethod
    async def _send(
        connection: WebSocketConnection,
        audio: AsyncIterator[AudioFrame],
        replay: _Replay,
        backlog: list[AudioFrame],
    ) -> None:
        try:
            for frame in backlog:
                await connection.send(bytes(frame.payload))
            if not replay.audio_done:
                async for frame in audio:
                    replay.remember(frame)
                    await connection.send(bytes(frame.payload))
                replay.audio_done = True
            await connection.send(CLOSE_STREAM)
        except TransportClosedError:
            return  # the receive loop observes the drop and reconnects

    def _chunk(self, result: DeepgramResult, replay: _Replay) -> TranscriptChunk:
        if result.is_final:
            replay.commit_through(replay.absolute_ns(result.start_s + result.duration_s))
        return TranscriptChunk(
            text=result.transcript,
            is_final=result.is_final,
            confidence=result.confidence,
            words=tuple(
                WordTiming(
                    word=word.word,
                    start_ns=replay.absolute_ns(word.start_s),
                    end_ns=replay.absolute_ns(word.end_s),
                )
                for word in result.words
            ),
            trace=replay.trace,
            at_ns=self._clock(),
        )
