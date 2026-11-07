"""Utterance segmentation between the ingress tap and the ASR stage.

The VAD declares onset only after ``min_speech_ms`` of speech, so the first syllable
would be lost without look-back. :class:`SpeechCapture` keeps a short rolling
**pre-roll** of recent frames. On onset it opens an :class:`Utterance` whose bounded,
drop-oldest ring is seeded with the pre-roll; it forwards frames until offset, then
closes the ring so the ASR stream ends with a final result.

``observe`` is synchronous and never blocks, so the realtime ingress loop is never
slowed by a slow recogniser. Queueing delay is bounded by the utterance ring, and
overflow is reported as ``BufferOverflow`` telemetry.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import deque
from typing import TYPE_CHECKING, Final, Self, final, override

from voice_ai_realtime_gateway.core.ring import AudioRingBuffer, OverflowPolicy
from voice_ai_realtime_gateway.vad.base import VadEvent

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.inference.conversation import UserTurn
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.vad.base import VadDecision

__all__ = ["DEFAULT_PREROLL_MS", "SpeechCapture", "Utterance"]

DEFAULT_PREROLL_MS: Final = 300
_NS_PER_MS: Final = 1_000_000


@final
class Utterance:
    """One VAD-delimited user utterance on its way to the recogniser."""

    __slots__ = ("_transcript", "ended_ns", "number", "ring", "started_ns", "trace")

    def __init__(self, number: int, ring: AudioRingBuffer, first: AudioFrame) -> None:
        """Open utterance ``number`` starting with ``first``."""
        self.number = number
        self.ring = ring
        self.trace: TraceContext = first.trace
        self.started_ns = first.origin_ns
        self.ended_ns: int | None = None
        self._transcript: asyncio.Future[UserTurn | None] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def transcript(self) -> asyncio.Future[UserTurn | None]:
        """Resolves to the recorded user turn, or ``None`` if nothing was recognised."""
        return self._transcript

    def resolve(self, turn: UserTurn | None) -> None:
        """Publish the recognition result. Idempotent; a cancelled waiter is fine."""
        if not self._transcript.done():
            self._transcript.set_result(turn)

    @override
    def __repr__(self) -> str:
        return f"Utterance(number={self.number}, frames={len(self.ring)}, ended={self.ended_ns})"


@final
class SpeechCapture:
    """Turns the ingress frame stream plus VAD edges into a stream of utterances."""

    __slots__ = (
        "_closed",
        "_current",
        "_last",
        "_max_bytes",
        "_max_frames",
        "_numbers",
        "_preroll",
        "_preroll_limit_ns",
        "_preroll_ns",
        "_queue",
        "_sink",
    )

    def __init__(
        self,
        *,
        sink: TelemetrySink,
        preroll_ms: int = DEFAULT_PREROLL_MS,
        max_frames: int = 512,
        max_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        """Buffer each utterance in a ring of ``max_frames`` / ``max_bytes``.

        Raises:
            ValueError: if ``preroll_ms`` is negative (ring bounds are validated by the
                ring itself on first use).
        """
        if preroll_ms < 0:
            msg = f"preroll_ms must be non-negative, got {preroll_ms}"
            raise ValueError(msg)
        self._sink = sink
        self._preroll_limit_ns = preroll_ms * _NS_PER_MS
        self._max_frames = max_frames
        self._max_bytes = max_bytes
        self._preroll: deque[AudioFrame] = deque()
        self._preroll_ns = 0
        self._current: Utterance | None = None
        self._last: Utterance | None = None
        self._numbers = itertools.count()
        self._queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        self._closed = False

    @property
    def current(self) -> Utterance | None:
        """The utterance being captured, if the user is speaking."""
        return self._current

    @property
    def last_completed(self) -> Utterance | None:
        """The most recent utterance closed by a VAD offset."""
        return self._last

    @property
    def preroll_ns(self) -> int:
        """Media duration held in the pre-roll."""
        return self._preroll_ns

    def __aiter__(self) -> Self:
        """Iterate over utterances as they open."""
        return self

    async def __anext__(self) -> Utterance:
        """Next opened utterance; ends after :meth:`close`."""
        utterance = await self._queue.get()
        if utterance is None:
            self._queue.put_nowait(None)  # keep the stream ended for later readers
            raise StopAsyncIteration
        return utterance

    def observe(self, frame: AudioFrame, decision: VadDecision) -> None:
        """Ingress tap: route ``frame`` according to the VAD ``decision``."""
        if self._closed:
            return
        current = self._current
        if current is None:
            self._remember(frame)
            if decision.event is VadEvent.ONSET:
                self._open()
            return
        current.ring.put_nowait(frame)
        if decision.event is VadEvent.OFFSET:
            current.ended_ns = decision.at_ns
            current.ring.close()
            self._last = current
            self._current = None

    def close(self) -> None:
        """Stop capturing: abort the open utterance and end the stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._current is not None:
            self._current.ring.abort()
            self._current = None
        self._queue.put_nowait(None)

    def _remember(self, frame: AudioFrame) -> None:
        self._preroll.append(frame)
        self._preroll_ns += frame.duration_ns
        while len(self._preroll) > 1 and self._preroll_ns > self._preroll_limit_ns:
            self._preroll_ns -= self._preroll.popleft().duration_ns

    def _open(self) -> None:
        ring = AudioRingBuffer(
            capacity_frames=self._max_frames,
            capacity_bytes=self._max_bytes,
            policy=OverflowPolicy.DROP_OLDEST,
            sink=self._sink,
        )
        utterance = Utterance(next(self._numbers), ring, self._preroll[0])
        for frame in self._preroll:
            ring.put_nowait(frame)
        self._preroll.clear()
        self._preroll_ns = 0
        self._current = utterance
        self._queue.put_nowait(utterance)
