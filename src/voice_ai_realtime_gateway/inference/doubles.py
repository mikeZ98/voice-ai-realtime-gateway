"""Deterministic in-memory model doubles for tests, demos and offline development.

* :class:`ToneASR` recognises *words encoded as tones*: every frame's PCM16 peak
  amplitude is looked up in a lexicon. Runs of identical amplitude form one word, and
  silence separates words. Synthetic audio therefore yields fully predictable
  transcripts and word timings.
* :class:`ScriptedLLM` streams a scripted reply as whitespace-led token deltas.
* :class:`ToneTTS` synthesizes a square wave with an exact, linear text alignment.

All three yield to the event loop between items, giving the scheduler realistic
cancellation points. They count open generators, so tests can prove that nothing was
orphaned.
"""

from __future__ import annotations

import asyncio
import math
import re
import struct
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, final

from voice_ai_realtime_gateway.inference.ports import (
    SynthesizedAudio,
    TokenDelta,
    TranscriptChunk,
    WordTiming,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.inference.conversation import Turn
    from voice_ai_realtime_gateway.inference.ports import FrameStamper

__all__ = ["UNKNOWN_WORD", "ScriptedLLM", "ToneASR", "ToneTTS", "pcm16_peak"]

UNKNOWN_WORD: Final = "<unk>"
_TOKEN: Final = re.compile(r"\s*\S+")
_NS_PER_MS: Final = 1_000_000

type Reply = str | Callable[[Sequence[Turn], str], str]


def pcm16_peak(payload: memoryview) -> int:
    """Largest absolute sample of little-endian PCM16 ``payload`` (0 when empty)."""
    count = payload.nbytes // 2
    if count == 0:
        return 0
    samples: tuple[int, ...] = struct.unpack_from(f"<{count}h", payload)
    loudest, quietest = max(samples), min(samples)  # C-level scans, no per-sample genexpr
    return max(loudest, -quietest)


@final
class ToneASR:
    """Lexicon-driven recogniser of tone-encoded words."""

    __slots__ = ("_confidence", "_lexicon", "_silence_peak", "active", "sessions")

    def __init__(
        self, lexicon: Mapping[int, str], *, silence_peak: int = 200, confidence: float = 0.95
    ) -> None:
        """Map frame peak amplitudes to words; peaks below ``silence_peak`` are silence."""
        self._lexicon = dict(lexicon)
        self._silence_peak = silence_peak
        self._confidence = confidence
        self.active = 0
        self.sessions = 0

    async def transcribe(
        self, audio: AsyncIterator[AudioFrame], /
    ) -> AsyncGenerator[TranscriptChunk, None]:
        """Yield an interim chunk per completed word, then one final chunk."""
        self.active += 1
        self.sessions += 1
        try:
            words: list[WordTiming] = []
            current: tuple[str, int] | None = None
            position = 0
            last: AudioFrame | None = None
            async for frame in audio:
                last = frame
                peak = pcm16_peak(frame.payload)
                word = None if peak < self._silence_peak else self._lexicon.get(peak, UNKNOWN_WORD)
                if current is not None and word != current[0]:
                    words.append(WordTiming(word=current[0], start_ns=current[1], end_ns=position))
                    current = None
                    yield self._chunk(words, frame, is_final=False)
                if word is not None and current is None:
                    current = (word, position)
                position += frame.duration_ns
                await asyncio.sleep(0)
            if last is not None:
                if current is not None:
                    words.append(WordTiming(word=current[0], start_ns=current[1], end_ns=position))
                yield self._chunk(words, last, is_final=True)
        finally:
            self.active -= 1

    def _chunk(
        self, words: list[WordTiming], frame: AudioFrame, *, is_final: bool
    ) -> TranscriptChunk:
        return TranscriptChunk(
            text=" ".join(timing.word for timing in words),
            is_final=is_final,
            confidence=self._confidence if is_final else self._confidence / 2,
            words=tuple(words),
            trace=frame.trace,
            at_ns=frame.origin_ns,
        )


@final
class ScriptedLLM:
    """Streams a scripted reply token by token."""

    __slots__ = ("_reply", "active", "calls")

    def __init__(self, reply: Reply) -> None:
        """Answer every prompt with ``reply`` (a string, or a function of the context)."""
        self._reply = reply
        self.active = 0
        self.calls: deque[tuple[tuple[Turn, ...], str]] = deque(maxlen=32)

    async def stream(
        self, history: Sequence[Turn], user_text: str, /
    ) -> AsyncGenerator[TokenDelta, None]:
        """Yield the reply as whitespace-led tokens (``"Hi"``, ``" there"``, ...)."""
        self.active += 1
        self.calls.append((tuple(history), user_text))
        try:
            reply = self._reply if isinstance(self._reply, str) else self._reply(history, user_text)
            for index, token in enumerate(_TOKEN.findall(reply)):
                await asyncio.sleep(0)
                yield TokenDelta(text=token, index=index)
        finally:
            self.active -= 1


@final
class ToneTTS:
    """Square-wave synthesizer with exact linear text alignment."""

    __slots__ = (
        "_amplitude",
        "_clock",
        "_frame_ms",
        "_ms_per_char",
        "_pace_s",
        "_payloads",
        "aborted",
        "active",
        "completed",
        "last_closed_ns",
    )

    def __init__(
        self,
        *,
        ms_per_char: int = 8,
        frame_ms: int = 20,
        amplitude: int = 6_000,
        pace_s: float = 0.0,
        clock: MonotonicClock = time.monotonic_ns,
    ) -> None:
        """Voice each character for ``ms_per_char`` (rounded up to whole frames).

        ``pace_s`` sleeps between frames to emulate a synthesizer slower than ``sleep(0)``.
        """
        self._ms_per_char = ms_per_char
        self._frame_ms = frame_ms
        self._amplitude = amplitude
        self._pace_s = pace_s
        self._clock = clock
        self._payloads: dict[tuple[int, int], bytes] = {}
        self.active = 0
        self.completed = 0
        self.aborted = 0
        self.last_closed_ns: int | None = None

    def _payload(self, sample_rate_hz: int, channels: int) -> bytes:
        key = (sample_rate_hz, channels)
        payload = self._payloads.get(key)
        if payload is None:
            samples = sample_rate_hz * self._frame_ms // 1000 * channels
            amplitude = self._amplitude
            payload = struct.pack(
                f"<{samples}h", *(amplitude if i % 2 == 0 else -amplitude for i in range(samples))
            )
            self._payloads[key] = payload
        return payload

    async def synthesize(
        self, text: str, stamper: FrameStamper, /
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        """Yield ``ceil(len(text) * ms_per_char / frame_ms)`` aligned frames (at least 1)."""
        self.active += 1
        finished = False
        try:
            frames = max(1, math.ceil(len(text) * self._ms_per_char / self._frame_ms))
            payload = self._payload(stamper.sample_rate_hz, stamper.channels)
            duration_ns = self._frame_ms * _NS_PER_MS
            for index in range(frames):
                await asyncio.sleep(self._pace_s)
                yield SynthesizedAudio(
                    frame=stamper.stamp(payload, duration_ns=duration_ns),
                    text_offset=len(text) * (index + 1) // frames,
                )
            finished = True
        finally:
            self.active -= 1
            self.last_closed_ns = self._clock()
            if finished:
                self.completed += 1
            else:
                self.aborted += 1
