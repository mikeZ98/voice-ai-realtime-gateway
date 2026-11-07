"""Model ports: streaming ASR, LLM and TTS.

Every port returns an :class:`~collections.abc.AsyncGenerator`, so a consumer can close
it deterministically with :func:`contextlib.aclosing`. When a barge-in cancels a stage,
the generator's ``finally`` blocks (closing provider streams, releasing sockets) run
immediately instead of whenever the garbage collector finds a suspended generator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, final, override

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Buffer, Sequence

    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.inference.conversation import Turn
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

__all__ = [
    "FrameStamper",
    "StreamingASR",
    "StreamingLLM",
    "StreamingTTS",
    "SynthesizedAudio",
    "TokenDelta",
    "TranscriptChunk",
    "WordTiming",
]


# --------------------------------------------------------------------------- ASR


@dataclass(frozen=True, slots=True, kw_only=True)
class WordTiming:
    """One recognised word; times are media offsets from the start of the utterance."""

    word: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptChunk:
    """An interim or final recognition result for one utterance.

    Interim results are revisable hypotheses; the final result is the committed text.
    ``trace`` is the span of the client audio the text was recognised from.
    """

    text: str
    is_final: bool
    confidence: float
    words: tuple[WordTiming, ...]
    trace: TraceContext
    at_ns: int


class StreamingASR(Protocol):
    """Incremental speech recognition over one utterance."""

    def transcribe(
        self, audio: AsyncIterator[AudioFrame], /
    ) -> AsyncGenerator[TranscriptChunk, None]:
        """Recognise ``audio`` until it ends; the last chunk yielded is final."""


# --------------------------------------------------------------------------- LLM


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenDelta:
    """An incremental piece of generated text."""

    text: str
    index: int


class StreamingLLM(Protocol):
    """Incremental text generation."""

    def stream(
        self, history: Sequence[Turn], user_text: str, /
    ) -> AsyncGenerator[TokenDelta, None]:
        """Generate the reply to ``user_text`` given the preceding ``history``."""


# --------------------------------------------------------------------------- TTS


@dataclass(frozen=True, slots=True, kw_only=True)
class SynthesizedAudio:
    """One synthesized frame plus its text alignment.

    ``text_offset`` is how many characters of the unit being synthesized are voiced by
    the end of this frame. Offsets never decrease within a unit, and the last frame
    reports the full unit length. Barge-in truncation relies on this alignment.
    """

    frame: AudioFrame
    text_offset: int


@final
class FrameStamper:
    """Wraps synthesized PCM in frames of the active response span (zero-copy).

    One stamper serves a whole response, so sequence numbers stay continuous across
    text units.
    """

    __slots__ = ("_channels", "_clock", "_codec", "_next", "_rate", "_trace")

    def __init__(
        self,
        trace: TraceContext,
        *,
        sample_rate_hz: int = 16_000,
        channels: int = 1,
        codec: AudioCodec = AudioCodec.PCM16,
        clock: MonotonicClock = time.monotonic_ns,
    ) -> None:
        """Stamp frames for ``trace``'s span in the given format."""
        self._trace = trace
        self._rate = sample_rate_hz
        self._channels = channels
        self._codec = codec
        self._clock = clock
        self._next = 0

    @property
    def trace(self) -> TraceContext:
        """The response span being stamped."""
        return self._trace

    @property
    def sample_rate_hz(self) -> int:
        """Output sample rate."""
        return self._rate

    @property
    def channels(self) -> int:
        """Output channel count."""
        return self._channels

    @property
    def stamped(self) -> int:
        """Frames stamped so far (also the next sequence number)."""
        return self._next

    def stamp(self, data: Buffer, *, duration_ns: int) -> AudioFrame:
        """Wrap ``data`` (not copied) as the next frame of the span.

        Raises:
            ValueError: if ``data`` violates a frame invariant.
        """
        frame = AudioFrame.wrap(
            data,
            codec=self._codec,
            sample_rate_hz=self._rate,
            channels=self._channels,
            duration_ns=duration_ns,
            sequence=self._next,
            origin_ns=self._clock(),
            trace=self._trace,
        )
        self._next += 1
        return frame

    @override
    def __repr__(self) -> str:
        return f"FrameStamper(span={self._trace.span_id}, stamped={self._next})"


class StreamingTTS(Protocol):
    """Incremental speech synthesis of one text unit."""

    def synthesize(
        self, text: str, stamper: FrameStamper, /
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        """Yield aligned frames for ``text``, stamped through ``stamper``."""
