"""What the user actually heard: playback accounting for barge-in truncation.

Two pieces cooperate:

* :class:`SpeechMap` aligns one response's synthesized audio with its text. TTS reports,
  for every frame, how many characters of the current text unit are voiced by the end
  of that frame (``text_offset``). The map stores the cumulative media position of each
  frame in preallocated ``array('q')`` columns (24 B per frame).
* :class:`PlaybackLedger` wraps the session's egress and records, per response span,
  when the first frame left the gateway and how much media has been sent.

On interruption at time ``t``, the client has played at most
``min(media_sent, t - first_sent)`` (it plays in real time from the first packet). That
position maps through the alignment to a character offset, which snaps back to a whole
word. A cut inside a word drops the word: the history may under-report what was said,
but never contains words the user did not hear.
"""

from __future__ import annotations

import bisect
import time
from array import array
from typing import TYPE_CHECKING, final, override

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.domain.duplex import AudioEgress
    from voice_ai_realtime_gateway.telemetry.ids import AudioSpanId

__all__ = ["PlaybackLedger", "SpeechMap", "word_prefix"]


def word_prefix(text: str, cut: int) -> str:
    """``text[:cut]`` without a trailing partial word, stripped.

    Raises:
        ValueError: if ``cut`` is outside ``[0, len(text)]``.
    """
    if not 0 <= cut <= len(text):
        msg = f"cut {cut} outside [0, {len(text)}]"
        raise ValueError(msg)
    if cut < len(text) and not text[cut].isspace():
        while cut > 0 and not text[cut - 1].isspace():
            cut -= 1
    return text[:cut].strip()


@final
class SpeechMap:
    """Text-to-audio alignment of one agent response."""

    __slots__ = ("_ends", "_media_ns", "_offsets", "_unit_of", "_units")

    def __init__(self) -> None:
        """Create an empty map."""
        self._units: list[str] = []
        self._ends = array("q")
        self._unit_of = array("q")
        self._offsets = array("q")
        self._media_ns = 0

    @property
    def units(self) -> tuple[str, ...]:
        """Text units sent to TTS, in order."""
        return tuple(self._units)

    @property
    def synthesized_ns(self) -> int:
        """Media duration synthesized so far."""
        return self._media_ns

    @property
    def frames(self) -> int:
        """Frames recorded."""
        return len(self._ends)

    def begin_unit(self, text: str) -> None:
        """Start the next text unit; subsequent frames belong to it."""
        self._units.append(text)

    def add_frame(self, duration_ns: int, text_offset: int) -> None:
        """Record one synthesized frame of the current unit.

        Raises:
            ValueError: without a current unit, or with an offset outside it.
        """
        if not self._units:
            msg = "add_frame() before begin_unit()"
            raise ValueError(msg)
        unit = len(self._units) - 1
        if not 0 <= text_offset <= len(self._units[unit]):
            msg = f"text_offset {text_offset} outside unit of {len(self._units[unit])} chars"
            raise ValueError(msg)
        self._media_ns += duration_ns
        self._ends.append(self._media_ns)
        self._unit_of.append(unit)
        self._offsets.append(text_offset)

    def text_through(self, media_ns: int) -> str:
        """Whole words voiced within the first ``media_ns`` of the response."""
        index = bisect.bisect_right(self._ends, media_ns) - 1
        if index < 0:
            return ""
        unit = self._unit_of[index]
        parts = [text.strip() for text in self._units[:unit]]
        parts.append(word_prefix(self._units[unit], self._offsets[index]))
        return " ".join(part for part in parts if part)

    @override
    def __repr__(self) -> str:
        return f"SpeechMap(units={len(self._units)}, frames={len(self._ends)})"


@final
class PlaybackLedger:
    """Egress wrapper recording what left the gateway for the current response span."""

    __slots__ = ("_clock", "_first_sent_ns", "_inner", "_sent_ns", "_span")

    def __init__(self, inner: AudioEgress, *, clock: MonotonicClock = time.monotonic_ns) -> None:
        """Forward every frame to ``inner`` and account for it."""
        self._inner = inner
        self._clock = clock
        self._span: AudioSpanId | None = None
        self._first_sent_ns = 0
        self._sent_ns = 0

    async def send(self, frame: AudioFrame, /) -> None:
        """Deliver ``frame`` through the wrapped egress, then record it as sent."""
        await self._inner.send(frame)
        span = frame.trace.span_id
        if span != self._span:
            self._span = span
            self._first_sent_ns = self._clock()
            self._sent_ns = 0
        self._sent_ns += frame.duration_ns

    def first_sent_ns(self, span: AudioSpanId) -> int | None:
        """When the first frame of ``span`` left the gateway, if any did."""
        return self._first_sent_ns if span == self._span else None

    def sent_ns(self, span: AudioSpanId) -> int:
        """Media duration of ``span`` sent so far."""
        return self._sent_ns if span == self._span else 0

    def heard_ns(self, span: AudioSpanId, now_ns: int) -> int:
        """Upper bound of ``span`` media the client has played by ``now_ns``."""
        if span != self._span:
            return 0
        return min(self._sent_ns, max(0, now_ns - self._first_sent_ns))
