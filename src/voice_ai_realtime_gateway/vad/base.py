"""Streaming voice-activity detection contract.

A :class:`StreamingVAD` consumes ingress frames one at a time and returns a
:class:`VadDecision`. Decisions carry *edge* events (:attr:`VadEvent.ONSET` /
:attr:`VadEvent.OFFSET`) exactly once per speech segment; between edges only the level
and the smoothed ``is_speech`` state change. Implementations must be synchronous,
deterministic for a given frame sequence, and allocation-light: they run inline on the
ingress hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame

__all__ = [
    "SILENCE_FLOOR_DBFS",
    "StreamingVAD",
    "VadConfig",
    "VadDecision",
    "VadEvent",
]

SILENCE_FLOOR_DBFS: Final = -120.0
"""Level reported for digital silence (the log of zero energy is undefined)."""

_MAX_MIN_SPEECH_MS: Final = 2_000
_MAX_HANGOVER_MS: Final = 10_000


@unique
class VadEvent(StrEnum):
    """Speech segment edges."""

    ONSET = "onset"
    OFFSET = "offset"


@dataclass(frozen=True, slots=True, kw_only=True)
class VadConfig:
    """Hysteresis parameters.

    Attributes:
        start_threshold_dbfs: level at or above which a frame counts towards onset.
        stop_threshold_dbfs: level below which a frame counts towards offset. Levels
            between the two thresholds keep the current state (hysteresis band).
        min_speech_ms: consecutive above-start audio required to declare onset
            (speech start confidence: rejects clicks and short transients).
        hangover_ms: consecutive below-stop audio required to declare offset
            (trailing silence: bridges natural pauses inside an utterance).
    """

    start_threshold_dbfs: float = -35.0
    stop_threshold_dbfs: float = -45.0
    min_speech_ms: int = 60
    hangover_ms: int = 300

    def __post_init__(self) -> None:
        """Validate ranges and threshold ordering.

        Raises:
            ValueError: on a non-finite or out-of-range parameter.
        """
        start, stop = self.start_threshold_dbfs, self.stop_threshold_dbfs
        if not (math.isfinite(start) and math.isfinite(stop)):
            msg = "VAD thresholds must be finite"
            raise ValueError(msg)
        if not SILENCE_FLOOR_DBFS <= stop <= start <= 0.0:
            msg = (
                f"VAD thresholds must satisfy {SILENCE_FLOOR_DBFS} <= stop <= start <= 0, "
                f"got start={start}, stop={stop}"
            )
            raise ValueError(msg)
        if not 1 <= self.min_speech_ms <= _MAX_MIN_SPEECH_MS:
            msg = f"min_speech_ms must be in [1, {_MAX_MIN_SPEECH_MS}], got {self.min_speech_ms}"
            raise ValueError(msg)
        if not 1 <= self.hangover_ms <= _MAX_HANGOVER_MS:
            msg = f"hangover_ms must be in [1, {_MAX_HANGOVER_MS}], got {self.hangover_ms}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class VadDecision:
    """Outcome of processing one frame.

    Attributes:
        is_speech: smoothed (post-hysteresis) state after this frame.
        event: the edge crossed on this frame, if any.
        level_dbfs: frame level in dBFS (``SILENCE_FLOOR_DBFS`` for digital silence).
        at_ns: monotonic timestamp of the decision -- the frame's ingress
            ``origin_ns``, so downstream latency budgets include ingress queueing.
    """

    is_speech: bool
    event: VadEvent | None
    level_dbfs: float
    at_ns: int


class StreamingVAD(Protocol):
    """Incremental voice-activity detector."""

    @property
    def config(self) -> VadConfig:
        """Current hysteresis parameters."""

    def process_chunk(self, frame: AudioFrame, /) -> VadDecision:
        """Classify one frame, emitting an edge event at most once per transition."""

    def reconfigure(self, config: VadConfig, /) -> None:
        """Apply new parameters without losing the current speech state."""

    def reset(self) -> None:
        """Forget all state (back to silence)."""
