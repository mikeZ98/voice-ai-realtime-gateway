"""Deterministic energy VAD with dual-threshold hysteresis (no C extensions).

Level estimation is the RMS of the frame's PCM16 samples in dBFS (0 dBFS = full-scale
square wave; a full-scale sine reads about -3 dBFS). Samples are decoded explicitly as
little-endian, so results are identical on every host, and the sum of squares uses
:func:`math.sumprod` (C speed). A 20 ms 16 kHz frame costs a few microseconds.

Hysteresis, per frame of duration ``d``::

    SILENCE: level >= start  -> speech_run += d    else speech_run = 0
             speech_run >= min_speech           -> SPEECH, emit ONSET
    SPEECH:  level <  stop   -> silence_run += d   else silence_run = 0
             silence_run >= hangover            -> SILENCE, emit OFFSET

Accumulating media *duration*, not frame counts, keeps the behaviour independent of
the packetisation interval.
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Final, final, override

from voice_ai_realtime_gateway.core.frames import PCM16_SAMPLE_WIDTH, AudioCodec
from voice_ai_realtime_gateway.vad.base import (
    SILENCE_FLOOR_DBFS,
    VadConfig,
    VadDecision,
    VadEvent,
)

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame

__all__ = ["EnergyVAD", "pcm16_level_dbfs"]

_FULL_SCALE_ENERGY: Final = 32768.0**2
_NS_PER_MS: Final = 1_000_000


def pcm16_level_dbfs(payload: memoryview) -> float:
    """RMS level of little-endian PCM16 ``payload`` in dBFS, floored at -120 dBFS.

    Raises:
        ValueError: if ``payload`` is not a whole number of samples.
    """
    nbytes = payload.nbytes
    if nbytes == 0 or nbytes % PCM16_SAMPLE_WIDTH:
        msg = f"PCM16 payload must hold whole samples, got {nbytes} bytes"
        raise ValueError(msg)
    count = nbytes // PCM16_SAMPLE_WIDTH
    samples = struct.unpack_from(f"<{count}h", payload)
    energy = math.sumprod(samples, samples) / count
    if energy == 0:
        return SILENCE_FLOOR_DBFS
    return max(SILENCE_FLOOR_DBFS, 10.0 * math.log10(energy / _FULL_SCALE_ENERGY))


@final
class EnergyVAD:
    """Baseline streaming VAD for PCM16 ingress."""

    __slots__ = (
        "_config",
        "_hangover_ns",
        "_min_speech_ns",
        "_silence_ns",
        "_speaking",
        "_speech_ns",
    )

    def __init__(self, config: VadConfig | None = None) -> None:
        """Start in silence with ``config`` (defaults to :class:`VadConfig`)."""
        self._config = VadConfig() if config is None else config
        self._min_speech_ns = self._config.min_speech_ms * _NS_PER_MS
        self._hangover_ns = self._config.hangover_ms * _NS_PER_MS
        self._speaking = False
        self._speech_ns = 0
        self._silence_ns = 0

    @property
    def config(self) -> VadConfig:
        """Current hysteresis parameters."""
        return self._config

    @property
    def speaking(self) -> bool:
        """Smoothed speech state."""
        return self._speaking

    def reconfigure(self, config: VadConfig, /) -> None:
        """Apply ``config``; the speech state and accumulated runs are preserved."""
        self._config = config
        self._min_speech_ns = config.min_speech_ms * _NS_PER_MS
        self._hangover_ns = config.hangover_ms * _NS_PER_MS

    def reset(self) -> None:
        """Return to silence and forget accumulated runs."""
        self._speaking = False
        self._speech_ns = 0
        self._silence_ns = 0

    def process_chunk(self, frame: AudioFrame, /) -> VadDecision:
        """Classify one PCM16 frame.

        Raises:
            ValueError: for a non-PCM16 frame (encoded audio must be decoded first).
        """
        if frame.codec is not AudioCodec.PCM16:
            msg = f"EnergyVAD needs PCM16 input, got {frame.codec.value}"
            raise ValueError(msg)
        level = pcm16_level_dbfs(frame.payload)
        duration = frame.duration_ns
        event: VadEvent | None = None
        if not self._speaking:
            self._speech_ns = (
                self._speech_ns + duration if level >= self._config.start_threshold_dbfs else 0
            )
            if self._speech_ns >= self._min_speech_ns:
                self._speaking = True
                self._speech_ns = self._silence_ns = 0
                event = VadEvent.ONSET
        else:
            self._silence_ns = (
                self._silence_ns + duration if level < self._config.stop_threshold_dbfs else 0
            )
            if self._silence_ns >= self._hangover_ns:
                self._speaking = False
                self._speech_ns = self._silence_ns = 0
                event = VadEvent.OFFSET
        return VadDecision(
            is_speech=self._speaking, event=event, level_dbfs=level, at_ns=frame.origin_ns
        )

    @override
    def __repr__(self) -> str:
        return f"EnergyVAD(speaking={self._speaking}, config={self._config})"
