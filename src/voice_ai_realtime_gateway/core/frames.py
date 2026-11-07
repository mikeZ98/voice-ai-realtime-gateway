"""Audio frame value object.

Zero-copy contract: :meth:`AudioFrame.wrap` stores a *read-only* :class:`memoryview`
over the caller's buffer -- payload bytes are never copied anywhere in the pipeline.
Passing a mutable buffer (``bytearray``) transfers ownership: the producer must not
mutate it after wrapping. Immutable ``bytes`` sources are aliasing-safe by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Buffer

    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

__all__ = ["OPUS_SAMPLE_RATES_HZ", "PCM16_SAMPLE_WIDTH", "AudioCodec", "AudioFrame"]

PCM16_SAMPLE_WIDTH: Final = 2
OPUS_SAMPLE_RATES_HZ: Final = frozenset({8_000, 12_000, 16_000, 24_000, 48_000})
_MAX_CHANNELS: Final = 2


@unique
class AudioCodec(StrEnum):
    """Payload encoding of an :class:`AudioFrame`."""

    PCM16 = "pcm16"
    """Signed 16-bit little-endian linear PCM, interleaved channels."""
    OPUS = "opus"
    """One Opus packet (RFC 6716)."""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AudioFrame:
    """One immutable audio packet plus its correlation and timing metadata.

    Equality is identity (``eq=False``): comparing payloads would be O(n) on the hot
    path and frames are unique events, not values.

    Attributes:
        payload: read-only, 1-D, unsigned-byte view over the encoded audio.
        codec: payload encoding.
        sample_rate_hz: sampling rate.
        channels: 1 (mono) or 2 (stereo).
        duration_ns: media duration carried by the frame.
        sequence: monotonically increasing per-span packet sequence number.
        origin_ns: monotonic timestamp at which the packet entered the pipeline
            (mic capture for ingress, synthesis for egress) -- the glass-to-glass origin.
        trace: correlation pair (``X-Session-ID`` / ``X-Audio-Span-ID``).
    """

    payload: memoryview
    codec: AudioCodec
    sample_rate_hz: int
    channels: int
    duration_ns: int
    sequence: int
    origin_ns: int
    trace: TraceContext

    def __post_init__(self) -> None:
        """Enforce structural invariants once, at construction."""
        view = self.payload
        if not view.readonly or view.ndim != 1 or view.format != "B":
            msg = "payload must be a read-only 1-D unsigned-byte memoryview; use AudioFrame.wrap"
            raise ValueError(msg)
        if view.nbytes == 0:
            msg = "payload must not be empty"
            raise ValueError(msg)
        if self.channels < 1 or self.channels > _MAX_CHANNELS:
            msg = f"channels must be 1 or 2, got {self.channels}"
            raise ValueError(msg)
        if self.sample_rate_hz <= 0:
            msg = f"sample_rate_hz must be positive, got {self.sample_rate_hz}"
            raise ValueError(msg)
        if self.duration_ns <= 0:
            msg = f"duration_ns must be positive, got {self.duration_ns}"
            raise ValueError(msg)
        if self.sequence < 0:
            msg = f"sequence must be non-negative, got {self.sequence}"
            raise ValueError(msg)
        if self.codec is AudioCodec.PCM16:
            if view.nbytes % (PCM16_SAMPLE_WIDTH * self.channels):
                msg = "PCM16 payload must contain whole interleaved sample frames"
                raise ValueError(msg)
        elif self.sample_rate_hz not in OPUS_SAMPLE_RATES_HZ:
            msg = f"Opus sample rate must be one of {sorted(OPUS_SAMPLE_RATES_HZ)}"
            raise ValueError(msg)

    @classmethod
    def wrap(  # noqa: PLR0913 -- a frame is exactly this many orthogonal facts
        cls,
        data: Buffer,
        *,
        codec: AudioCodec,
        sample_rate_hz: int,
        channels: int,
        duration_ns: int,
        sequence: int,
        origin_ns: int,
        trace: TraceContext,
    ) -> AudioFrame:
        """Build a frame over ``data`` without copying it.

        Raises:
            ValueError: if ``data`` is not C-contiguous or violates a frame invariant.
        """
        view = memoryview(data)
        if not view.c_contiguous:
            msg = "payload buffer must be C-contiguous for zero-copy framing"
            raise ValueError(msg)
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        return cls(
            payload=view.toreadonly(),
            codec=codec,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            duration_ns=duration_ns,
            sequence=sequence,
            origin_ns=origin_ns,
            trace=trace,
        )

    @property
    def nbytes(self) -> int:
        """Encoded payload size in bytes."""
        return self.payload.nbytes
