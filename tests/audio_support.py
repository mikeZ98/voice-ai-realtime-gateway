"""Deterministic PCM16 signal generators for VAD and pipeline tests."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Final

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

FRAME_MS: Final = 20
NS_PER_MS: Final = 1_000_000
RATE: Final = 16_000
LOUD: Final = 8_000
"""Square-wave amplitude ~ -12.3 dBFS: well above the default start threshold."""


def square_wave(amplitude: int, samples: int) -> bytes:
    """Alternating +/-amplitude: RMS equals ``amplitude`` exactly."""
    values = [amplitude if index % 2 == 0 else -amplitude for index in range(samples)]
    return struct.pack(f"<{samples}h", *values)


def pcm_frame(  # noqa: PLR0913 -- frame metadata is orthogonal
    trace: TraceContext,
    *,
    amplitude: int = 0,
    sequence: int = 0,
    origin_ns: int = 0,
    ms: int = FRAME_MS,
    rate: int = RATE,
) -> AudioFrame:
    samples = rate * ms // 1000
    return AudioFrame.wrap(
        square_wave(amplitude, samples),
        codec=AudioCodec.PCM16,
        sample_rate_hz=rate,
        channels=1,
        duration_ns=ms * NS_PER_MS,
        sequence=sequence,
        origin_ns=origin_ns,
        trace=trace,
    )
