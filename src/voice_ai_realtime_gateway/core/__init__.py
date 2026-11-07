"""Core data plane: zero-copy audio frames and the bounded frame ring."""

from voice_ai_realtime_gateway.core.frames import (
    OPUS_SAMPLE_RATES_HZ,
    PCM16_SAMPLE_WIDTH,
    AudioCodec,
    AudioFrame,
)
from voice_ai_realtime_gateway.core.ratelimit import TokenBucket
from voice_ai_realtime_gateway.core.ring import (
    AudioRingBuffer,
    OverflowPolicy,
    RingClosedError,
    RingEmptyError,
    RingError,
    RingFullError,
    RingStats,
)

__all__ = [
    "OPUS_SAMPLE_RATES_HZ",
    "PCM16_SAMPLE_WIDTH",
    "AudioCodec",
    "AudioFrame",
    "AudioRingBuffer",
    "OverflowPolicy",
    "RingClosedError",
    "RingEmptyError",
    "RingError",
    "RingFullError",
    "RingStats",
    "TokenBucket",
]
