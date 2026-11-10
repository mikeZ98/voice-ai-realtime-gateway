from __future__ import annotations

import array
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.telemetry.ids import TraceContext

FRAME_NS = 20_000_000


def wrap(data: Any, **overrides: Any) -> AudioFrame:  # noqa: ANN401 -- exercises bad input
    fields: dict[str, Any] = {
        "codec": AudioCodec.PCM16,
        "sample_rate_hz": 16_000,
        "channels": 1,
        "duration_ns": FRAME_NS,
        "sequence": 0,
        "origin_ns": 0,
        "trace": TraceContext.new(),
    }
    fields.update(overrides)
    return AudioFrame.wrap(data, **fields)


def test_wrap_is_zero_copy_and_read_only() -> None:
    source = bytearray(640)
    frame = wrap(source)
    assert frame.payload.obj is source
    assert frame.payload.readonly
    assert frame.nbytes == 640
    source[0] = 0x7F
    assert frame.payload[0] == 0x7F, "frame aliases the producer buffer (no copy)"


def test_wrap_flattens_typed_buffers_without_copying() -> None:
    samples = array.array("h", [1, -1, 2, -2])
    frame = wrap(samples)
    assert frame.payload.format == "B"
    assert frame.payload.ndim == 1
    assert frame.nbytes == 8
    assert frame.payload.obj is samples


def test_wrap_rejects_non_contiguous_buffers() -> None:
    strided = memoryview(bytes(64))[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        wrap(strided)


def test_direct_construction_requires_a_read_only_byte_view() -> None:
    with pytest.raises(ValueError, match="read-only"):
        AudioFrame(
            payload=memoryview(bytearray(4)),
            codec=AudioCodec.PCM16,
            sample_rate_hz=16_000,
            channels=1,
            duration_ns=FRAME_NS,
            sequence=0,
            origin_ns=0,
            trace=TraceContext.new(),
        )


@pytest.mark.parametrize(
    ("data", "overrides", "match"),
    [
        (b"", {}, "empty"),
        (bytes(4), {"channels": 0}, "channels"),
        (bytes(4), {"channels": 3}, "channels"),
        (bytes(4), {"sample_rate_hz": 0}, "sample_rate_hz"),
        (bytes(4), {"duration_ns": 0}, "duration_ns"),
        (bytes(4), {"sequence": -1}, "sequence"),
        (bytes(3), {}, "whole interleaved"),
        (bytes(6), {"channels": 2}, "whole interleaved"),
        (bytes(10), {"codec": AudioCodec.OPUS, "sample_rate_hz": 44_100}, "Opus sample rate"),
    ],
)
def test_structural_invariants_are_enforced(
    data: bytes, overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        wrap(data, **overrides)


@pytest.mark.parametrize("rate", [8_000, 12_000, 16_000, 24_000, 48_000])
def test_opus_accepts_every_rfc6716_rate_and_odd_packet_sizes(rate: int) -> None:
    frame = wrap(bytes(37), codec=AudioCodec.OPUS, sample_rate_hz=rate, channels=2)
    assert frame.codec is AudioCodec.OPUS
    assert frame.nbytes == 37


def test_frames_compare_by_identity_not_payload() -> None:
    trace = TraceContext.new()
    first, second = wrap(bytes(4), trace=trace), wrap(bytes(4), trace=trace)
    assert first == first  # noqa: PLR0124 -- identity semantics is the contract
    assert first != second


@given(nbytes=st.integers(min_value=1, max_value=512), channels=st.sampled_from([1, 2]))
def test_pcm16_frames_exist_exactly_on_interleaved_sample_boundaries(
    nbytes: int, channels: int
) -> None:
    aligned = nbytes % (2 * channels) == 0
    if aligned:
        frame = wrap(bytes(nbytes), channels=channels)
        assert frame.nbytes == nbytes
    else:
        with pytest.raises(ValueError, match="whole interleaved sample frames"):
            wrap(bytes(nbytes), channels=channels)
