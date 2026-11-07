"""Packet framing: golden layout, zero-copy decode, strict validation and fuzzing."""

from __future__ import annotations

import contextlib
import struct

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.telemetry.ids import TraceContext, parse_session_id, parse_span_id
from voice_ai_realtime_gateway.transport.errors import CloseCode, ProtocolError
from voice_ai_realtime_gateway.transport.framing import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    HEADER_SIZE,
    PACKET_MAGIC,
    PacketStreamDecoder,
    decode_packet,
    encode_packet,
)

TRACE = TraceContext(parse_session_id("0123456789abcdef" * 2), parse_span_id("fedcba9876543210"))


def frame(  # noqa: PLR0913 -- one knob per header field under test
    payload: bytes = bytes(640),
    *,
    codec: AudioCodec = AudioCodec.PCM16,
    rate: int = 16_000,
    channels: int = 1,
    duration_ns: int = 20_000_000,
    sequence: int = 7,
    origin_ns: int = 123_456_789,
    trace: TraceContext = TRACE,
) -> AudioFrame:
    return AudioFrame.wrap(
        payload,
        codec=codec,
        sample_rate_hz=rate,
        channels=channels,
        duration_ns=duration_ns,
        sequence=sequence,
        origin_ns=origin_ns,
        trace=trace,
    )


def patched(packet: bytes, offset: int, fmt: str, value: int | bytes) -> bytes:
    data = bytearray(packet)
    struct.pack_into(fmt, data, offset, value)
    return bytes(data)


# ------------------------------------------------------------------------ layout


def test_header_is_54_bytes_and_golden() -> None:
    assert HEADER_SIZE == 54
    packet = encode_packet(frame(b"\x01\x02\x03\x04"))
    assert packet[:6] == b"VG\x01\x01\x01\x00"
    assert struct.unpack_from(">IIIQ", packet, 6) == (16_000, 20_000_000, 7, 123_456_789)
    assert packet[26:42].hex() == TRACE.session_id
    assert packet[42:50].hex() == TRACE.span_id
    assert struct.unpack_from(">I", packet, 50) == (4,)
    assert packet[54:] == b"\x01\x02\x03\x04"


def test_decode_is_zero_copy_and_rebases_origin_on_the_receiver_clock() -> None:
    packet = encode_packet(frame(bytes(range(256)) * 2, sequence=99, origin_ns=5))
    decoded = decode_packet(packet, received_at_ns=777)
    assert decoded.frame.payload.obj is packet, "payload is a view into the message"
    assert bytes(decoded.frame.payload) == bytes(range(256)) * 2
    assert decoded.frame.origin_ns == 777
    assert decoded.sender_timestamp_ns == 5
    assert decoded.frame.sequence == 99
    assert decoded.frame.trace == TRACE


def test_opus_and_stereo_round_trip() -> None:
    original = frame(bytes(73), codec=AudioCodec.OPUS, rate=48_000, channels=2)
    decoded = decode_packet(encode_packet(original), received_at_ns=0).frame
    assert (decoded.codec, decoded.sample_rate_hz, decoded.channels, decoded.nbytes) == (
        AudioCodec.OPUS,
        48_000,
        2,
        73,
    )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"duration_ns": 2**32}, "duration_ns"),
        ({"sequence": 2**32}, "sequence"),
        ({"origin_ns": -1}, "origin_ns"),
        ({"rate": 2**32}, "sample_rate_hz"),
    ],
)
def test_encode_rejects_values_that_do_not_fit(overrides: dict[str, int], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        encode_packet(frame(**overrides))  # type: ignore[arg-type]


# ------------------------------------------------------------------------ validation

GOOD = encode_packet(frame(bytes(64)))


@pytest.mark.parametrize(
    ("data", "code", "match"),
    [
        (GOOD[:53], CloseCode.INVALID_PAYLOAD, "truncated"),
        (patched(GOOD, 0, "2s", b"XX"), CloseCode.INVALID_PAYLOAD, "magic"),
        (patched(GOOD, 2, "B", 2), CloseCode.UNSUPPORTED_DATA, "version"),
        (patched(GOOD, 3, "B", 9), CloseCode.UNSUPPORTED_DATA, "codec"),
        (patched(GOOD, 5, "B", 1), CloseCode.INVALID_PAYLOAD, "flags"),
        (patched(GOOD, 50, ">I", 2**20), CloseCode.MESSAGE_TOO_BIG, "exceeds"),
        (GOOD + b"\x00", CloseCode.INVALID_PAYLOAD, "length mismatch"),
        (GOOD[:-1], CloseCode.INVALID_PAYLOAD, "length mismatch"),
        (patched(GOOD, 26, "16s", bytes(16)), CloseCode.INVALID_PAYLOAD, "invalid packet"),
        (patched(GOOD, 42, "8s", bytes(8)), CloseCode.INVALID_PAYLOAD, "invalid packet"),
        (patched(GOOD, 4, "B", 3), CloseCode.INVALID_PAYLOAD, "channels"),
        (patched(GOOD, 10, ">I", 0), CloseCode.INVALID_PAYLOAD, "duration_ns"),
        (
            patched(GOOD[:-1], 50, ">I", 63),
            CloseCode.INVALID_PAYLOAD,
            "whole interleaved",
        ),
        (patched(GOOD[:54], 50, ">I", 0), CloseCode.INVALID_PAYLOAD, "empty"),
    ],
)
def test_malformed_packets_map_to_close_codes(data: bytes, code: CloseCode, match: str) -> None:
    with pytest.raises(ProtocolError, match=match) as excinfo:
        decode_packet(data, received_at_ns=0)
    assert excinfo.value.code is code


def test_payload_limit_is_configurable() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        decode_packet(GOOD, received_at_ns=0, max_payload=63)
    assert excinfo.value.code is CloseCode.MESSAGE_TOO_BIG
    assert decode_packet(GOOD, received_at_ns=0, max_payload=64).frame.nbytes == 64


# ------------------------------------------------------------------------ fuzzing

hex_ids = st.integers(min_value=1, max_value=2**64 - 1)


@settings(max_examples=300)
@given(
    samples=st.integers(min_value=1, max_value=4096),
    stereo=st.booleans(),
    rate=st.integers(min_value=1, max_value=2**32 - 1),
    duration_ns=st.integers(min_value=1, max_value=2**32 - 1),
    sequence=st.integers(min_value=0, max_value=2**32 - 1),
    origin_ns=st.integers(min_value=0, max_value=2**64 - 1),
    session=st.integers(min_value=1, max_value=2**128 - 1),
    span=hex_ids,
    fill=st.integers(min_value=0, max_value=255),
)
def test_pcm_round_trip_is_lossless(  # noqa: PLR0913 -- one strategy per header field
    samples: int,
    *,
    stereo: bool,
    rate: int,
    duration_ns: int,
    sequence: int,
    origin_ns: int,
    session: int,
    span: int,
    fill: int,
) -> None:
    channels = 2 if stereo else 1
    trace = TraceContext(parse_session_id(f"{session:032x}"), parse_span_id(f"{span:016x}"))
    payload = bytes([fill]) * (samples * 2 * channels)
    original = frame(
        payload,
        rate=rate,
        channels=channels,
        duration_ns=duration_ns,
        sequence=sequence,
        origin_ns=origin_ns,
        trace=trace,
    )
    packet = encode_packet(original)
    assert len(packet) == HEADER_SIZE + len(payload)
    decoded = decode_packet(packet, received_at_ns=1, max_payload=len(payload))
    restored = decoded.frame
    assert bytes(restored.payload) == payload
    assert (restored.sample_rate_hz, restored.channels, restored.duration_ns) == (
        rate,
        channels,
        duration_ns,
    )
    assert (restored.sequence, restored.trace, decoded.sender_timestamp_ns) == (
        sequence,
        trace,
        origin_ns,
    )


@settings(max_examples=500)
@given(st.binary(max_size=200))
def test_arbitrary_bytes_never_escape_as_anything_but_protocol_error(data: bytes) -> None:
    with contextlib.suppress(ProtocolError):
        decode_packet(data, received_at_ns=0)


@settings(max_examples=500)
@given(offset=st.integers(min_value=0, max_value=HEADER_SIZE + 63), value=st.integers(0, 255))
def test_single_byte_corruption_is_rejected_or_benign(offset: int, value: int) -> None:
    corrupted = bytearray(GOOD)
    corrupted[offset] = value
    with contextlib.suppress(ProtocolError):
        decoded = decode_packet(bytes(corrupted), received_at_ns=0)
        assert decoded.frame.nbytes == 64, "a corruption that decodes must stay well-formed"


# ------------------------------------------------------------------------ stream decoding


def test_stream_decoder_reassembles_and_splits_packets() -> None:
    packets = [encode_packet(frame(bytes(32), sequence=i)) for i in range(3)]
    stream = b"".join(packets)
    decoder = PacketStreamDecoder()
    assert decoder.feed(stream[:10], received_at_ns=1) == []
    assert decoder.buffered == 10
    first = decoder.feed(stream[10:100], received_at_ns=2)
    assert [p.frame.sequence for p in first] == [0]
    rest = decoder.feed(stream[100:], received_at_ns=3)
    assert [p.frame.sequence for p in rest] == [1, 2]
    assert all(p.frame.origin_ns == 3 for p in rest)
    assert decoder.buffered == 0


@settings(max_examples=200)
@given(
    sizes=st.lists(st.integers(min_value=1, max_value=200), min_size=1, max_size=12),
    cuts=st.lists(st.integers(min_value=1, max_value=97), max_size=40),
)
def test_any_chunking_yields_the_same_packets(sizes: list[int], cuts: list[int]) -> None:
    packets = [encode_packet(frame(bytes(2 * n), sequence=i)) for i, n in enumerate(sizes)]
    stream = b"".join(packets)
    decoder = PacketStreamDecoder()
    out = []
    position = 0
    for cut in cuts:
        out += decoder.feed(stream[position : position + cut], received_at_ns=0)
        position += cut
    out += decoder.feed(stream[position:], received_at_ns=0)
    assert [p.frame.sequence for p in out] == list(range(len(sizes)))
    assert [p.frame.nbytes for p in out] == [2 * n for n in sizes]
    assert decoder.buffered == 0


def test_stream_decoder_rejects_desync_and_oversize_before_buffering_the_body() -> None:
    with pytest.raises(ProtocolError, match="magic"):
        PacketStreamDecoder().feed(b"XX" + GOOD[2:], received_at_ns=0)
    oversized = patched(GOOD[:HEADER_SIZE], 50, ">I", DEFAULT_MAX_PAYLOAD_BYTES + 1)
    decoder = PacketStreamDecoder()
    with pytest.raises(ProtocolError) as excinfo:
        decoder.feed(oversized, received_at_ns=0)
    assert excinfo.value.code is CloseCode.MESSAGE_TOO_BIG
    assert PACKET_MAGIC == b"VG"
