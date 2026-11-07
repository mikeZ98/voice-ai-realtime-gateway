"""Binary audio packet framing.

Wire format, version 1: a fixed 54-byte big-endian header followed by the payload.

======  ====  ===============  ==============================================
offset  size  field            notes
======  ====  ===============  ==============================================
0       2     magic            ``b"VG"``
2       1     version          ``1``
3       1     codec            ``1`` = PCM16, ``2`` = Opus
4       1     channels         ``1`` or ``2``
5       1     flags            reserved, must be ``0``
6       4     sample_rate_hz   u32
10      4     duration_ns      u32 media duration of the packet
14      4     sequence         u32 per-span sequence number
18      8     timestamp_ns     u64 sender-clock timestamp (opaque to the receiver)
26      16    session_id       ``X-Session-ID`` (128 bit)
42      8     span_id          ``X-Audio-Span-ID`` (64 bit)
50      4     payload_length   u32
54      n     payload          PCM16 little-endian samples or one Opus packet
======  ====  ===============  ==============================================

Clocks of different machines are not comparable, so the decoder never trusts the
sender's timestamp for latency: it stamps ``origin_ns`` with the *receiver's* clock and
returns the sender timestamp separately. Decoding is zero-copy: the frame payload is a
read-only view into the received message.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.telemetry.ids import TraceContext, parse_session_id, parse_span_id
from voice_ai_realtime_gateway.transport.errors import CloseCode, ProtocolError

if TYPE_CHECKING:
    from collections.abc import Buffer

__all__ = [
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "HEADER_SIZE",
    "PACKET_MAGIC",
    "PACKET_VERSION",
    "DecodedPacket",
    "PacketStreamDecoder",
    "decode_packet",
    "encode_packet",
]

PACKET_MAGIC: Final = b"VG"
PACKET_VERSION: Final = 1
DEFAULT_MAX_PAYLOAD_BYTES: Final = 32 * 1024
"""Large enough for 80 ms of 48 kHz stereo PCM16; bounds per-packet memory."""

_HEADER: Final = struct.Struct(">2sBBBBIIIQ16s8sI")
HEADER_SIZE: Final = _HEADER.size
_LENGTH_OFFSET: Final = HEADER_SIZE - 4
_U32_MAX: Final = 0xFFFF_FFFF
_U64_MAX: Final = 0xFFFF_FFFF_FFFF_FFFF

_CODEC_IDS: Final = MappingProxyType({AudioCodec.PCM16: 1, AudioCodec.OPUS: 2})
_CODECS_BY_ID: Final = MappingProxyType({wire: codec for codec, wire in _CODEC_IDS.items()})


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedPacket:
    """A decoded packet: the frame (receiver clock) and the sender's timestamp."""

    frame: AudioFrame
    sender_timestamp_ns: int


def _bounded(name: str, value: int, maximum: int) -> int:
    if not 0 <= value <= maximum:
        msg = f"{name}={value} does not fit the packet header (0..{maximum})"
        raise ValueError(msg)
    return value


def encode_packet(frame: AudioFrame) -> bytes:
    """Serialise ``frame`` as one packet (one allocation: header + payload).

    Raises:
        ValueError: if a numeric field does not fit its header slot.
    """
    header = _HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        _CODEC_IDS[frame.codec],
        frame.channels,
        0,
        _bounded("sample_rate_hz", frame.sample_rate_hz, _U32_MAX),
        _bounded("duration_ns", frame.duration_ns, _U32_MAX),
        _bounded("sequence", frame.sequence, _U32_MAX),
        _bounded("origin_ns", frame.origin_ns, _U64_MAX),
        bytes.fromhex(frame.trace.session_id),
        bytes.fromhex(frame.trace.span_id),
        frame.nbytes,
    )
    return header + frame.payload


def decode_packet(
    data: Buffer, *, received_at_ns: int, max_payload: int = DEFAULT_MAX_PAYLOAD_BYTES
) -> DecodedPacket:
    """Parse exactly one packet without copying its payload.

    Raises:
        ProtocolError: ``MESSAGE_TOO_BIG`` for an oversized payload,
            ``UNSUPPORTED_DATA`` for an unknown version or codec, and
            ``INVALID_PAYLOAD`` for anything malformed.
    """
    view = memoryview(data).cast("B")
    if view.nbytes < HEADER_SIZE:
        raise ProtocolError(CloseCode.INVALID_PAYLOAD, "truncated packet header")
    (
        magic,
        version,
        codec_id,
        channels,
        flags,
        sample_rate_hz,
        duration_ns,
        sequence,
        timestamp_ns,
        session_raw,
        span_raw,
        length,
    ) = _HEADER.unpack_from(view)
    if magic != PACKET_MAGIC:
        raise ProtocolError(CloseCode.INVALID_PAYLOAD, "bad packet magic")
    if version != PACKET_VERSION:
        raise ProtocolError(CloseCode.UNSUPPORTED_DATA, f"unsupported packet version {version}")
    codec = _CODECS_BY_ID.get(codec_id)
    if codec is None:
        raise ProtocolError(CloseCode.UNSUPPORTED_DATA, f"unsupported codec id {codec_id}")
    if flags:
        raise ProtocolError(CloseCode.INVALID_PAYLOAD, "reserved packet flags set")
    if length > max_payload:
        raise ProtocolError(CloseCode.MESSAGE_TOO_BIG, f"payload of {length} B exceeds limit")
    if view.nbytes != HEADER_SIZE + length:
        raise ProtocolError(CloseCode.INVALID_PAYLOAD, "packet length mismatch")
    try:
        trace = TraceContext(parse_session_id(session_raw.hex()), parse_span_id(span_raw.hex()))
        frame = AudioFrame.wrap(
            view[HEADER_SIZE:],
            codec=codec,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            duration_ns=duration_ns,
            sequence=sequence,
            origin_ns=received_at_ns,
            trace=trace,
        )
    except ValueError as exc:
        raise ProtocolError(CloseCode.INVALID_PAYLOAD, f"invalid packet: {exc}") from exc
    return DecodedPacket(frame=frame, sender_timestamp_ns=timestamp_ns)


@final
class PacketStreamDecoder:
    """Incremental decoder for byte-stream transports (TCP, pipes, memory streams).

    Memory is bounded by one maximum-size packet plus the bytes of the latest chunk:
    the length field is validated before the body is awaited. Any framing error is
    fatal, because a byte stream cannot be resynchronised safely.
    """

    __slots__ = ("_buffer", "_max_payload")

    def __init__(self, *, max_payload: int = DEFAULT_MAX_PAYLOAD_BYTES) -> None:
        """Create an empty decoder accepting payloads up to ``max_payload`` bytes."""
        self._buffer = bytearray()
        self._max_payload = max_payload

    @property
    def buffered(self) -> int:
        """Bytes held while waiting for the rest of a packet."""
        return len(self._buffer)

    def feed(self, chunk: Buffer, *, received_at_ns: int) -> list[DecodedPacket]:
        """Append ``chunk`` and return every packet it completed, in order.

        Raises:
            ProtocolError: on a framing violation (see :func:`decode_packet`).
        """
        buffer = self._buffer
        buffer += memoryview(chunk).cast("B")
        packets: list[DecodedPacket] = []
        while len(buffer) >= HEADER_SIZE:
            if buffer[:2] != PACKET_MAGIC:
                raise ProtocolError(CloseCode.INVALID_PAYLOAD, "bad packet magic")
            (length,) = struct.unpack_from(">I", buffer, _LENGTH_OFFSET)
            if length > self._max_payload:
                raise ProtocolError(
                    CloseCode.MESSAGE_TOO_BIG, f"payload of {length} B exceeds limit"
                )
            total = HEADER_SIZE + length
            if len(buffer) < total:
                break
            with memoryview(buffer) as whole:
                packet = bytes(whole[:total])
            del buffer[:total]
            packets.append(
                decode_packet(packet, received_at_ns=received_at_ns, max_payload=self._max_payload)
            )
        return packets
