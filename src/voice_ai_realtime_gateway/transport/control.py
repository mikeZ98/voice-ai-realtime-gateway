"""Typed control protocol carried in WebSocket text frames (compact JSON).

Every message is an object with a ``"type"`` discriminator and *exactly* the fields
listed below. Unknown types, missing or extra fields, wrong JSON types (``true`` is not
an integer), non-finite numbers and out-of-range values are all rejected.

===================  =========  ==================================================
type                 direction  fields
===================  =========  ==================================================
``ping``             both       ``nonce`` (int)
``pong``             both       ``nonce`` (int), echoing the ping
``session.config``   client     ``codec``, ``sample_rate_hz``, ``channels``
``vad.config``       client     ``start_threshold_dbfs``, ``stop_threshold_dbfs``,
                                ``min_speech_ms``, ``hangover_ms``
``response.cancel``  client     (none): stop the agent's current response
``session.close``    client     ``reason`` (str): hang up
``playback.flush``   server     (none): drop client-side buffered agent audio
===================  =========  ==================================================
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NoReturn

from voice_ai_realtime_gateway.core.frames import OPUS_SAMPLE_RATES_HZ, AudioCodec
from voice_ai_realtime_gateway.transport.errors import CloseCode, ProtocolError
from voice_ai_realtime_gateway.vad.base import VadConfig

__all__ = [
    "MAX_NONCE",
    "MAX_SAMPLE_RATE_HZ",
    "ApplicationMessage",
    "AudioFormat",
    "CancelResponse",
    "ClientMessage",
    "Hangup",
    "Ping",
    "PlaybackFlush",
    "Pong",
    "ServerMessage",
    "SessionConfig",
    "VadConfigUpdate",
    "decode_client_message",
    "decode_server_message",
    "encode_control",
]

MAX_NONCE: Final = 2**53 - 1
"""Largest integer every JSON implementation represents exactly."""
MAX_SAMPLE_RATE_HZ: Final = 384_000
_MAX_REASON_CHARS: Final = 120
_MAX_CHANNELS: Final = 2


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Ingress audio format a client commits to."""

    codec: AudioCodec
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        """Validate the format.

        Raises:
            ValueError: on an impossible rate or channel count.
        """
        if not 1 <= self.sample_rate_hz <= MAX_SAMPLE_RATE_HZ:
            msg = f"sample_rate_hz must be in [1, {MAX_SAMPLE_RATE_HZ}], got {self.sample_rate_hz}"
            raise ValueError(msg)
        if not 1 <= self.channels <= _MAX_CHANNELS:
            msg = f"channels must be 1 or 2, got {self.channels}"
            raise ValueError(msg)
        if self.codec is AudioCodec.OPUS and self.sample_rate_hz not in OPUS_SAMPLE_RATES_HZ:
            msg = f"Opus sample rate must be one of {sorted(OPUS_SAMPLE_RATES_HZ)}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Ping:
    """Liveness probe; the receiver answers with a :class:`Pong` echoing ``nonce``."""

    nonce: int


@dataclass(frozen=True, slots=True)
class Pong:
    """Answer to a :class:`Ping`."""

    nonce: int


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Client declaration of its ingress audio format."""

    audio: AudioFormat


@dataclass(frozen=True, slots=True)
class VadConfigUpdate:
    """Client-tuned VAD hysteresis."""

    config: VadConfig


@dataclass(frozen=True, slots=True)
class CancelResponse:
    """Client request to stop the agent's current response."""


@dataclass(frozen=True, slots=True)
class Hangup:
    """Client request to end the session."""

    reason: str = ""


@dataclass(frozen=True, slots=True)
class PlaybackFlush:
    """Server instruction to drop client-side buffered agent audio (after a barge-in)."""


type ClientMessage = Ping | Pong | SessionConfig | VadConfigUpdate | CancelResponse | Hangup
type ServerMessage = Ping | Pong | PlaybackFlush
type ApplicationMessage = VadConfigUpdate | CancelResponse | Hangup
"""Client messages the transport hands to the application (the rest it handles itself)."""

type _Fields = Mapping[str, object]


# --------------------------------------------------------------------------- encoding


def _wire(message: ClientMessage | ServerMessage) -> dict[str, object]:  # noqa: PLR0911 -- one arm per message type
    if isinstance(message, Ping):
        return {"type": "ping", "nonce": message.nonce}
    if isinstance(message, Pong):
        return {"type": "pong", "nonce": message.nonce}
    if isinstance(message, SessionConfig):
        audio = message.audio
        return {
            "type": "session.config",
            "codec": audio.codec.value,
            "sample_rate_hz": audio.sample_rate_hz,
            "channels": audio.channels,
        }
    if isinstance(message, VadConfigUpdate):
        config = message.config
        return {
            "type": "vad.config",
            "start_threshold_dbfs": config.start_threshold_dbfs,
            "stop_threshold_dbfs": config.stop_threshold_dbfs,
            "min_speech_ms": config.min_speech_ms,
            "hangover_ms": config.hangover_ms,
        }
    if isinstance(message, CancelResponse):
        return {"type": "response.cancel"}
    if isinstance(message, Hangup):
        return {"type": "session.close", "reason": message.reason}
    return {"type": "playback.flush"}


def encode_control(message: ClientMessage | ServerMessage) -> str:
    """Serialise ``message`` as compact JSON."""
    return json.dumps(_wire(message), separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------- decoding


def _invalid(reason: str) -> NoReturn:
    raise ProtocolError(CloseCode.INVALID_PAYLOAD, reason)


def _reject_constant(token: str) -> NoReturn:
    msg = f"non-finite JSON constant {token}"
    raise ValueError(msg)


def _int(fields: _Fields, key: str, low: int, high: int) -> int:
    value = fields[key]
    if type(value) is not int or not low <= value <= high:
        _invalid(f"{key} must be an integer in [{low}, {high}]")
    return value


def _number(fields: _Fields, key: str) -> float:
    value = fields[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        _invalid(f"{key} must be a finite number")
    return float(value)


def _text(fields: _Fields, key: str, max_chars: int) -> str:
    value = fields[key]
    if not isinstance(value, str) or len(value) > max_chars:
        _invalid(f"{key} must be a string of at most {max_chars} characters")
    return value


def _ping(fields: _Fields) -> Ping:
    return Ping(_int(fields, "nonce", 0, MAX_NONCE))


def _pong(fields: _Fields) -> Pong:
    return Pong(_int(fields, "nonce", 0, MAX_NONCE))


def _session_config(fields: _Fields) -> SessionConfig:
    raw_codec = _text(fields, "codec", 16)
    try:
        codec = AudioCodec(raw_codec)
    except ValueError:
        raise ProtocolError(
            CloseCode.UNSUPPORTED_DATA, f"unsupported codec {raw_codec!r}"
        ) from None
    try:
        audio = AudioFormat(
            codec,
            _int(fields, "sample_rate_hz", 1, MAX_SAMPLE_RATE_HZ),
            _int(fields, "channels", 1, _MAX_CHANNELS),
        )
    except ValueError as exc:
        _invalid(str(exc))
    return SessionConfig(audio)


def _vad_config(fields: _Fields) -> VadConfigUpdate:
    try:
        config = VadConfig(
            start_threshold_dbfs=_number(fields, "start_threshold_dbfs"),
            stop_threshold_dbfs=_number(fields, "stop_threshold_dbfs"),
            min_speech_ms=_int(fields, "min_speech_ms", 1, 2**31),
            hangover_ms=_int(fields, "hangover_ms", 1, 2**31),
        )
    except ValueError as exc:
        _invalid(str(exc))
    return VadConfigUpdate(config)


def _cancel(_fields: _Fields) -> CancelResponse:
    return CancelResponse()


def _hangup(fields: _Fields) -> Hangup:
    return Hangup(_text(fields, "reason", _MAX_REASON_CHARS))


def _flush(_fields: _Fields) -> PlaybackFlush:
    return PlaybackFlush()


@dataclass(frozen=True, slots=True)
class _Schema[M]:
    fields: frozenset[str]
    parse: Callable[[_Fields], M]


_CLIENT: Final[Mapping[str, _Schema[ClientMessage]]] = MappingProxyType(
    {
        "ping": _Schema(frozenset({"nonce"}), _ping),
        "pong": _Schema(frozenset({"nonce"}), _pong),
        "session.config": _Schema(
            frozenset({"codec", "sample_rate_hz", "channels"}), _session_config
        ),
        "vad.config": _Schema(
            frozenset(
                {"start_threshold_dbfs", "stop_threshold_dbfs", "min_speech_ms", "hangover_ms"}
            ),
            _vad_config,
        ),
        "response.cancel": _Schema(frozenset(), _cancel),
        "session.close": _Schema(frozenset({"reason"}), _hangup),
    }
)

_SERVER: Final[Mapping[str, _Schema[ServerMessage]]] = MappingProxyType(
    {
        "ping": _Schema(frozenset({"nonce"}), _ping),
        "pong": _Schema(frozenset({"nonce"}), _pong),
        "playback.flush": _Schema(frozenset(), _flush),
    }
)


def _decode[M](text: str, schemas: Mapping[str, _Schema[M]]) -> M:
    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        _invalid("malformed control message")
    if not isinstance(document, dict):
        _invalid("control message must be a JSON object")
    kind = document.get("type")
    if not isinstance(kind, str):
        _invalid("control message needs a string 'type'")
    schema = schemas.get(kind)
    if schema is None:
        raise ProtocolError(CloseCode.UNSUPPORTED_DATA, f"unsupported control message {kind!r}")
    if document.keys() - {"type"} != schema.fields:
        _invalid(f"{kind} expects exactly the fields {sorted(schema.fields)}")
    return schema.parse(document)


def decode_client_message(text: str) -> ClientMessage:
    """Parse a client-to-server control message.

    Raises:
        ProtocolError: ``UNSUPPORTED_DATA`` for an unknown type or codec,
            ``INVALID_PAYLOAD`` for anything malformed or out of range.
    """
    return _decode(text, _CLIENT)


def decode_server_message(text: str) -> ServerMessage:
    """Parse a server-to-client control message (used by clients and tests).

    Raises:
        ProtocolError: as for :func:`decode_client_message`.
    """
    return _decode(text, _SERVER)
