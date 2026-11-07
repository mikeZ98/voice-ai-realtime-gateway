"""Control protocol: golden encoding, round trips, strict schema validation, fuzzing."""

from __future__ import annotations

import contextlib
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.transport.control import (
    MAX_NONCE,
    AudioFormat,
    CancelResponse,
    ClientMessage,
    Hangup,
    Ping,
    PlaybackFlush,
    Pong,
    ServerMessage,
    SessionConfig,
    VadConfigUpdate,
    decode_client_message,
    decode_server_message,
    encode_control,
)
from voice_ai_realtime_gateway.transport.errors import (
    MAX_CLOSE_REASON_BYTES,
    CloseCode,
    ProtocolError,
    truncate_reason,
)
from voice_ai_realtime_gateway.vad.base import VadConfig

CLIENT_MESSAGES: list[ClientMessage] = [
    Ping(1),
    Pong(MAX_NONCE),
    SessionConfig(AudioFormat(AudioCodec.PCM16, 16_000, 1)),
    SessionConfig(AudioFormat(AudioCodec.OPUS, 48_000, 2)),
    VadConfigUpdate(VadConfig(start_threshold_dbfs=-30.5, stop_threshold_dbfs=-42.0)),
    CancelResponse(),
    Hangup("user pressed end"),
]
SERVER_MESSAGES: list[ServerMessage] = [Ping(0), Pong(42), PlaybackFlush()]


def test_golden_encodings() -> None:
    assert encode_control(Ping(7)) == '{"type":"ping","nonce":7}'
    assert encode_control(PlaybackFlush()) == '{"type":"playback.flush"}'
    assert encode_control(CancelResponse()) == '{"type":"response.cancel"}'
    assert encode_control(SessionConfig(AudioFormat(AudioCodec.PCM16, 16_000, 1))) == (
        '{"type":"session.config","codec":"pcm16","sample_rate_hz":16000,"channels":1}'
    )


@pytest.mark.parametrize("message", CLIENT_MESSAGES)
def test_client_messages_round_trip(message: ClientMessage) -> None:
    assert decode_client_message(encode_control(message)) == message


@pytest.mark.parametrize("message", SERVER_MESSAGES)
def test_server_messages_round_trip(message: ServerMessage) -> None:
    assert decode_server_message(encode_control(message)) == message


@given(
    start=st.floats(min_value=-120.0, max_value=0.0),
    gap=st.floats(min_value=0.0, max_value=120.0),
    min_speech=st.integers(min_value=1, max_value=2_000),
    hangover=st.integers(min_value=1, max_value=10_000),
)
def test_vad_config_round_trips_for_every_valid_value(
    start: float, gap: float, min_speech: int, hangover: int
) -> None:
    stop = max(-120.0, start - gap)
    config = VadConfig(
        start_threshold_dbfs=start,
        stop_threshold_dbfs=stop,
        min_speech_ms=min_speech,
        hangover_ms=hangover,
    )
    assert decode_client_message(encode_control(VadConfigUpdate(config))) == VadConfigUpdate(config)


def test_integer_thresholds_are_accepted_as_numbers() -> None:
    text = (
        '{"type":"vad.config","start_threshold_dbfs":-30,"stop_threshold_dbfs":-40,'
        '"min_speech_ms":40,"hangover_ms":200}'
    )
    message = decode_client_message(text)
    assert isinstance(message, VadConfigUpdate)
    assert message.config.start_threshold_dbfs == -30.0


VAD_BASE = {
    "type": "vad.config",
    "start_threshold_dbfs": -30.0,
    "stop_threshold_dbfs": -40.0,
    "min_speech_ms": 40,
    "hangover_ms": 200,
}


@pytest.mark.parametrize(
    ("text", "code", "match"),
    [
        ("{", CloseCode.INVALID_PAYLOAD, "malformed"),
        ('{"type":"ping","nonce":NaN}', CloseCode.INVALID_PAYLOAD, "malformed"),
        ('{"type":"ping","nonce":Infinity}', CloseCode.INVALID_PAYLOAD, "malformed"),
        ("[" * 100_000, CloseCode.INVALID_PAYLOAD, "malformed"),
        ("[1]", CloseCode.INVALID_PAYLOAD, "JSON object"),
        ('{"nonce":1}', CloseCode.INVALID_PAYLOAD, "string 'type'"),
        ('{"type":7}', CloseCode.INVALID_PAYLOAD, "string 'type'"),
        ('{"type":"teleport"}', CloseCode.UNSUPPORTED_DATA, "unsupported control"),
        ('{"type":"playback.flush"}', CloseCode.UNSUPPORTED_DATA, "unsupported control"),
        ('{"type":"ping"}', CloseCode.INVALID_PAYLOAD, "exactly the fields"),
        ('{"type":"ping","nonce":1,"x":2}', CloseCode.INVALID_PAYLOAD, "exactly the fields"),
        ('{"type":"ping","nonce":true}', CloseCode.INVALID_PAYLOAD, "integer"),
        ('{"type":"ping","nonce":1.0}', CloseCode.INVALID_PAYLOAD, "integer"),
        ('{"type":"ping","nonce":-1}', CloseCode.INVALID_PAYLOAD, "integer"),
        (f'{{"type":"pong","nonce":{MAX_NONCE + 1}}}', CloseCode.INVALID_PAYLOAD, "integer"),
        ('{"type":"session.close","reason":5}', CloseCode.INVALID_PAYLOAD, "string"),
        ('{"type":"session.close","reason":"' + "x" * 121 + '"}', CloseCode.INVALID_PAYLOAD, "120"),
        (
            '{"type":"session.config","codec":"aac","sample_rate_hz":16000,"channels":1}',
            CloseCode.UNSUPPORTED_DATA,
            "codec",
        ),
        (
            '{"type":"session.config","codec":1,"sample_rate_hz":16000,"channels":1}',
            CloseCode.INVALID_PAYLOAD,
            "codec",
        ),
        (
            '{"type":"session.config","codec":"opus","sample_rate_hz":44100,"channels":1}',
            CloseCode.INVALID_PAYLOAD,
            "Opus",
        ),
        (
            '{"type":"session.config","codec":"pcm16","sample_rate_hz":16000,"channels":3}',
            CloseCode.INVALID_PAYLOAD,
            "channels",
        ),
        (
            json.dumps({**VAD_BASE, "stop_threshold_dbfs": -10.0}),
            CloseCode.INVALID_PAYLOAD,
            "stop <= start",
        ),
        (
            json.dumps({**VAD_BASE, "start_threshold_dbfs": "loud"}),
            CloseCode.INVALID_PAYLOAD,
            "finite number",
        ),
        (
            json.dumps({**VAD_BASE, "start_threshold_dbfs": True}),
            CloseCode.INVALID_PAYLOAD,
            "finite number",
        ),
        (
            json.dumps({**VAD_BASE}).replace("-30.0", "-1e999"),
            CloseCode.INVALID_PAYLOAD,
            "finite number",
        ),
    ],
)
def test_invalid_client_messages_map_to_close_codes(text: str, code: CloseCode, match: str) -> None:
    with pytest.raises(ProtocolError, match=match) as excinfo:
        decode_client_message(text)
    assert excinfo.value.code is code


def test_server_decoder_rejects_client_only_messages() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        decode_server_message(encode_control(Hangup("bye")))
    assert excinfo.value.code is CloseCode.UNSUPPORTED_DATA


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((AudioCodec.PCM16, 0, 1), "sample_rate_hz"),
        ((AudioCodec.PCM16, 384_001, 1), "sample_rate_hz"),
        ((AudioCodec.PCM16, 16_000, 0), "channels"),
        ((AudioCodec.OPUS, 22_050, 1), "Opus"),
    ],
)
def test_audio_format_is_validated(args: tuple[AudioCodec, int, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AudioFormat(*args)


@settings(max_examples=500)
@given(st.text(max_size=200))
def test_arbitrary_text_never_escapes_as_anything_but_protocol_error(text: str) -> None:
    with contextlib.suppress(ProtocolError):
        decode_client_message(text)


@settings(max_examples=300)
@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
        lambda children: (
            st.lists(children, max_size=4)
            | st.dictionaries(st.sampled_from(["type", "nonce", "reason", "codec"]), children)
        ),
        max_leaves=12,
    )
)
def test_arbitrary_json_documents_are_rejected_cleanly(document: object) -> None:
    with contextlib.suppress(ProtocolError):
        decode_client_message(json.dumps(document))


def test_close_reasons_are_truncated_on_a_utf8_boundary() -> None:
    assert truncate_reason("short") == "short"
    clipped = truncate_reason("é" * 100)  # 2 bytes each
    assert len(clipped.encode()) <= MAX_CLOSE_REASON_BYTES
    assert clipped == "é" * 61
    error = ProtocolError(CloseCode.POLICY_VIOLATION, "x" * 500)
    assert len(error.reason) == MAX_CLOSE_REASON_BYTES
    assert "POLICY_VIOLATION (1008)" in str(error)


def test_transport_closed_error_carries_the_close_handshake() -> None:
    from voice_ai_realtime_gateway.transport.errors import (  # noqa: PLC0415 -- local to the contract
        TransportClosedError,
    )

    error = TransportClosedError(CloseCode.ABNORMAL, "network lost")
    assert (error.code, error.reason) == (1006, "network lost")
    assert "connection closed (1006)" in str(error)
    assert TransportClosedError(1000).reason == ""
