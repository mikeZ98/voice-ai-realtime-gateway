"""Deepgram adapter: parser contract, fuzzing, streaming, reconnect with replay, failures."""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from provider_support import FakeProvider, instant_sleep
from voice_ai_realtime_gateway.adapters.deepgram import (
    CLOSE_STREAM,
    DeepgramMetadata,
    DeepgramResult,
    DeepgramStreamingASR,
    parse_deepgram_message,
)
from voice_ai_realtime_gateway.adapters.resilience import ErrorKind, RetryPolicy, UpstreamError
from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.telemetry.events import ProviderError
from voice_ai_realtime_gateway.transport.errors import CloseCode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from voice_ai_realtime_gateway.inference.ports import TranscriptChunk
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder
    from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket

pytestmark = pytest.mark.usefixtures("strict_loop")

FRAME_S = 0.02


def results(
    words: list[tuple[str, float, float]], *, final: bool, start: float, duration: float
) -> str:
    return json.dumps(
        {
            "type": "Results",
            "is_final": final,
            "speech_final": final,
            "start": start,
            "duration": duration,
            "channel": {
                "alternatives": [
                    {
                        "transcript": " ".join(w for w, _, _ in words),
                        "confidence": 0.97,
                        "words": [
                            {
                                "word": w.lower(),
                                "punctuated_word": w,
                                "start": s,
                                "end": e,
                                "confidence": 0.9,
                            }
                            for w, s, e in words
                        ],
                    }
                ]
            },
        }
    )


# ------------------------------------------------------------------------ parser contract


def test_parses_results_and_prefers_punctuated_words() -> None:
    result = parse_deepgram_message(
        results([("Hello,", 0.1, 0.4)], final=True, start=0.0, duration=1.0)
    )
    assert isinstance(result, DeepgramResult)
    assert (result.transcript, result.is_final, result.confidence) == ("Hello,", True, 0.97)
    assert [(w.word, w.start_s, w.end_s) for w in result.words] == [("Hello,", 0.1, 0.4)]
    plain = json.loads(results([("x", 0.0, 0.1)], final=False, start=0, duration=0.1))
    del plain["channel"]["alternatives"][0]["words"][0]["punctuated_word"]
    parsed = parse_deepgram_message(json.dumps(plain))
    assert isinstance(parsed, DeepgramResult)
    assert parsed.words[0].word == "x"
    del plain["channel"]["alternatives"][0]["words"]
    assert parse_deepgram_message(json.dumps(plain)) is not None, "words are optional"


@pytest.mark.parametrize(
    "message",
    [
        '{"type":"SpeechStarted","timestamp":1.0}',
        '{"type":"UtteranceEnd","last_word_end":2.0}',
        '{"type":"FutureType"}',
        "{}",
    ],
)
def test_non_transcript_messages_are_ignored(message: str) -> None:
    assert parse_deepgram_message(message) is None


def mutate(path: tuple[str | int, ...], value: object) -> str:
    document = json.loads(results([("a", 0.0, 0.1)], final=True, start=0.0, duration=0.1))
    target = document
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return json.dumps(document)


@pytest.mark.parametrize(
    "message",
    [
        "not json",
        "[1, 2]",
        "[" * 50_000,
        mutate(("channel",), "x"),
        mutate(("channel", "alternatives"), []),
        mutate(("channel", "alternatives", 0), 5),
        mutate(("channel", "alternatives", 0, "transcript"), 7),
        mutate(("is_final",), "yes"),
        mutate(("channel", "alternatives", 0, "words"), {}),
        mutate(("channel", "alternatives", 0, "words", 0), "w"),
        mutate(("channel", "alternatives", 0, "words", 0, "punctuated_word"), 3),
        mutate(("channel", "alternatives", 0, "words", 0, "start"), 0.5),
        mutate(("channel", "alternatives", 0, "words", 0, "end"), "late"),
        mutate(("channel", "alternatives", 0, "confidence"), value=True),
        mutate(("channel", "alternatives", 0, "confidence"), value=1.5),
        mutate(("channel", "alternatives", 0, "confidence"), value=-0.25),
        mutate(("start",), -1),
        mutate(("duration",), KeyError),
        (
            '{"type":"Results","start":1e999,"duration":1,"is_final":true,'
            '"channel":{"alternatives":[{"transcript":"","confidence":1}]}}'
        ),
    ],
)
def test_malformed_results_are_protocol_errors(message: str) -> None:
    with pytest.raises(UpstreamError) as excinfo:
        parse_deepgram_message(message)
    assert excinfo.value.kind is ErrorKind.PROTOCOL
    assert excinfo.value.provider == "deepgram"


@settings(max_examples=400)
@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=6),
        lambda inner: (
            st.lists(inner, max_size=3)
            | st.dictionaries(
                st.sampled_from(
                    [
                        "type",
                        "channel",
                        "alternatives",
                        "transcript",
                        "words",
                        "is_final",
                        "start",
                        "duration",
                        "confidence",
                        "word",
                        "end",
                    ]
                ),
                inner,
                max_size=6,
            )
        ),
        max_leaves=20,
    ).map(lambda doc: {**doc, "type": "Results"} if isinstance(doc, dict) else doc)
)
def test_arbitrary_documents_never_escape_as_anything_but_protocol_errors(document: object) -> None:
    with contextlib.suppress(UpstreamError):
        parse_deepgram_message(json.dumps(document))


# ------------------------------------------------------------------------ streaming


def word_frame(trace: TraceContext, index: int) -> AudioFrame:
    payload = struct.pack("<h", index) + bytes(638)
    return AudioFrame.wrap(
        payload,
        codec=AudioCodec.PCM16,
        sample_rate_hz=16_000,
        channels=1,
        duration_ns=20_000_000,
        sequence=index,
        origin_ns=0,
        trace=trace,
    )


async def frames(trace: TraceContext, count: int) -> AsyncIterator[AudioFrame]:
    for index in range(count):
        yield word_frame(trace, index)


async def apply_chaos(ws: MemoryWebSocket, action: str) -> bool:
    """Inject one failure; ``True`` when the connection is gone."""
    if action == "sever":
        await ws.sever()
        return True
    if action == "close_normal":
        await ws.close(CloseCode.NORMAL)
        return True
    await ws.send("{this is not json" if action == "garbage" else b"\x00\x01")
    return False


def deepgram_server(
    *, final_every: int = 4, chaos: dict[int, tuple[str, int]] | None = None
) -> FakeProvider:
    """Live endpoint double: word ``w<n>`` per frame (n read from the payload)."""
    plan = chaos or {}
    received: list[list[str]] = []

    async def handler(ws: MemoryWebSocket, connection: int) -> None:
        received.append([])
        segment: list[tuple[str, float, float]] = []
        position = 0.0
        segment_start = 0.0
        while True:
            message = await ws.recv()
            if message == CLOSE_STREAM:
                if segment:
                    await ws.send(
                        results(
                            segment,
                            final=True,
                            start=segment_start,
                            duration=position - segment_start,
                        )
                    )
                await ws.send('{"type":"Metadata"}')
                await ws.close(CloseCode.NORMAL)
                return
            assert isinstance(message, bytes)
            word = f"w{struct.unpack_from('<h', message)[0]}"
            received[connection].append(word)
            segment.append((word, position, position + FRAME_S))
            position = round(position + FRAME_S, 6)
            action = plan.get(connection)
            if (
                action is not None
                and len(received[connection]) == action[1]
                and await apply_chaos(ws, action[0])
            ):
                return
            await ws.send(
                results(
                    segment, final=False, start=segment_start, duration=position - segment_start
                )
            )
            if len(segment) == final_every:
                await ws.send(
                    results(
                        segment, final=True, start=segment_start, duration=position - segment_start
                    )
                )
                segment, segment_start = [], position

    provider = FakeProvider(handler)
    provider.received = received  # type: ignore[attr-defined]
    return provider


def adapter(
    provider: FakeProvider, recorder: FlightRecorder, trace: TraceContext, **kwargs: object
) -> DeepgramStreamingASR:
    return DeepgramStreamingASR(
        provider,
        sink=recorder,
        trace=trace,
        sleep=instant_sleep,
        jitter=lambda: 0.5,
        **kwargs,  # type: ignore[arg-type]
    )


async def collect(
    asr: DeepgramStreamingASR, trace: TraceContext, count: int
) -> list[TranscriptChunk]:
    return [chunk async for chunk in asr.transcribe(frames(trace, count))]


def final_words(chunks: list[TranscriptChunk]) -> list[tuple[str, int, int]]:
    return [(w.word, w.start_ns, w.end_ns) for c in chunks if c.is_final for w in c.words]


def expected(count: int) -> list[tuple[str, int, int]]:
    return [(f"w{i}", i * 20_000_000, (i + 1) * 20_000_000) for i in range(count)]


async def test_streams_interim_and_final_results_with_absolute_timings(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = deepgram_server()
    chunks = await collect(adapter(provider, recorder, trace), trace, 10)
    assert final_words(chunks) == expected(10)
    assert [c.text for c in chunks if c.is_final] == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    assert any(not c.is_final for c in chunks)
    assert all(c.trace == trace for c in chunks)
    assert provider.connections == 1
    await provider.finish()


@pytest.mark.parametrize("action", ["sever", "garbage", "close_normal"])
async def test_upstream_drop_reconnects_and_replays_unfinalised_audio(
    recorder: FlightRecorder, trace: TraceContext, action: str
) -> None:
    provider = deepgram_server(chaos={0: (action, 6)})
    chunks = await collect(adapter(provider, recorder, trace), trace, 10)
    assert final_words(chunks) == expected(10), "every word exactly once, timings absolute"
    received = provider.received  # type: ignore[attr-defined]
    if action != "sever":  # network loss also drops the w0-w3 final in flight
        assert received[1][0] == "w4", "only audio without a received final is replayed"
    assert provider.connections == 2
    (error,) = recorder.of_type(ProviderError)
    assert (error.provider, error.fatal) == ("deepgram", False)
    assert error.kind == ("protocol" if action == "garbage" else "transient")
    await provider.finish()


async def test_binary_server_messages_are_ignored(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = deepgram_server(chaos={0: ("binary", 2)})
    chunks = await collect(adapter(provider, recorder, trace), trace, 4)
    assert final_words(chunks) == expected(4)
    await provider.finish()


async def test_replay_memory_is_bounded(recorder: FlightRecorder, trace: TraceContext) -> None:
    provider = deepgram_server(final_every=100, chaos={0: ("sever", 6)})
    chunks = await collect(adapter(provider, recorder, trace, replay_limit_frames=2), trace, 8)
    received = provider.received  # type: ignore[attr-defined]
    assert received[1] == ["w6", "w7"], "only the newest 2 unfinalised frames are kept"
    assert [w for w, _, _ in final_words(chunks)] == ["w6", "w7"]
    timings = final_words(chunks)
    assert timings[0][1] == 6 * 20_000_000, "timings stay absolute after a lossy replay"
    await provider.finish()


async def test_auth_failure_at_handshake_is_fatal(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = FakeProvider(deepgram_server().handler, UpstreamError(ErrorKind.AUTH, "401"))
    with pytest.raises(UpstreamError) as excinfo:
        await collect(adapter(provider, recorder, trace), trace, 3)
    assert excinfo.value.kind is ErrorKind.AUTH
    assert provider.attempts == 1


async def test_repeated_drops_exhaust_the_reconnect_budget(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = deepgram_server(chaos=dict.fromkeys(range(5), ("sever", 1)))
    asr = adapter(provider, recorder, trace, policy=RetryPolicy(max_reconnects=2))
    with pytest.raises(UpstreamError, match="reconnect budget spent"):
        await collect(asr, trace, 5)
    assert provider.connections == 3
    assert recorder.of_type(ProviderError)[-1].fatal
    await provider.finish()


async def test_closing_the_stream_early_releases_the_upstream_connection(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = deepgram_server()
    asr = adapter(provider, recorder, trace)
    async with contextlib.aclosing(asr.transcribe(frames(trace, 50))) as chunks:
        async for _ in chunks:
            break  # e.g. the call ended mid-utterance
    await provider.finish()
    assert provider.servers[0].closed


def test_metadata_marks_the_end_of_stream() -> None:
    assert parse_deepgram_message('{"type":"Metadata","request_id":"r"}') == DeepgramMetadata()


async def test_normal_close_after_a_complete_flush_without_metadata_ends_cleanly(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def handler(ws: MemoryWebSocket, _connection: int) -> None:
        position = 0.0
        while (message := await ws.recv()) != CLOSE_STREAM:
            assert isinstance(message, bytes)
            word = f"w{struct.unpack_from('<h', message)[0]}"
            await ws.send(
                results(
                    [(word, position, position + FRAME_S)],
                    final=True,
                    start=position,
                    duration=FRAME_S,
                )
            )
            position = round(position + FRAME_S, 6)
        await ws.close(CloseCode.NORMAL)  # no Metadata, but every frame was finalised

    provider = FakeProvider(handler)
    chunks = await collect(adapter(provider, recorder, trace), trace, 3)
    assert final_words(chunks) == expected(3)
    assert provider.connections == 1, "no needless reconnect"
    await provider.finish()


async def paced(trace: TraceContext, count: int) -> AsyncIterator[AudioFrame]:
    for index in range(count):
        await asyncio.sleep(0)  # real audio arrives over time
        yield word_frame(trace, index)


async def test_drop_while_sending_audio_recovers(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = deepgram_server(chaos={0: ("sever", 1)})
    asr = adapter(provider, recorder, trace)
    chunks = [chunk async for chunk in asr.transcribe(paced(trace, 6))]
    assert final_words(chunks) == expected(6)
    assert provider.connections == 2
    await provider.finish()
