"""Cartesia adapter: framing, alignment, contexts, chaos, parser contract and fuzzing."""

from __future__ import annotations

import base64
import contextlib
import itertools
import json
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from provider_support import FakeProvider, instant_sleep
from voice_ai_realtime_gateway.adapters.cartesia import (
    AudioChunk,
    CartesiaStreamingTTS,
    ContextDone,
    ContextFailed,
    WordTimestamps,
    _Framer,
    classify_status,
    parse_cartesia_message,
)
from voice_ai_realtime_gateway.adapters.resilience import ErrorKind, RetryPolicy, UpstreamError
from voice_ai_realtime_gateway.inference.ports import FrameStamper
from voice_ai_realtime_gateway.telemetry.events import ProviderError
from voice_ai_realtime_gateway.telemetry.ids import TraceContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from voice_ai_realtime_gateway.inference.ports import SynthesizedAudio
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder
    from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket

pytestmark = pytest.mark.usefixtures("strict_loop")

BYTES_PER_CHAR = 320  # 10 ms of 16 kHz mono PCM16 per character
TEXT = "Hi there, friend."


def pcm_for(text: str) -> bytes:
    return bytes((i * 7) % 256 for i in range(len(text) * BYTES_PER_CHAR))


def chunk(context: str, data: bytes) -> str:
    return json.dumps(
        {
            "type": "chunk",
            "context_id": context,
            "data": base64.b64encode(data).decode(),
            "done": False,
        }
    )


def timestamps(context: str, text: str) -> str:
    words, ends, cursor = [], [], 0
    for raw in text.split():
        word = raw.strip(",.!?")
        start = text.index(word, cursor)
        cursor = start + len(word)
        words.append(word)
        ends.append(cursor * 0.01)  # each character takes 10 ms
    return json.dumps(
        {
            "type": "timestamps",
            "context_id": context,
            "word_timestamps": {"words": words, "start": [0.0] * len(words), "end": ends},
        }
    )


def done(context: str) -> str:
    return json.dumps({"type": "done", "context_id": context, "done": True})


def failure(context: str | None, status: int, message: str = "boom") -> str:
    return json.dumps(
        {
            "type": "error",
            "context_id": context,
            "status_code": status,
            "error": message,
            "done": True,
        }
    )


type Script = Callable[[MemoryWebSocket, int, int, dict[str, object]], Awaitable[bool]]


async def synthesize_all(ws: MemoryWebSocket, _c: int, _n: int, request: dict[str, object]) -> bool:
    context, text = str(request["context_id"]), str(request["transcript"])
    pcm = pcm_for(text)
    await ws.send(chunk(context, pcm[:1000]))
    await ws.send(timestamps(context, text))  # alignment arrives after the first audio
    for start in range(1000, len(pcm), 1000):
        await ws.send(chunk(context, pcm[start : start + 1000]))
    await ws.send(done(context))
    return True


def cartesia(script: Script = synthesize_all, *refusals: BaseException) -> FakeProvider:
    log: list[dict[str, object]] = []

    async def handler(ws: MemoryWebSocket, connection: int) -> None:
        number = 0
        while True:
            request = json.loads(await ws.recv())
            log.append(request)
            if request.get("cancel"):
                continue
            if not await script(ws, connection, number, request):
                return
            number += 1

    provider = FakeProvider(handler, *refusals)
    provider.log = log  # type: ignore[attr-defined]
    return provider


def tts(
    provider: FakeProvider, recorder: FlightRecorder, trace: TraceContext, **kwargs: object
) -> CartesiaStreamingTTS:
    return CartesiaStreamingTTS(
        provider, sink=recorder, trace=trace, voice_id="voice-1",
        sleep=instant_sleep, jitter=lambda: 0.5, **kwargs,  # type: ignore[arg-type]
    )  # fmt: skip


async def speak(
    adapter: CartesiaStreamingTTS, trace: TraceContext, text: str = TEXT
) -> list[SynthesizedAudio]:
    return [audio async for audio in adapter.synthesize(text, FrameStamper(trace))]


# ------------------------------------------------------------------------ framing & alignment


async def test_frames_are_exact_zero_copy_and_conservatively_aligned(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = cartesia()
    adapter = tts(provider, recorder, trace)
    audio = await speak(adapter, trace)
    sizes = [a.frame.nbytes for a in audio]
    assert sizes == [640] * 8 + [320], "17 chars x 10 ms = 8 full 20 ms frames + 10 ms"
    assert b"".join(bytes(a.frame.payload) for a in audio) == pcm_for(TEXT)
    assert [a.frame.sequence for a in audio] == list(range(9))
    assert audio[-1].frame.duration_ns == 10_000_000
    assert all(a.frame.trace == trace for a in audio)
    # frame 0 precedes the timestamps (0); "Hi" ends at 20 ms (2); "there," at 80 ms (9);
    # "friend." at 160 ms (17); the remainder at done reports the full unit.
    assert [a.text_offset for a in audio] == [0, 2, 2, 9, 9, 9, 9, 17, 17]
    # 1000-byte chunks vs 640-byte frames: frames 0, 2 and 5 lie wholly inside one chunk
    # and must be zero-copy views of it; the others straddle chunks and are assembled.
    sliced = [i for i, a in enumerate(audio) if len(a.frame.payload.obj) > a.frame.nbytes]  # type: ignore[arg-type]
    assert sliced == [0, 2, 5]
    (request,) = provider.log  # type: ignore[attr-defined]
    assert request["output_format"] == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
    }
    assert (request["transcript"], request["add_timestamps"], request["continue"]) == (
        TEXT,
        True,
        False,
    )
    assert request["voice"] == {"mode": "id", "id": "voice-1"}
    await adapter.aclose()
    await adapter.aclose()
    await provider.finish()


async def test_connection_is_reused_and_foreign_contexts_ignored(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def noisy(
        ws: MemoryWebSocket, connection: int, number: int, request: dict[str, object]
    ) -> bool:
        await ws.send(chunk("someone-else", b"\x01\x02" * 400))
        await ws.send(b"\x00")
        await ws.send(
            json.dumps({"type": "phoneme_timestamps", "context_id": request["context_id"]})
        )
        await ws.send('{"type":"flush_done"}')
        return await synthesize_all(ws, connection, number, request)

    provider = cartesia(noisy)
    adapter = tts(provider, recorder, trace)
    first, second = await speak(adapter, trace, "One."), await speak(adapter, trace, "Two.")
    assert b"".join(bytes(a.frame.payload) for a in first) == pcm_for("One.")
    assert len(second) == len(first)
    assert provider.connections == 1
    await adapter.aclose()
    await provider.finish()


def test_framer_trims_a_dangling_byte_and_flushes_nothing_when_empty() -> None:
    framer = _Framer(4)
    assert [bytes(f) for f in framer.feed(b"\x01")] == []
    assert [bytes(f) for f in framer.feed(b"\x02")] == [], "a straddling frame waits for more"
    assert [bytes(f) for f in framer.feed(b"\x03")] == []
    assert [bytes(f) for f in framer.feed(b"\x04\x05")] == [b"\x01\x02\x03\x04"]
    remainder = framer.flush()
    assert remainder is None, "a single dangling byte is not a sample"
    assert framer.flush() is None


def test_framer_flush_keeps_whole_interleaved_stereo_sample_frames() -> None:
    framer = _Framer(16, sample_frame_bytes=4)  # 4 stereo PCM16 sample frames per frame
    assert [bytes(f) for f in framer.feed(bytes(range(22)))] == [bytes(range(16))]
    remainder = framer.flush()
    assert remainder is not None
    assert bytes(remainder) == bytes(range(16, 20)), "1.5 trailing sample frames -> 1"


async def test_stereo_tail_is_trimmed_to_whole_sample_frames(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def stereo_with_ragged_tail(
        ws: MemoryWebSocket, _c: int, _n: int, request: dict[str, object]
    ) -> bool:
        context = str(request["context_id"])
        await ws.send(chunk(context, bytes(1290)))  # 1280-byte frame + 2.5 sample frames
        await ws.send(done(context))
        return True

    provider = cartesia(stereo_with_ragged_tail)
    adapter = tts(provider, recorder, trace)
    stamper = FrameStamper(trace, channels=2)
    audio = [a async for a in adapter.synthesize("Hi.", stamper)]
    assert [a.frame.nbytes for a in audio] == [1280, 8]
    assert all(a.frame.channels == 2 for a in audio)
    assert audio[-1].frame.duration_ns == 2 * 1_000_000_000 // 16_000, "2 sample frames"
    await adapter.aclose()
    await provider.finish()


@given(
    data=st.binary(max_size=600),
    cuts=st.lists(st.integers(min_value=0, max_value=600), max_size=12),
    channels=st.sampled_from([1, 2]),
    samples_per_frame=st.integers(min_value=1, max_value=40),
)
def test_framing_is_lossless_and_sample_aligned_for_any_chunking(
    data: bytes, cuts: list[int], channels: int, samples_per_frame: int
) -> None:
    sample_frame = 2 * channels
    frame_bytes = samples_per_frame * sample_frame
    framer = _Framer(frame_bytes, sample_frame_bytes=sample_frame)
    bounds = [0, *sorted(c for c in cuts if c <= len(data)), len(data)]
    frames = [
        bytes(f) for start, end in itertools.pairwise(bounds) for f in framer.feed(data[start:end])
    ]
    remainder = framer.flush()
    tail = b"" if remainder is None else bytes(remainder)

    assert all(len(f) == frame_bytes for f in frames), "every streamed frame is exact"
    assert len(tail) < frame_bytes
    assert len(tail) % sample_frame == 0, "the tail never ends mid sample frame"
    usable = len(data) - len(data) % sample_frame
    assert b"".join(frames) + tail == data[:usable], "nothing lost, duplicated or reordered"
    stamper = FrameStamper(TraceContext.new(), channels=channels)
    for payload in [*frames, *([tail] if tail else [])]:
        assert stamper.stamp(memoryview(payload), duration_ns=1).nbytes == len(payload)


# ------------------------------------------------------------------------ resilience


def chaos(action: str) -> Script:
    async def script(
        ws: MemoryWebSocket, connection: int, number: int, request: dict[str, object]
    ) -> bool:
        context = str(request["context_id"])
        if connection == 0 and number == 0:
            if action == "drop":
                await ws.sever()
                return False
            if action == "garbage":
                await ws.send('{"type":"chunk","context_id":"%s","data":"@@@"}' % context)  # noqa: UP031
                return False
            await ws.send(failure(context, int(action)))
            return False
        return await synthesize_all(ws, connection, number, request)

    return script


@pytest.mark.parametrize(
    ("action", "kind"),
    [("drop", "transient"), ("garbage", "protocol"), ("500", "transient"), ("429", "rate_limit")],
)
async def test_failures_before_audio_retry_the_unit(
    recorder: FlightRecorder, trace: TraceContext, action: str, kind: str
) -> None:
    provider = cartesia(chaos(action))
    adapter = tts(provider, recorder, trace)
    audio = await speak(adapter, trace)
    assert b"".join(bytes(a.frame.payload) for a in audio) == pcm_for(TEXT)
    assert provider.connections == 2
    assert recorder.of_type(ProviderError)[0].kind == kind
    await adapter.aclose()
    await provider.finish()


@pytest.mark.parametrize(
    ("status", "kind"), [(401, ErrorKind.AUTH), (403, ErrorKind.AUTH), (402, ErrorKind.QUOTA)]
)
async def test_auth_and_quota_failures_are_fatal(
    recorder: FlightRecorder, trace: TraceContext, status: int, kind: ErrorKind
) -> None:
    provider = cartesia(chaos(str(status)))
    with pytest.raises(UpstreamError) as excinfo:
        await speak(tts(provider, recorder, trace), trace)
    assert excinfo.value.kind is kind
    await provider.finish()


async def test_drop_after_frames_ends_the_unit_gracefully(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = cartesia()
    adapter = tts(provider, recorder, trace)
    audio = []
    async for synthesized in adapter.synthesize(TEXT * 10, FrameStamper(trace)):
        audio.append(synthesized)
        if len(audio) == 2:
            await provider.servers[0].sever()
    assert 2 <= len(audio) < 85, "frames already spoken stand; the unit ends"
    assert recorder.of_type(ProviderError)[-1].kind == "transient"
    await adapter.aclose()
    await provider.finish()


async def test_closing_early_cancels_the_context(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = cartesia()
    adapter = tts(provider, recorder, trace)
    async with contextlib.aclosing(adapter.synthesize(TEXT * 10, FrameStamper(trace))) as audio:
        async for _ in audio:
            break
    await adapter.aclose()
    await provider.finish()
    log = provider.log  # type: ignore[attr-defined]
    assert log[-1] == {"context_id": log[0]["context_id"], "cancel": True}


async def test_cancel_on_a_closed_connection_is_harmless(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = cartesia()
    adapter = tts(provider, recorder, trace)
    async with contextlib.aclosing(adapter.synthesize(TEXT * 10, FrameStamper(trace))) as audio:
        async for _ in audio:
            await provider.servers[0].close()
            break
    await adapter.aclose()
    await provider.finish()


async def test_persistent_failures_exhaust_the_budget(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def always_drop(ws: MemoryWebSocket, _c: int, _n: int, _r: dict[str, object]) -> bool:
        await ws.sever()
        return False

    provider = cartesia(always_drop)
    with pytest.raises(UpstreamError, match="reconnect budget spent"):
        await speak(tts(provider, recorder, trace, policy=RetryPolicy(max_reconnects=1)), trace)
    await provider.finish()


async def test_handshake_refusal_is_fatal(recorder: FlightRecorder, trace: TraceContext) -> None:
    provider = cartesia(synthesize_all, UpstreamError(ErrorKind.QUOTA, "402"))
    with pytest.raises(UpstreamError):
        await speak(tts(provider, recorder, trace), trace)


# ------------------------------------------------------------------------ parser contract


def test_parser_contract() -> None:
    assert parse_cartesia_message(chunk("c", b"\x00\x01")) == AudioChunk(
        context_id="c", data=b"\x00\x01"
    )
    assert parse_cartesia_message(done("c")) == ContextDone(context_id="c")
    assert parse_cartesia_message(timestamps("c", "Hi you")) == WordTimestamps(
        context_id="c", words=("Hi", "you"), ends_s=(0.02, 0.06)
    )
    assert parse_cartesia_message('{"type":"error","error":5,"status_code":true}') == ContextFailed(
        context_id=None, status_code=None, message=""
    )
    assert parse_cartesia_message('{"type":"something_new"}') is None
    assert [classify_status(s) for s in (401, 402, 403, 429, 503, 400, None)] == [
        ErrorKind.AUTH, ErrorKind.QUOTA, ErrorKind.AUTH, ErrorKind.RATE_LIMIT,
        ErrorKind.TRANSIENT, ErrorKind.PROTOCOL, ErrorKind.PROTOCOL,
    ]  # fmt: skip


@pytest.mark.parametrize(
    "message",
    [
        "x",
        "[]",
        '{"type":"chunk","context_id":1,"data":""}',
        '{"type":"chunk","context_id":"c","data":7}',
        '{"type":"chunk","context_id":"c","data":"!!"}',
        '{"type":"done"}',
        '{"context_id":"c","word_timestamps":[]}',
        '{"context_id":"c","word_timestamps":{"words":["a"],"end":[]}}',
        '{"context_id":"c","word_timestamps":{"words":[1],"end":[0.1]}}',
        '{"context_id":"c","word_timestamps":{"words":["a"],"end":[-1]}}',
        '{"context_id":"c","word_timestamps":{"words":["a"],"end":[true]}}',
    ],
)
def test_malformed_messages_are_protocol_errors(message: str) -> None:
    with pytest.raises(UpstreamError) as excinfo:
        parse_cartesia_message(message)
    assert excinfo.value.kind is ErrorKind.PROTOCOL


@settings(max_examples=400)
@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
        lambda inner: (
            st.lists(inner, max_size=3)
            | st.dictionaries(
                st.sampled_from(
                    [
                        "type",
                        "context_id",
                        "data",
                        "word_timestamps",
                        "words",
                        "end",
                        "status_code",
                        "error",
                    ]
                ),
                inner,
                max_size=5,
            )
        ),
        max_leaves=15,
    ),
    st.sampled_from(["chunk", "done", "error", "timestamps", "other"]),
)
def test_arbitrary_messages_never_escape_as_anything_but_protocol_errors(
    document: object, kind: str
) -> None:
    if isinstance(document, dict):
        document = {**document, "type": kind}
    with contextlib.suppress(UpstreamError):
        parse_cartesia_message(json.dumps(document))


async def test_alignment_ignores_words_missing_from_the_text(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def odd(ws: MemoryWebSocket, _c: int, _n: int, request: dict[str, object]) -> bool:
        context = str(request["context_id"])
        await ws.send(
            json.dumps(
                {
                    "context_id": context,
                    "word_timestamps": {"words": ["", "zzz", "Hi"], "end": [0.0, 0.01, 0.02]},
                }
            )
        )
        await ws.send(chunk(context, pcm_for("Hi")))
        await ws.send(done(context))
        return True

    provider = cartesia(odd)
    adapter = tts(provider, recorder, trace)
    audio = await speak(adapter, trace, "Hi")
    assert [a.text_offset for a in audio] == [2]
    await adapter.aclose()
    await provider.finish()
