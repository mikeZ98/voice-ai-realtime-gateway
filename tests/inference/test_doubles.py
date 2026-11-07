"""Model ports and deterministic doubles: transcripts, tokens, aligned audio, closing."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from audio_support import pcm_frame
from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.inference.conversation import UserTurn
from voice_ai_realtime_gateway.inference.doubles import (
    UNKNOWN_WORD,
    ScriptedLLM,
    ToneASR,
    ToneTTS,
    pcm16_peak,
)
from voice_ai_realtime_gateway.inference.ports import FrameStamper, TranscriptChunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from conftest import FakeClock
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.inference.conversation import Turn
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

pytestmark = pytest.mark.usefixtures("strict_loop")

LEXICON = {1_000: "hello", 2_000: "world", 3_000: "again"}
FRAME = 20_000_000


async def stream(frames: list[AudioFrame]) -> AsyncIterator[AudioFrame]:
    for frame in frames:
        yield frame


def tones(trace: TraceContext, amplitudes: list[int]) -> list[AudioFrame]:
    return [
        pcm_frame(trace, amplitude=amplitude, sequence=index, origin_ns=index)
        for index, amplitude in enumerate(amplitudes)
    ]


# ------------------------------------------------------------------------ ASR


def test_peak_of_pcm16() -> None:
    assert pcm16_peak(memoryview(b"")) == 0
    assert pcm16_peak(memoryview(b"\x00\x80\xff\x7f")) == 32_768


async def test_tone_asr_yields_interim_words_then_a_final_transcript(trace: TraceContext) -> None:
    asr = ToneASR(LEXICON)
    frames = tones(trace, [0, 1_000, 1_000, 0, 2_000, 3_000, 3_000])
    chunks = [chunk async for chunk in asr.transcribe(stream(frames))]
    assert [(c.text, c.is_final) for c in chunks] == [
        ("hello", False),
        ("hello world", False),
        ("hello world again", True),
    ]
    final = chunks[-1]
    assert [(w.word, w.start_ns, w.end_ns) for w in final.words] == [
        ("hello", 1 * FRAME, 3 * FRAME),
        ("world", 4 * FRAME, 5 * FRAME),
        ("again", 5 * FRAME, 7 * FRAME),
    ]
    assert final.confidence > chunks[0].confidence
    assert final.trace == trace
    assert final.at_ns == 6
    assert (asr.active, asr.sessions) == (0, 1)


async def test_tone_asr_handles_unknown_tones_silence_and_empty_audio(
    trace: TraceContext,
) -> None:
    asr = ToneASR(LEXICON)
    chunks = [c async for c in asr.transcribe(stream(tones(trace, [4_321, 0])))]
    assert chunks[-1].text == UNKNOWN_WORD
    silent = [c async for c in asr.transcribe(stream(tones(trace, [0, 0])))]
    assert [(c.text, c.is_final, c.words) for c in silent] == [("", True, ())]
    assert [c async for c in asr.transcribe(stream([]))] == []


async def test_closing_the_asr_stream_early_releases_it(trace: TraceContext) -> None:
    asr = ToneASR(LEXICON)
    frames = tones(trace, [1_000, 0, 2_000, 0, 3_000])
    first: TranscriptChunk | None = None
    async with contextlib.aclosing(asr.transcribe(stream(frames))) as chunks:
        async for chunk in chunks:
            first = chunk
            break
    assert first is not None
    assert first.text == "hello"
    assert asr.active == 0


# ------------------------------------------------------------------------ LLM


async def test_scripted_llm_streams_whitespace_led_tokens(trace: TraceContext) -> None:
    llm = ScriptedLLM("Sure, the weather is mild.")
    history: list[Turn] = [UserTurn(text="earlier", trace=trace, at_ns=0)]
    deltas = [d async for d in llm.stream(history, "weather?")]
    assert [d.text for d in deltas] == ["Sure,", " the", " weather", " is", " mild."]
    assert [d.index for d in deltas] == [0, 1, 2, 3, 4]
    assert llm.calls[-1] == (tuple(history), "weather?")
    assert llm.active == 0


async def test_scripted_llm_can_answer_from_context() -> None:
    def echo(history: Sequence[Turn], user_text: str) -> str:
        return f"{len(history)} turns before: {user_text}"

    llm = ScriptedLLM(echo)
    text = "".join([d.text async for d in llm.stream([], "hi")])
    assert text == "0 turns before: hi"


async def test_closing_the_llm_stream_early_releases_it() -> None:
    llm = ScriptedLLM("one two three four")
    async with contextlib.aclosing(llm.stream([], "go")) as deltas:
        async for _ in deltas:
            break
    assert llm.active == 0


# ------------------------------------------------------------------------ TTS


async def test_stamper_wraps_without_copying_and_numbers_continuously(
    trace: TraceContext, clock: FakeClock
) -> None:
    stamper = FrameStamper(trace, sample_rate_hz=24_000, channels=2, clock=clock)
    data = bytes(1920)
    first = stamper.stamp(data, duration_ns=FRAME)
    second = stamper.stamp(data, duration_ns=FRAME)
    assert first.payload.obj is data
    assert (first.sequence, second.sequence, stamper.stamped) == (0, 1, 2)
    assert (first.trace, first.origin_ns, first.channels) == (trace, clock.now, 2)
    assert first.codec is AudioCodec.PCM16
    assert repr(stamper) == f"FrameStamper(span={trace.span_id}, stamped=2)"
    with pytest.raises(ValueError, match="whole interleaved"):
        stamper.stamp(bytes(3), duration_ns=FRAME)


async def test_tone_tts_alignment_reaches_the_full_unit(trace: TraceContext) -> None:
    tts = ToneTTS(ms_per_char=10)
    stamper = FrameStamper(trace)
    audio = [a async for a in tts.synthesize("Hello world.", stamper)]  # 12 chars -> 6 frames
    assert len(audio) == 6
    offsets = [a.text_offset for a in audio]
    assert offsets == sorted(offsets)
    assert offsets[-1] == len("Hello world.")
    assert [a.frame.sequence for a in audio] == list(range(6))
    assert all(a.frame.payload.obj is audio[0].frame.payload.obj for a in audio), "shared buffer"
    assert all(a.frame.trace == trace for a in audio)
    assert (tts.active, tts.completed, tts.aborted) == (0, 1, 0)
    assert tts.last_closed_ns is not None


async def test_tone_tts_gives_short_text_at_least_one_frame(trace: TraceContext) -> None:
    tts = ToneTTS(ms_per_char=1)
    audio = [a async for a in tts.synthesize("Ok", FrameStamper(trace))]
    assert [a.text_offset for a in audio] == [2]


async def test_closing_tts_mid_unit_counts_as_aborted(trace: TraceContext) -> None:
    tts = ToneTTS(pace_s=0.0)
    async with contextlib.aclosing(
        tts.synthesize("A long sentence here.", FrameStamper(trace))
    ) as audio:
        async for _ in audio:
            break
    assert (tts.active, tts.completed, tts.aborted) == (0, 0, 1)


async def test_one_stamper_spans_units_and_tts_reuses_its_frame_buffer(
    trace: TraceContext,
) -> None:
    tts = ToneTTS(ms_per_char=20)
    stamper = FrameStamper(trace)
    first = [a async for a in tts.synthesize("One.", stamper)]
    second = [a async for a in tts.synthesize("Two.", stamper)]
    assert stamper.trace == trace
    assert [a.frame.sequence for a in first + second] == list(range(8)), "continuous numbering"
    assert first[0].frame.payload.obj is second[0].frame.payload.obj, "payload cached per format"
