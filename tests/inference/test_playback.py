"""Barge-in truncation: the history keeps only whole words the user actually heard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from audio_support import pcm_frame
from voice_ai_realtime_gateway.inference.playback import PlaybackLedger, SpeechMap, word_prefix
from voice_ai_realtime_gateway.telemetry.ids import TraceContext

if TYPE_CHECKING:
    from conftest import FakeClock
    from voice_ai_realtime_gateway.core.frames import AudioFrame

FRAME = 20_000_000


# ------------------------------------------------------------------------ word snapping


@pytest.mark.parametrize(
    ("text", "cut", "expected"),
    [
        ("The weather is sunny", 0, ""),
        ("The weather is sunny", 3, "The"),
        ("The weather is sunny", 4, "The"),
        ("The weather is sunny", 6, "The"),
        ("The weather is sunny", 11, "The weather"),
        ("The weather is sunny", 20, "The weather is sunny"),
        ("Hello, world", 5, ""),  # "Hello" without its comma is still a cut word
        ("Hello, world", 6, "Hello,"),
        ("  padded  ", 10, "padded"),
    ],
)
def test_word_prefix_never_keeps_a_partial_word(text: str, cut: int, expected: str) -> None:
    assert word_prefix(text, cut) == expected


def test_word_prefix_validates_the_cut() -> None:
    with pytest.raises(ValueError, match="outside"):
        word_prefix("abc", 4)


# ------------------------------------------------------------------------ speech map


def aligned(units: list[str], frames_per_unit: int = 4) -> SpeechMap:
    speech = SpeechMap()
    for unit in units:
        speech.begin_unit(unit)
        for index in range(frames_per_unit):
            speech.add_frame(FRAME, len(unit) * (index + 1) // frames_per_unit)
    return speech


def test_truncation_mid_second_sentence_keeps_only_heard_words() -> None:
    speech = aligned(["Sure, I can help.", "The weather in Oslo is four degrees."])
    assert speech.text_through(0) == ""
    assert speech.text_through(FRAME * 4) == "Sure, I can help."
    # 2 of 4 frames into unit 2: 18 of 36 chars voiced ("The weather in Osl")
    assert speech.text_through(FRAME * 6) == "Sure, I can help. The weather in"
    assert speech.text_through(FRAME * 6 + FRAME // 2) == "Sure, I can help. The weather in"
    assert speech.text_through(FRAME * 100) == (
        "Sure, I can help. The weather in Oslo is four degrees."
    )
    assert speech.synthesized_ns == FRAME * 8
    assert speech.frames == 8
    assert speech.units == ("Sure, I can help.", "The weather in Oslo is four degrees.")
    assert repr(speech) == "SpeechMap(units=2, frames=8)"


def test_speech_map_rejects_misuse() -> None:
    speech = SpeechMap()
    with pytest.raises(ValueError, match="before begin_unit"):
        speech.add_frame(FRAME, 0)
    speech.begin_unit("abc")
    with pytest.raises(ValueError, match="outside unit"):
        speech.add_frame(FRAME, 4)


words = st.text(alphabet="abcdefghij", min_size=1, max_size=7)
units_strategy = st.lists(
    st.lists(words, min_size=1, max_size=6).map(" ".join), min_size=1, max_size=5
)


@settings(max_examples=300)
@given(
    units=units_strategy,
    frames=st.lists(st.integers(min_value=1, max_value=6), min_size=5, max_size=5),
    cuts=st.lists(st.integers(min_value=0, max_value=60 * FRAME), min_size=1, max_size=8),
)
def test_heard_text_is_a_monotone_whole_word_prefix_of_the_response(
    units: list[str], frames: list[int], cuts: list[int]
) -> None:
    speech = SpeechMap()
    for unit, count in zip(units, frames, strict=False):
        speech.begin_unit(unit)
        for index in range(count):
            speech.add_frame(FRAME, len(unit) * (index + 1) // count)
    full_words = " ".join(units).split()
    previous: list[str] = []
    for cut in sorted(cuts):
        heard = speech.text_through(cut).split()
        assert heard == full_words[: len(heard)], "no word outside the spoken prefix"
        assert heard[: len(previous)] == previous, "hearing more never forgets words"
        previous = heard
    assert speech.text_through(speech.synthesized_ns).split() == full_words


# ------------------------------------------------------------------------ ledger


class Sink:
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []

    async def send(self, frame: AudioFrame, /) -> None:
        self.frames.append(frame)


async def test_ledger_forwards_and_models_real_time_playback(clock: FakeClock) -> None:
    inner = Sink()
    ledger = PlaybackLedger(inner, clock=clock)
    span = TraceContext.new()
    assert ledger.first_sent_ns(span.span_id) is None
    assert ledger.heard_ns(span.span_id, clock.now) == 0
    clock.now = 1_000_000_000
    for sequence in range(10):  # 200 ms of audio sent in a burst
        await ledger.send(pcm_frame(span, sequence=sequence))
    assert len(inner.frames) == 10
    assert ledger.first_sent_ns(span.span_id) == 1_000_000_000
    assert ledger.sent_ns(span.span_id) == 10 * FRAME
    assert ledger.heard_ns(span.span_id, 1_000_000_000 + 50_000_000) == 50_000_000
    assert ledger.heard_ns(span.span_id, 1_000_000_000 + 900_000_000) == 10 * FRAME
    assert ledger.heard_ns(span.span_id, 0) == 0, "clock skew never yields negative playback"


async def test_ledger_resets_on_a_new_response_span(clock: FakeClock) -> None:
    ledger = PlaybackLedger(Sink(), clock=clock)
    first, second = TraceContext.new(), TraceContext.new()
    await ledger.send(pcm_frame(first))
    clock.now += 5_000_000
    await ledger.send(pcm_frame(second))
    assert ledger.sent_ns(first.span_id) == 0
    assert ledger.sent_ns(second.span_id) == FRAME
    assert ledger.first_sent_ns(second.span_id) == clock.now
