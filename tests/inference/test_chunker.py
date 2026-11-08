"""Sentence chunker: linguistic edge cases, split invariance, conservation, bounds."""

from __future__ import annotations

import re
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from voice_ai_realtime_gateway.inference.chunker import ChunkerConfig, SentenceChunker

STOP, BANG, QUERY = "\u3002", "\uff01", "\uff1f"  # ideographic full stop, full-width ! ?


def chunk(
    text: str, config: ChunkerConfig | None = None, *, pieces: list[int] | None = None
) -> list[str]:
    chunker = SentenceChunker(config)
    units: list[str] = []
    if pieces is None:
        units += chunker.feed(text)
    else:
        position = 0
        for size in pieces:
            units += chunker.feed(text[position : position + size])
            position += size
        units += chunker.feed(text[position:])
    return units + chunker.flush()


# ------------------------------------------------------------------------ boundaries


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there. How are you?", ["Hello there.", "How are you?"]),
        ("Wait! Really?! Yes.", ["Wait!", "Really?!", "Yes."]),
        ("Well... I think so. Okay", ["Well...", "I think so.", "Okay"]),
        ("Pi is 3.14 today. Sure.", ["Pi is 3.14 today.", "Sure."]),
        ("See Dr. Smith now. Bye.", ["See Dr. Smith now.", "Bye."]),
        ("Use tools, e.g. a hammer. Done.", ["Use tools, e.g. a hammer.", "Done."]),
        ("Ask J. Smith first. Then me.", ["Ask J. Smith first.", "Then me."]),
        ("Neither do I. Fine.", ["Neither do I.", "Fine."]),
        ('He said "go." Then left.', ['He said "go."', "Then left."]),
        ("(This is it.) Next one.", ["(This is it.)", "Next one."]),
        ("Visit example.com today. Ok.", ["Visit example.com today.", "Ok."]),
        (
            f"你好{STOP}今天天气很好{BANG}好吗{QUERY}",
            [f"你好{STOP}", f"今天天气很好{BANG}", f"好吗{QUERY}"],
        ),
        ("Trailing text without punctuation", ["Trailing text without punctuation"]),
        ("Hi . . . there.", ["Hi .", ". . there."]),
    ],
)
def test_sentence_boundaries(text: str, expected: list[str]) -> None:
    config = ChunkerConfig(first_clause_min_chars=200, clause_min_chars=200, max_chars=200)
    assert chunk(text, config) == expected


def test_first_unit_may_cut_early_at_a_clause_for_latency() -> None:
    config = ChunkerConfig(first_clause_min_chars=16, clause_min_chars=40)
    text = "Okay, so the weather today, as far as I can tell, is mild and dry. Next, rain."
    assert chunk(text, config) == [
        "Okay, so the weather today,",
        "as far as I can tell, is mild and dry.",
        "Next, rain.",
    ]


def test_short_clauses_do_not_cut() -> None:
    config = ChunkerConfig(first_clause_min_chars=16, clause_min_chars=40)
    assert chunk("Yes, no, maybe.", config) == ["Yes, no, maybe."]
    assert chunk("A list: one; two — three.", ChunkerConfig(first_clause_min_chars=30)) == [
        "A list: one; two — three."
    ]
    assert chunk("A list: one; two — three.", config) == ["A list: one; two —", "three."]


def test_clause_marks_without_whitespace_are_not_boundaries() -> None:
    config = ChunkerConfig(first_clause_min_chars=1)
    assert chunk("It costs 1,000,000 dollars", config) == ["It costs 1,000,000 dollars"]


def test_max_chars_splits_at_the_last_whitespace() -> None:
    config = ChunkerConfig(max_chars=20, first_clause_min_chars=20, clause_min_chars=20)
    units = chunk("one two three four five six seven eight", config)
    assert units == ["one two three four", "five six seven eight"]
    assert all(len(unit) <= 20 for unit in units)


def test_max_chars_hard_cuts_a_run_without_whitespace() -> None:
    config = ChunkerConfig(max_chars=8, first_clause_min_chars=8, clause_min_chars=8)
    assert chunk("abcdefghijklmnopqrst", config) == ["abcdefgh", "ijklmnop", "qrst"]


def test_overlong_sentence_is_capped_even_when_its_end_is_known() -> None:
    config = ChunkerConfig(max_chars=12, first_clause_min_chars=12, clause_min_chars=12)
    assert chunk("aaaa bbbb cccc dddd. Ok.", config) == ["aaaa bbbb", "cccc dddd.", "Ok."]


def test_ambiguous_terminator_waits_for_the_next_token() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("The value is 3.") == []
    assert chunker.feed("5 exactly. ") == ["The value is 3.5 exactly."]
    assert chunker.feed("Done") == []
    assert chunker.pending == "Done"
    assert chunker.flush() == ["Done"]


def test_punctuation_only_remainder_is_dropped_and_chunker_is_reusable() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("All set. ") == ["All set."]
    assert chunker.feed(" ...") == []
    assert chunker.flush() == []
    assert chunker.feed("Again. ") == ["Again."]
    assert repr(chunker) == "SentenceChunker(pending=0, emitted=1)"


def test_punctuation_alone_never_forms_a_unit_mid_stream() -> None:
    assert chunk("... !! ?? Real words. Yes.") == ["... !! ?? Real words.", "Yes."]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_chars": 0}, "max_chars"),
        ({"max_chars": 10, "first_clause_min_chars": 11}, "first_clause_min_chars"),
        ({"clause_min_chars": 0}, "clause_min_chars"),
    ],
)
def test_config_is_validated(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ChunkerConfig(**kwargs)


def test_config_is_exposed() -> None:
    config = ChunkerConfig(max_chars=100)
    assert SentenceChunker(config).config is config


# ------------------------------------------------------------------------ fuzzing

TOKENS = st.sampled_from(
    [
        "Hello",
        "world",
        "3.14",
        "Dr.",
        "e.g.",
        "J.",
        "I.",
        "ok",
        "...",
        "?!",
        ".",
        ",",
        ";",
        ":",
        "—",
        '"',
        ")",
        "(",
        "1,000",
        "你好",
        STOP,
        BANG,
        " ",
        " ",
        " ",
        "\n",
        "x" * 30,
    ]
)
texts = st.lists(TOKENS, max_size=80).map("".join)
configs = st.builds(
    lambda cap, first, later: ChunkerConfig(
        max_chars=cap, first_clause_min_chars=min(first, cap), clause_min_chars=min(later, cap)
    ),
    st.integers(min_value=1, max_value=120),
    st.integers(min_value=1, max_value=120),
    st.integers(min_value=1, max_value=120),
)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


@settings(max_examples=400)
@given(text=texts, config=configs, pieces=st.lists(st.integers(min_value=0, max_value=9)))
def test_units_are_independent_of_token_splitting(
    text: str, config: ChunkerConfig, pieces: list[int]
) -> None:
    assert chunk(text, config, pieces=pieces) == chunk(text, config)
    assert chunk(text, config, pieces=[1] * len(text)) == chunk(text, config)


@settings(max_examples=400)
@given(text=texts, config=configs)
def test_no_speech_is_lost_duplicated_or_reordered(text: str, config: ChunkerConfig) -> None:
    units = chunk(text, config)
    spoken = squash("".join(units))
    original = squash(text)
    assert original.startswith(spoken)
    assert not any(char.isalnum() for char in original[len(spoken) :]), "only punctuation dropped"
    for unit in units:
        assert unit
        assert unit == unit.strip()
        assert len(unit) <= config.max_chars


def test_streaming_cost_is_linear_in_characters() -> None:
    sentence = "The quick brown fox jumps over the lazy dog, again and again. "
    tokens = re.findall(r"\S+\s*", sentence * 400)
    chunker = SentenceChunker()
    started = time.perf_counter()
    units = [unit for token in tokens for unit in chunker.feed(token)] + chunker.flush()
    elapsed = time.perf_counter() - started
    assert len(units) >= 400
    assert elapsed < 0.5, f"{len(tokens)} tokens took {elapsed:.3f}s"


# ------------------------------------------------------------------ mutation-derived contracts


def test_documented_defaults() -> None:
    assert ChunkerConfig() == ChunkerConfig(
        max_chars=200, first_clause_min_chars=16, clause_min_chars=60
    )


def test_clause_exactly_at_the_minimum_cuts_and_reuse_restores_the_first_unit_rule() -> None:
    chunker = SentenceChunker(ChunkerConfig(first_clause_min_chars=6, clause_min_chars=40))
    assert chunker.feed("Hello, world") == ["Hello,"], "6 chars == first_clause_min_chars"
    assert chunker.flush() == ["world"]
    assert chunker.feed("Again, more") == ["Again,"], "a new response gets the early cut again"


def test_sentence_of_exactly_max_chars_is_kept_whole() -> None:
    config = ChunkerConfig(max_chars=10, first_clause_min_chars=10, clause_min_chars=10)
    assert chunk("abcd efgh. Ok.", config) == ["abcd efgh.", "Ok."]


def test_overflow_is_measured_from_the_current_unit_start() -> None:
    config = ChunkerConfig(max_chars=6, first_clause_min_chars=6, clause_min_chars=6)
    assert chunk("aaaa. b c.", config) == ["aaaa.", "b c."]
    # The overflow window of the 2nd unit ("b. c d") has whitespace after the sentence end,
    # so measuring from the wrong origin would cut "b. c" instead of "b.".
    assert chunk("aaaa. b. c d e", config) == ["aaaa.", "b.", "c d e"]


def test_overflow_cut_on_window_edges() -> None:
    config = ChunkerConfig(max_chars=10, first_clause_min_chars=10, clause_min_chars=10)
    assert chunk("abcdefghi jklmn", config) == ["abcdefghi", "jklmn"], "window ends on a space"
    tight = ChunkerConfig(max_chars=5, first_clause_min_chars=5, clause_min_chars=5)
    assert chunk("a bbbbbbbbb", tight) == ["a", "bbbbb", "bbbb"], "a space at offset 1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dr. Smith is here. Bye.", ["Dr. Smith is here.", "Bye."]),
        ("It is OK. Next one.", ["It is OK.", "Next one."]),
        ("Plan x. Then go.", ["Plan x.", "Then go."]),
    ],
)
def test_abbreviation_edges(text: str, expected: list[str]) -> None:
    config = ChunkerConfig(first_clause_min_chars=200, clause_min_chars=200, max_chars=200)
    assert chunk(text, config) == expected
