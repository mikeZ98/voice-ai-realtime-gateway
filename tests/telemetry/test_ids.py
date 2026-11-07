from __future__ import annotations

import secrets

import pytest
from hypothesis import given
from hypothesis import strategies as st

from voice_ai_realtime_gateway.telemetry.ids import (
    AUDIO_SPAN_ID_HEADER,
    SESSION_ID_HEADER,
    InvalidTraceIdError,
    TraceContext,
    new_session_id,
    new_span_id,
    parse_session_id,
    parse_span_id,
)

HEX = "0123456789abcdef"


def test_header_names_are_the_wire_contract() -> None:
    assert SESSION_ID_HEADER == "X-Session-ID"
    assert AUDIO_SPAN_ID_HEADER == "X-Audio-Span-ID"


def test_generated_ids_have_fixed_width_lowercase_hex() -> None:
    session, span = new_session_id(), new_span_id()
    assert len(session) == 32
    assert len(span) == 16
    assert set(session) <= set(HEX)
    assert set(span) <= set(HEX)
    assert parse_session_id(session) == session
    assert parse_span_id(span) == span


def test_generation_retries_the_reserved_all_zero_value(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["0" * 16, "00000000000000ab"])
    monkeypatch.setattr(secrets, "token_hex", lambda _n: next(outputs))
    assert new_span_id() == "00000000000000ab"


@pytest.mark.parametrize(
    "raw",
    ["", "0" * 32, "A" * 32, "g" * 32, "a" * 31, "a" * 33, " " + "a" * 31, "a" * 32 + "\n"],
)
def test_malformed_session_ids_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidTraceIdError):
        parse_session_id(raw)


@pytest.mark.parametrize("raw", ["", "0" * 16, "F" * 16, "a" * 15, "a" * 17])
def test_malformed_span_ids_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidTraceIdError):
        parse_span_id(raw)


def test_child_span_keeps_session_and_rotates_span() -> None:
    parent = TraceContext.new()
    child = parent.child()
    assert child.session_id == parent.session_id
    assert child.span_id != parent.span_id


def test_trace_context_is_immutable_and_hashable() -> None:
    trace = TraceContext.new()
    with pytest.raises(AttributeError):
        trace.span_id = new_span_id()  # type: ignore[misc]
    assert {trace: 1}[TraceContext(trace.session_id, trace.span_id)] == 1


@given(
    session=st.text(alphabet=HEX, min_size=32, max_size=32).filter(lambda s: set(s) != {"0"}),
    span=st.text(alphabet=HEX, min_size=16, max_size=16).filter(lambda s: set(s) != {"0"}),
    upper=st.booleans(),
)
def test_header_round_trip_is_lossless_and_case_insensitive(
    session: str, span: str, *, upper: bool
) -> None:
    trace = TraceContext(parse_session_id(session), parse_span_id(span))
    headers = trace.to_headers()
    if upper:
        headers = {name.upper(): value for name, value in headers.items()}
    assert TraceContext.from_headers(headers) == trace


def test_missing_header_is_rejected() -> None:
    with pytest.raises(InvalidTraceIdError, match="missing"):
        TraceContext.from_headers({SESSION_ID_HEADER: new_session_id()})


def test_duplicate_header_under_case_folding_is_rejected() -> None:
    trace = TraceContext.new()
    headers = {**trace.to_headers(), SESSION_ID_HEADER.lower(): new_session_id()}
    with pytest.raises(InvalidTraceIdError, match="duplicate"):
        TraceContext.from_headers(headers)


def test_malformed_header_value_is_rejected() -> None:
    with pytest.raises(InvalidTraceIdError):
        TraceContext.from_headers({SESSION_ID_HEADER: "nope", AUDIO_SPAN_ID_HEADER: "nope"})
