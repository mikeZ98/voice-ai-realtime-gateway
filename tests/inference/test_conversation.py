from __future__ import annotations

import pytest

from voice_ai_realtime_gateway.inference.conversation import (
    AgentTurn,
    ConversationState,
    ToolCallTurn,
    UserTurn,
)
from voice_ai_realtime_gateway.telemetry.ids import TraceContext

TRACE = TraceContext.new()


def user(text: str) -> UserTurn:
    return UserTurn(text=text, trace=TRACE, at_ns=0)


def agent(text: str, *, interrupted: bool = False) -> AgentTurn:
    return AgentTurn(
        text=text, trace=TRACE, at_ns=0, interrupted=interrupted, generated_chars=len(text)
    )


def test_bounds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        ConversationState(max_turns=0)
    with pytest.raises(ValueError, match="positive"):
        ConversationState(max_chars=0)


def test_turns_are_kept_in_order_with_their_cost() -> None:
    state = ConversationState()
    tool = ToolCallTurn(
        name="weather", arguments='{"city":"Oslo"}', result="4C", trace=TRACE, at_ns=0
    )
    for turn in (user("hi"), agent("hello"), tool):
        state.append(turn)
    assert [type(t).__name__ for t in state.turns] == ["UserTurn", "AgentTurn", "ToolCallTurn"]
    assert len(state) == 3
    assert state.total_chars == 2 + 5 + len("weather") + len('{"city":"Oslo"}') + 2
    assert repr(state) == "ConversationState(turns=3, chars=31, pruned=0)"


def test_turn_window_prunes_oldest_first() -> None:
    state = ConversationState(max_turns=3)
    for index in range(5):
        state.append(user(f"u{index}"))
    assert [t.text for t in state.turns if isinstance(t, UserTurn)] == ["u2", "u3", "u4"]
    assert state.pruned == 2


def test_char_window_prunes_but_never_drops_the_newest_turn() -> None:
    state = ConversationState(max_chars=10)
    state.append(user("12345"))
    state.append(user("67890"))
    assert len(state) == 2
    state.append(agent("x" * 50))
    assert len(state) == 1, "an oversized newest turn survives alone"
    assert state.total_chars == 50
    assert state.pruned == 2


def test_history_before_returns_the_context_a_turn_was_said_in() -> None:
    state = ConversationState()
    first, reply, second = user("one"), agent("two"), user("one")  # equal text, distinct turns
    for turn in (first, reply, second):
        state.append(turn)
    assert state.history_before(second) == (first, reply)
    assert state.history_before(first) == ()
    assert state.history_before(user("unknown")) == state.turns, "pruned or foreign: full window"
