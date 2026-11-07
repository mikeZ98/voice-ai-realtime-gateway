"""Contract test: the transition matrix, written out independently and checked exhaustively."""

from __future__ import annotations

import itertools

import pytest

from voice_ai_realtime_gateway.domain.states import (
    ACTIVE_OUTPUT_STATES,
    TERMINAL_STATES,
    DrainReason,
    SessionState,
    SessionTrigger,
    drain_reason_for,
    resolve_transition,
    transition_table,
)

S = SessionState
T = SessionTrigger
R = DrainReason

# Independent oracle: (source, trigger, drain_reason) -> target. None = any/unguarded.
EXPECTED: dict[tuple[SessionState, SessionTrigger, DrainReason | None], SessionState] = {
    (S.IDLE, T.CONNECT, None): S.CONNECTING,
    (S.IDLE, T.CLOSE, None): S.TERMINATED,
    (S.IDLE, T.FAULT, None): S.TERMINATED,
    (S.CONNECTING, T.CONNECTED, None): S.LISTENING,
    (S.CONNECTING, T.CLOSE, None): S.TERMINATED,
    (S.CONNECTING, T.FAULT, None): S.TERMINATED,
    (S.LISTENING, T.USER_TURN_ENDED, None): S.PROCESSING_THOUGHT,
    (S.LISTENING, T.CLOSE, None): S.DRAINING,
    (S.LISTENING, T.FAULT, None): S.TERMINATED,
    (S.PROCESSING_THOUGHT, T.RESPONSE_AUDIO_STARTED, None): S.SPEAKING,
    (S.PROCESSING_THOUGHT, T.RESPONSE_SKIPPED, None): S.LISTENING,
    (S.PROCESSING_THOUGHT, T.BARGE_IN, None): S.BARGED_IN,
    (S.PROCESSING_THOUGHT, T.CLOSE, None): S.DRAINING,
    (S.PROCESSING_THOUGHT, T.FAULT, None): S.TERMINATED,
    (S.SPEAKING, T.RESPONSE_AUDIO_FINISHED, None): S.LISTENING,
    (S.SPEAKING, T.BARGE_IN, None): S.BARGED_IN,
    (S.SPEAKING, T.CLOSE, None): S.DRAINING,
    (S.SPEAKING, T.FAULT, None): S.TERMINATED,
    (S.BARGED_IN, T.DRAIN, None): S.DRAINING,
    (S.BARGED_IN, T.CLOSE, None): S.DRAINING,
    (S.BARGED_IN, T.FAULT, None): S.TERMINATED,
    (S.DRAINING, T.DRAIN_COMPLETE, R.BARGE_IN): S.LISTENING,
    (S.DRAINING, T.DRAIN_COMPLETE, R.SHUTDOWN): S.TERMINATED,
    (S.DRAINING, T.CLOSE, None): S.DRAINING,
    (S.DRAINING, T.FAULT, None): S.TERMINATED,
}


def oracle(
    state: SessionState, trigger: SessionTrigger, reason: DrainReason | None
) -> SessionState | None:
    if (state, trigger, reason) in EXPECTED:
        return EXPECTED[state, trigger, reason]
    return EXPECTED.get((state, trigger, None)) if trigger is not T.DRAIN_COMPLETE else None


def test_published_table_equals_the_oracle() -> None:
    assert transition_table() == EXPECTED


@pytest.mark.parametrize(
    ("state", "trigger", "reason"),
    list(itertools.product(SessionState, SessionTrigger, [None, *DrainReason])),
)
def test_every_state_trigger_reason_triple(
    state: SessionState, trigger: SessionTrigger, reason: DrainReason | None
) -> None:
    assert resolve_transition(state, trigger, reason) == oracle(state, trigger, reason)


def test_terminated_is_absorbing() -> None:
    for trigger, reason in itertools.product(SessionTrigger, [None, *DrainReason]):
        assert resolve_transition(S.TERMINATED, trigger, reason) is None
    assert frozenset({S.TERMINATED}) == TERMINAL_STATES


def test_every_non_terminal_state_can_fault_and_close() -> None:
    for state in SessionState:
        if state in TERMINAL_STATES:
            continue
        assert resolve_transition(state, T.FAULT, None) is S.TERMINATED
        assert resolve_transition(state, T.CLOSE, None) in {S.DRAINING, S.TERMINATED}


def test_every_state_is_reachable_from_idle() -> None:
    reachable = {S.IDLE}
    frontier = [S.IDLE]
    while frontier:
        state = frontier.pop()
        for (source, _, _), target in EXPECTED.items():
            if source is state and target not in reachable:
                reachable.add(target)
                frontier.append(target)
    assert reachable == set(SessionState)


def test_barge_in_only_interrupts_agent_output() -> None:
    accepting = {s for s in SessionState if resolve_transition(s, T.BARGE_IN, None) is not None}
    assert accepting == ACTIVE_OUTPUT_STATES == {S.PROCESSING_THOUGHT, S.SPEAKING}


@pytest.mark.parametrize(
    ("trigger", "previous", "expected"),
    [
        (T.DRAIN, None, R.BARGE_IN),
        (T.CLOSE, None, R.SHUTDOWN),
        (T.CLOSE, R.BARGE_IN, R.SHUTDOWN),
        (T.FAULT, R.BARGE_IN, R.BARGE_IN),
        (T.CONNECT, None, None),
    ],
)
def test_drain_reason_escalation(
    trigger: SessionTrigger, previous: DrainReason | None, expected: DrainReason | None
) -> None:
    assert drain_reason_for(trigger, previous) == expected
