"""State machine fuzzing: random trigger streams against an independent reference model."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from voice_ai_realtime_gateway.domain.errors import InvalidTransitionError
from voice_ai_realtime_gateway.domain.machine import SessionStateMachine
from voice_ai_realtime_gateway.domain.states import DrainReason, SessionState, SessionTrigger
from voice_ai_realtime_gateway.telemetry.events import StateTransitioned, TransitionRejected
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

S = SessionState
T = SessionTrigger
HISTORY = 8


def reference_step(
    state: SessionState, reason: DrainReason | None, trigger: SessionTrigger
) -> tuple[SessionState, DrainReason | None] | None:
    """Hand-written reference semantics, deliberately not sharing code with the SUT."""
    if state is S.TERMINATED:
        return None
    if trigger is T.FAULT:
        return S.TERMINATED, None
    if trigger is T.CLOSE:
        if state in {S.IDLE, S.CONNECTING}:
            return S.TERMINATED, None
        return S.DRAINING, DrainReason.SHUTDOWN
    if state is S.DRAINING and trigger is T.DRAIN_COMPLETE:
        return (S.LISTENING if reason is DrainReason.BARGE_IN else S.TERMINATED), None
    return REFERENCE_SIMPLE.get((state, trigger))


REFERENCE_SIMPLE: dict[
    tuple[SessionState, SessionTrigger], tuple[SessionState, DrainReason | None]
] = {
    (S.IDLE, T.CONNECT): (S.CONNECTING, None),
    (S.CONNECTING, T.CONNECTED): (S.LISTENING, None),
    (S.LISTENING, T.USER_TURN_ENDED): (S.PROCESSING_THOUGHT, None),
    (S.PROCESSING_THOUGHT, T.RESPONSE_AUDIO_STARTED): (S.SPEAKING, None),
    (S.PROCESSING_THOUGHT, T.RESPONSE_SKIPPED): (S.LISTENING, None),
    (S.SPEAKING, T.RESPONSE_AUDIO_FINISHED): (S.LISTENING, None),
    (S.PROCESSING_THOUGHT, T.BARGE_IN): (S.BARGED_IN, None),
    (S.SPEAKING, T.BARGE_IN): (S.BARGED_IN, None),
    (S.BARGED_IN, T.DRAIN): (S.DRAINING, DrainReason.BARGE_IN),
}


class SessionFuzz(RuleBasedStateMachine):
    machine: SessionStateMachine
    recorder: FlightRecorder
    state: SessionState
    reason: DrainReason | None
    last_lamport: int
    committed: int
    rejected: int
    now: int

    @initialize()
    def setup(self) -> None:
        self.recorder = FlightRecorder(capacity=100_000)
        self.now = 0

        def clock() -> int:
            self.now += 1
            return self.now

        self.machine = SessionStateMachine(
            trace=TraceContext.new(), sink=self.recorder, clock=clock, history_size=HISTORY
        )
        self.state, self.reason = S.IDLE, None
        self.last_lamport = 0
        self.committed = self.rejected = 0

    @rule(
        trigger=st.sampled_from(SessionTrigger),
        observed=st.one_of(st.none(), st.integers(min_value=0, max_value=1_000)),
    )
    def fire(self, trigger: SessionTrigger, observed: int | None) -> None:
        expected = reference_step(self.state, self.reason, trigger)
        if expected is None:
            with pytest.raises(InvalidTransitionError) as excinfo:
                self.machine.fire(trigger, observed=observed)
            assert (excinfo.value.state, excinfo.value.trigger) == (self.state, trigger)
            self.rejected += 1
            return
        transition = self.machine.fire(trigger, observed=observed)
        assert transition.lamport > self.last_lamport
        if observed is not None:
            assert transition.lamport > observed
        self.state, self.reason = expected
        self.last_lamport = transition.lamport
        self.committed += 1

    @rule()
    def rotate_span(self) -> None:
        before = self.machine.trace
        after = self.machine.open_span()
        assert after.session_id == before.session_id

    @invariant()
    def agrees_with_reference(self) -> None:
        assert self.machine.state is self.state
        assert self.machine.drain_reason is self.reason
        assert self.machine.lamport == self.last_lamport
        assert self.machine.terminated is (self.state is S.TERMINATED)

    @invariant()
    def history_is_bounded_and_strictly_ordered(self) -> None:
        history = self.machine.history
        assert len(history) == min(self.committed, HISTORY)
        stamps = [t.lamport for t in history]
        assert stamps == sorted(set(stamps))
        times = [t.at_ns for t in history]
        assert times == sorted(times)
        for prev, nxt in itertools.pairwise(history):
            assert prev.target is nxt.source, "history is a connected path"

    @invariant()
    def every_input_is_observable(self) -> None:
        assert len(self.recorder.of_type(StateTransitioned)) == self.committed
        assert len(self.recorder.of_type(TransitionRejected)) == self.rejected


SessionFuzz.TestCase.settings = settings(max_examples=300, stateful_step_count=80)
TestSessionFuzz = SessionFuzz.TestCase
