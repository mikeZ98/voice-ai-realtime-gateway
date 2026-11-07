from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from voice_ai_realtime_gateway.domain.clock import LamportClock
from voice_ai_realtime_gateway.domain.errors import InvalidTransitionError
from voice_ai_realtime_gateway.domain.machine import SessionStateMachine
from voice_ai_realtime_gateway.domain.states import DrainReason, SessionState, SessionTrigger
from voice_ai_realtime_gateway.telemetry.events import StateTransitioned, TransitionRejected

if TYPE_CHECKING:
    from conftest import FakeClock
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

S = SessionState
T = SessionTrigger


@pytest.fixture
def machine(trace: TraceContext, recorder: FlightRecorder, clock: FakeClock) -> SessionStateMachine:
    return SessionStateMachine(trace=trace, sink=recorder, clock=clock)


def drive(machine: SessionStateMachine, *triggers: SessionTrigger) -> None:
    for trigger in triggers:
        machine.fire(trigger)


def test_initial_state(machine: SessionStateMachine, trace: TraceContext) -> None:
    assert machine.state is S.IDLE
    assert machine.lamport == 0
    assert machine.drain_reason is None
    assert not machine.terminated
    assert machine.trace == trace
    assert machine.history == ()
    assert (
        repr(machine) == f"SessionStateMachine(state=idle, lamport=0, session={trace.session_id})"
    )


def test_history_size_must_be_positive(trace: TraceContext, recorder: FlightRecorder) -> None:
    with pytest.raises(ValueError, match="history_size"):
        SessionStateMachine(trace=trace, sink=recorder, history_size=0)


def test_committed_transition_is_stamped_correlated_and_emitted(
    machine: SessionStateMachine, recorder: FlightRecorder, clock: FakeClock, trace: TraceContext
) -> None:
    clock.now = 42
    transition = machine.fire(T.CONNECT)
    assert (transition.source, transition.target, transition.trigger) == (
        S.IDLE,
        S.CONNECTING,
        T.CONNECT,
    )
    assert (transition.lamport, transition.at_ns, transition.trace) == (1, 42, trace)
    assert recorder.events == (
        StateTransitioned(
            trace=trace, at_ns=42, lamport=1, source="idle", target="connecting", trigger="connect"
        ),
    )
    assert machine.history == (transition,)


def test_rejection_is_side_effect_free_but_observable(
    machine: SessionStateMachine, recorder: FlightRecorder, trace: TraceContext
) -> None:
    machine.fire(T.CONNECT)
    before = (machine.state, machine.lamport, machine.history, machine.drain_reason)
    with pytest.raises(InvalidTransitionError) as excinfo:
        machine.fire(T.BARGE_IN, observed=1_000)
    assert (excinfo.value.state, excinfo.value.trigger) == (S.CONNECTING, T.BARGE_IN)
    assert "barge_in" in str(excinfo.value)
    assert (machine.state, machine.lamport, machine.history, machine.drain_reason) == before
    (rejected,) = recorder.of_type(TransitionRejected)
    assert (rejected.state, rejected.trigger, rejected.lamport, rejected.trace) == (
        "connecting",
        "barge_in",
        1,
        trace,
    )


def test_negative_observation_is_rejected_before_anything_changes(
    machine: SessionStateMachine, recorder: FlightRecorder
) -> None:
    with pytest.raises(ValueError, match="observed"):
        machine.fire(T.CONNECT, observed=-5)
    assert machine.state is S.IDLE
    assert recorder.events == ()


def test_remote_observation_is_merged_into_lamport_order(machine: SessionStateMachine) -> None:
    assert machine.fire(T.CONNECT, observed=100).lamport == 101
    assert machine.fire(T.CONNECTED).lamport == 102
    assert machine.fire(T.USER_TURN_ENDED, observed=5).lamport == 103


def test_injected_lamport_clock_is_shared(trace: TraceContext, recorder: FlightRecorder) -> None:
    shared = LamportClock(10)
    machine = SessionStateMachine(trace=trace, sink=recorder, lamport=shared)
    machine.fire(T.CONNECT)
    assert shared.value == 11


def test_can_fire_mirrors_fire(machine: SessionStateMachine) -> None:
    assert machine.can_fire(T.CONNECT)
    assert not machine.can_fire(T.CONNECTED)


def test_barge_in_drain_cycle_returns_to_listening(machine: SessionStateMachine) -> None:
    drive(machine, T.CONNECT, T.CONNECTED, T.USER_TURN_ENDED, T.RESPONSE_AUDIO_STARTED)
    drive(machine, T.BARGE_IN, T.DRAIN)
    assert machine.state is S.DRAINING
    assert machine.drain_reason is DrainReason.BARGE_IN
    assert machine.fire(T.DRAIN_COMPLETE).target is S.LISTENING
    assert machine.drain_reason is None


def test_close_during_barge_in_drain_escalates_to_shutdown(machine: SessionStateMachine) -> None:
    drive(machine, T.CONNECT, T.CONNECTED, T.USER_TURN_ENDED, T.BARGE_IN, T.DRAIN)
    escalation = machine.fire(T.CLOSE)
    assert (escalation.source, escalation.target) == (S.DRAINING, S.DRAINING)
    assert escalation.drain_reason is DrainReason.SHUTDOWN
    assert machine.fire(T.DRAIN_COMPLETE).target is S.TERMINATED
    assert machine.terminated


def test_open_span_rotates_trace_for_subsequent_transitions(
    machine: SessionStateMachine, trace: TraceContext
) -> None:
    span = machine.open_span()
    assert span.session_id == trace.session_id
    assert span.span_id != trace.span_id
    assert machine.fire(T.CONNECT).trace == span


def test_history_is_bounded(trace: TraceContext, recorder: FlightRecorder) -> None:
    machine = SessionStateMachine(trace=trace, sink=recorder, history_size=3)
    drive(machine, T.CONNECT, T.CONNECTED)
    for _ in range(10):
        drive(machine, T.USER_TURN_ENDED, T.RESPONSE_SKIPPED)
    assert len(machine.history) == 3
    assert machine.history[-1].lamport == machine.lamport == 22
