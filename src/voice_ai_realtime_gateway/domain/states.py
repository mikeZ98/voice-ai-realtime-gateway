"""Session states, triggers and the formal transition matrix.

The matrix is a total function ``resolve_transition(state, trigger, drain_reason)``
returning the target state, or ``None`` for a rejected pair. It is pure data plus one
guard: ``DRAINING --DRAIN_COMPLETE-->`` resolves to ``LISTENING`` after a barge-in drain
and to ``TERMINATED`` after a shutdown drain.

==================  =======================  ==================================
source              trigger                  target
==================  =======================  ==================================
IDLE                CONNECT                  CONNECTING
CONNECTING          CONNECTED                LISTENING
LISTENING           USER_TURN_ENDED          PROCESSING_THOUGHT
PROCESSING_THOUGHT  RESPONSE_AUDIO_STARTED   SPEAKING
PROCESSING_THOUGHT  RESPONSE_SKIPPED         LISTENING
SPEAKING            RESPONSE_AUDIO_FINISHED  LISTENING
PROCESSING_THOUGHT  BARGE_IN                 BARGED_IN
SPEAKING            BARGE_IN                 BARGED_IN
BARGED_IN           DRAIN                    DRAINING (reason = BARGE_IN)
DRAINING            DRAIN_COMPLETE           LISTENING  if reason = BARGE_IN
DRAINING            DRAIN_COMPLETE           TERMINATED if reason = SHUTDOWN
IDLE, CONNECTING    CLOSE                    TERMINATED (nothing to drain)
LISTENING .. DRAIN  CLOSE                    DRAINING (reason = SHUTDOWN)
any non-terminal    FAULT                    TERMINATED
TERMINATED          *                        rejected (absorbing)
==================  =======================  ==================================

Every pair not listed is rejected.
"""

from __future__ import annotations

from enum import StrEnum, unique
from types import MappingProxyType
from typing import Final

__all__ = [
    "ACTIVE_OUTPUT_STATES",
    "TERMINAL_STATES",
    "DrainReason",
    "SessionState",
    "SessionTrigger",
    "drain_reason_for",
    "resolve_transition",
    "transition_table",
]


@unique
class SessionState(StrEnum):
    """Lifecycle states of one full-duplex voice session."""

    IDLE = "idle"
    CONNECTING = "connecting"
    LISTENING = "listening"
    PROCESSING_THOUGHT = "processing_thought"
    SPEAKING = "speaking"
    BARGED_IN = "barged_in"
    DRAINING = "draining"
    TERMINATED = "terminated"


@unique
class SessionTrigger(StrEnum):
    """Inputs that drive the session state machine."""

    CONNECT = "connect"
    CONNECTED = "connected"
    USER_TURN_ENDED = "user_turn_ended"
    RESPONSE_AUDIO_STARTED = "response_audio_started"
    RESPONSE_SKIPPED = "response_skipped"
    RESPONSE_AUDIO_FINISHED = "response_audio_finished"
    BARGE_IN = "barge_in"
    DRAIN = "drain"
    DRAIN_COMPLETE = "drain_complete"
    CLOSE = "close"
    FAULT = "fault"


@unique
class DrainReason(StrEnum):
    """Why the session is in ``DRAINING``; selects the ``DRAIN_COMPLETE`` target."""

    BARGE_IN = "barge_in"
    SHUTDOWN = "shutdown"


TERMINAL_STATES: Final = frozenset({SessionState.TERMINATED})
ACTIVE_OUTPUT_STATES: Final = frozenset({SessionState.PROCESSING_THOUGHT, SessionState.SPEAKING})
"""States in which detected user speech constitutes a barge-in."""

_S = SessionState
_T = SessionTrigger

_DRAINABLE: Final = (_S.LISTENING, _S.PROCESSING_THOUGHT, _S.SPEAKING, _S.BARGED_IN, _S.DRAINING)

_UNGUARDED: Final[MappingProxyType[tuple[SessionState, SessionTrigger], SessionState]] = (
    MappingProxyType(
        {
            (_S.IDLE, _T.CONNECT): _S.CONNECTING,
            (_S.CONNECTING, _T.CONNECTED): _S.LISTENING,
            (_S.LISTENING, _T.USER_TURN_ENDED): _S.PROCESSING_THOUGHT,
            (_S.PROCESSING_THOUGHT, _T.RESPONSE_AUDIO_STARTED): _S.SPEAKING,
            (_S.PROCESSING_THOUGHT, _T.RESPONSE_SKIPPED): _S.LISTENING,
            (_S.SPEAKING, _T.RESPONSE_AUDIO_FINISHED): _S.LISTENING,
            (_S.PROCESSING_THOUGHT, _T.BARGE_IN): _S.BARGED_IN,
            (_S.SPEAKING, _T.BARGE_IN): _S.BARGED_IN,
            (_S.BARGED_IN, _T.DRAIN): _S.DRAINING,
            (_S.IDLE, _T.CLOSE): _S.TERMINATED,
            (_S.CONNECTING, _T.CLOSE): _S.TERMINATED,
            **dict.fromkeys(((state, _T.CLOSE) for state in _DRAINABLE), _S.DRAINING),
            **dict.fromkeys(
                ((state, _T.FAULT) for state in _S if state not in TERMINAL_STATES),
                _S.TERMINATED,
            ),
        }
    )
)

_DRAIN_TARGETS: Final[MappingProxyType[DrainReason, SessionState]] = MappingProxyType(
    {DrainReason.BARGE_IN: _S.LISTENING, DrainReason.SHUTDOWN: _S.TERMINATED}
)

_DRAIN_REASON_BY_TRIGGER: Final[MappingProxyType[SessionTrigger, DrainReason]] = MappingProxyType(
    {_T.DRAIN: DrainReason.BARGE_IN, _T.CLOSE: DrainReason.SHUTDOWN}
)


def resolve_transition(
    state: SessionState, trigger: SessionTrigger, drain_reason: DrainReason | None
) -> SessionState | None:
    """Return the target of ``state --trigger-->``, or ``None`` if the pair is rejected.

    ``drain_reason`` is only consulted for ``DRAINING --DRAIN_COMPLETE-->``.
    """
    if trigger is SessionTrigger.DRAIN_COMPLETE:
        if state is SessionState.DRAINING and drain_reason is not None:
            return _DRAIN_TARGETS[drain_reason]
        return None
    return _UNGUARDED.get((state, trigger))


def drain_reason_for(trigger: SessionTrigger, previous: DrainReason | None) -> DrainReason | None:
    """Drain reason in force after ``trigger`` lands the machine in ``DRAINING``.

    ``CLOSE`` always escalates to ``SHUTDOWN``; ``DRAIN`` starts a barge-in drain; any
    other trigger keeps ``previous``.
    """
    return _DRAIN_REASON_BY_TRIGGER.get(trigger, previous)


def transition_table() -> dict[
    tuple[SessionState, SessionTrigger, DrainReason | None], SessionState
]:
    """Enumerate every accepted ``(state, trigger, drain_reason)`` -> target triple.

    Unguarded transitions are listed with ``drain_reason=None``. Intended for docs,
    contract tests and visualisation -- not for the hot path.
    """
    table: dict[tuple[SessionState, SessionTrigger, DrainReason | None], SessionState] = {
        (state, trigger, None): target for (state, trigger), target in _UNGUARDED.items()
    }
    for reason, target in _DRAIN_TARGETS.items():
        table[SessionState.DRAINING, SessionTrigger.DRAIN_COMPLETE, reason] = target
    return table
