"""Session domain: states, transition matrix, Lamport clock, duplex controller."""

from voice_ai_realtime_gateway.domain.clock import LamportClock, MonotonicClock
from voice_ai_realtime_gateway.domain.duplex import (
    BARGE_IN_BUDGET_NS,
    DEFAULT_EGRESS_RING_BYTES,
    DEFAULT_EGRESS_RING_FRAMES,
    AudioEgress,
    BargeInReport,
    DuplexSession,
)
from voice_ai_realtime_gateway.domain.errors import (
    InvalidTransitionError,
    SessionError,
    SessionStateError,
    StaleSpanError,
)
from voice_ai_realtime_gateway.domain.machine import (
    SessionStateMachine,
    Transition,
    TransitionListener,
)
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

__all__ = [
    "ACTIVE_OUTPUT_STATES",
    "BARGE_IN_BUDGET_NS",
    "DEFAULT_EGRESS_RING_BYTES",
    "DEFAULT_EGRESS_RING_FRAMES",
    "TERMINAL_STATES",
    "AudioEgress",
    "BargeInReport",
    "DrainReason",
    "DuplexSession",
    "InvalidTransitionError",
    "LamportClock",
    "MonotonicClock",
    "SessionError",
    "SessionState",
    "SessionStateError",
    "SessionStateMachine",
    "SessionTrigger",
    "StaleSpanError",
    "Transition",
    "TransitionListener",
    "drain_reason_for",
    "resolve_transition",
    "transition_table",
]
