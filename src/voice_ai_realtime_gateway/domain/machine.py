"""Session state machine with Lamport-stamped, trace-correlated transitions.

Guarantees (asserted by the contract and fuzz suites):

1. Only pairs accepted by :func:`~.states.resolve_transition` are ever committed.
2. A rejected trigger is side-effect free: state, drain reason, Lamport clock and
   history are untouched (a :class:`TransitionRejected` event is still emitted).
3. Committed transitions carry strictly increasing Lamport timestamps, also when merging
   remote ``observed`` timestamps.
4. ``TERMINATED`` is absorbing.
5. Memory is O(``history_size``) for the life of the session.

Concurrency model: :meth:`SessionStateMachine.fire` is synchronous and never awaits, so
each transition is atomic with respect to every other coroutine on the loop.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, final, override

from voice_ai_realtime_gateway.domain.clock import LamportClock
from voice_ai_realtime_gateway.domain.errors import InvalidTransitionError
from voice_ai_realtime_gateway.domain.states import (
    TERMINAL_STATES,
    DrainReason,
    SessionState,
    SessionTrigger,
    drain_reason_for,
    resolve_transition,
)
from voice_ai_realtime_gateway.telemetry.events import StateTransitioned, TransitionRejected

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink

__all__ = ["SessionStateMachine", "Transition"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Transition:
    """One committed state change."""

    source: SessionState
    target: SessionState
    trigger: SessionTrigger
    lamport: int
    at_ns: int
    trace: TraceContext
    drain_reason: DrainReason | None


@final
class SessionStateMachine:
    """Formal full-duplex session lifecycle."""

    __slots__ = ("_clock", "_drain_reason", "_history", "_lamport", "_sink", "_state", "_trace")

    def __init__(
        self,
        *,
        trace: TraceContext,
        sink: TelemetrySink,
        clock: MonotonicClock = time.monotonic_ns,
        lamport: LamportClock | None = None,
        history_size: int = 64,
    ) -> None:
        """Create a machine in ``IDLE``.

        Raises:
            ValueError: if ``history_size`` is not positive.
        """
        if history_size < 1:
            msg = f"history_size must be >= 1, got {history_size}"
            raise ValueError(msg)
        self._trace = trace
        self._sink = sink
        self._clock = clock
        self._lamport = LamportClock() if lamport is None else lamport
        self._state = SessionState.IDLE
        self._drain_reason: DrainReason | None = None
        self._history: deque[Transition] = deque(maxlen=history_size)

    @property
    def state(self) -> SessionState:
        """Current state."""
        return self._state

    @property
    def drain_reason(self) -> DrainReason | None:
        """Why the session is draining, or ``None`` outside ``DRAINING``."""
        return self._drain_reason

    @property
    def terminated(self) -> bool:
        """Whether the absorbing ``TERMINATED`` state was reached."""
        return self._state in TERMINAL_STATES

    @property
    def lamport(self) -> int:
        """Lamport timestamp of the most recent committed transition (0 if none)."""
        return self._lamport.value

    @property
    def trace(self) -> TraceContext:
        """Correlation context stamped on subsequent transitions."""
        return self._trace

    @property
    def history(self) -> tuple[Transition, ...]:
        """Most recent committed transitions, oldest first (bounded)."""
        return tuple(self._history)

    def open_span(self) -> TraceContext:
        """Start a new audio span in the same session and make it current."""
        self._trace = self._trace.child()
        return self._trace

    def can_fire(self, trigger: SessionTrigger) -> bool:
        """Whether ``trigger`` would be accepted right now."""
        return resolve_transition(self._state, trigger, self._drain_reason) is not None

    def fire(self, trigger: SessionTrigger, *, observed: int | None = None) -> Transition:
        """Apply ``trigger`` atomically.

        Args:
            trigger: input to apply.
            observed: Lamport timestamp carried by a remote message that caused this
                trigger; merged as ``max(local, observed) + 1``.

        Returns:
            The committed :class:`Transition`.

        Raises:
            ValueError: if ``observed`` is negative (nothing is changed).
            InvalidTransitionError: if the pair is rejected (nothing is changed).
        """
        lamport = self._lamport.peek(observed)
        source = self._state
        target = resolve_transition(source, trigger, self._drain_reason)
        now = self._clock()
        if target is None:
            self._sink.emit(
                TransitionRejected(
                    trace=self._trace,
                    at_ns=now,
                    lamport=self._lamport.value,
                    state=source,
                    trigger=trigger,
                )
            )
            raise InvalidTransitionError(source, trigger)
        self._lamport.tick(observed)
        self._state = target
        self._drain_reason = (
            drain_reason_for(trigger, self._drain_reason)
            if target is SessionState.DRAINING
            else None
        )
        transition = Transition(
            source=source,
            target=target,
            trigger=trigger,
            lamport=lamport,
            at_ns=now,
            trace=self._trace,
            drain_reason=self._drain_reason,
        )
        self._history.append(transition)
        self._sink.emit(
            StateTransitioned(
                trace=self._trace,
                at_ns=now,
                lamport=lamport,
                source=source,
                target=target,
                trigger=trigger,
            )
        )
        return transition

    @override
    def __repr__(self) -> str:
        return (
            f"SessionStateMachine(state={self._state.value}, lamport={self._lamport.value}, "
            f"session={self._trace.session_id})"
        )
