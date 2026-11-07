"""Domain error hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.domain.states import SessionState, SessionTrigger

__all__ = ["InvalidTransitionError", "SessionError", "SessionStateError", "StaleSpanError"]


class SessionError(Exception):
    """Base class for session domain errors."""


class InvalidTransitionError(SessionError):
    """A trigger is not accepted in the current state. The machine is left unchanged."""

    def __init__(self, state: SessionState, trigger: SessionTrigger) -> None:
        """Record the rejected ``(state, trigger)`` pair."""
        super().__init__(f"trigger {trigger.value!r} is not valid in state {state.value!r}")
        self.state = state
        self.trigger = trigger


class SessionStateError(SessionError):
    """An operation was requested in a session state that does not permit it."""


class StaleSpanError(SessionError):
    """A frame was submitted for an audio span that is no longer current."""
