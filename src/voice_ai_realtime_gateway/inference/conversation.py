"""Conversation state with a bounded context window.

The history holds what was *said*, not what was *generated*: an interrupted agent
turn stores only the words the user heard (see :mod:`.playback`). The window is
bounded by both turn count and total characters. Pruning drops the oldest turns
first and never drops the newest turn, so the current exchange always survives.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final, override

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_TURNS",
    "AgentTurn",
    "ConversationState",
    "ToolCallTurn",
    "Turn",
    "UserTurn",
]

DEFAULT_MAX_TURNS: Final = 64
DEFAULT_MAX_CHARS: Final = 32_000


@dataclass(frozen=True, slots=True, kw_only=True)
class UserTurn:
    """A final user transcript.

    ``trace`` is the span of the client audio packets that carried the utterance.
    """

    text: str
    trace: TraceContext
    at_ns: int

    @property
    def chars(self) -> int:
        """Context-window cost of this turn."""
        return len(self.text)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTurn:
    """What the agent said in one response.

    Attributes:
        text: the words the user heard (the whole response if it completed).
        trace: the response span.
        interrupted: ``True`` if playback was cut short (barge-in, disconnect, fault).
        generated_chars: how much text the LLM had produced, heard or not.
    """

    text: str
    trace: TraceContext
    at_ns: int
    interrupted: bool
    generated_chars: int

    @property
    def chars(self) -> int:
        """Context-window cost of this turn."""
        return len(self.text)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallTurn:
    """A tool invocation made on the agent's behalf, with its result."""

    name: str
    arguments: str
    result: str
    trace: TraceContext
    at_ns: int

    @property
    def chars(self) -> int:
        """Context-window cost of this turn."""
        return len(self.name) + len(self.arguments) + len(self.result)


type Turn = UserTurn | AgentTurn | ToolCallTurn


@final
class ConversationState:
    """Ordered, bounded conversation history."""

    __slots__ = ("_chars", "_max_chars", "_max_turns", "_pruned", "_turns")

    def __init__(
        self, *, max_turns: int = DEFAULT_MAX_TURNS, max_chars: int = DEFAULT_MAX_CHARS
    ) -> None:
        """Create an empty history bounded by ``max_turns`` and ``max_chars``.

        Raises:
            ValueError: if a bound is not positive.
        """
        if max_turns < 1 or max_chars < 1:
            msg = f"bounds must be positive, got max_turns={max_turns}, max_chars={max_chars}"
            raise ValueError(msg)
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._turns: deque[Turn] = deque()
        self._chars = 0
        self._pruned = 0

    def __len__(self) -> int:
        """Number of retained turns."""
        return len(self._turns)

    @property
    def turns(self) -> tuple[Turn, ...]:
        """Retained turns, oldest first."""
        return tuple(self._turns)

    @property
    def total_chars(self) -> int:
        """Context-window cost of the retained turns."""
        return self._chars

    @property
    def pruned(self) -> int:
        """Turns dropped to honour the window bounds."""
        return self._pruned

    def append(self, turn: Turn) -> None:
        """Add ``turn`` and prune the oldest turns beyond the window."""
        self._turns.append(turn)
        self._chars += turn.chars
        while len(self._turns) > 1 and (
            len(self._turns) > self._max_turns or self._chars > self._max_chars
        ):
            self._chars -= self._turns.popleft().chars
            self._pruned += 1

    def history_before(self, turn: Turn) -> tuple[Turn, ...]:
        """Turns preceding ``turn`` (by identity), i.e. the context it was said in.

        If ``turn`` is not retained (pruned or foreign), the whole window is returned.
        """
        turns = self._turns
        for index in range(len(turns) - 1, -1, -1):
            if turns[index] is turn:
                return tuple(itertools.islice(turns, index))
        return tuple(turns)

    @override
    def __repr__(self) -> str:
        return (
            f"ConversationState(turns={len(self._turns)}, chars={self._chars}, "
            f"pruned={self._pruned})"
        )
