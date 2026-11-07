"""Logical and physical clocks.

* :class:`LamportClock` orders causally related events across the gateway and its peers
  (client signalling, model server). Every committed transition consumes exactly one
  tick, so timestamps within a session are *strictly* increasing.
* :data:`MonotonicClock` is an injectable ``() -> int`` nanosecond source used for
  latency measurement (defaults to :func:`time.monotonic_ns`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final, override

__all__ = ["LamportClock", "MonotonicClock"]

type MonotonicClock = Callable[[], int]


@final
class LamportClock:
    """Lamport logical clock (Lamport, 1978)."""

    __slots__ = ("_value",)

    def __init__(self, start: int = 0) -> None:
        """Start the clock at ``start`` (must be non-negative)."""
        if start < 0:
            msg = f"Lamport clock must start non-negative, got {start}"
            raise ValueError(msg)
        self._value = start

    @property
    def value(self) -> int:
        """Last issued timestamp."""
        return self._value

    def peek(self, observed: int | None = None) -> int:
        """Return the timestamp :meth:`tick` would issue, without advancing.

        Raises:
            ValueError: if ``observed`` is negative.
        """
        if observed is None:
            return self._value + 1
        if observed < 0:
            msg = f"observed Lamport timestamp must be non-negative, got {observed}"
            raise ValueError(msg)
        return max(self._value, observed) + 1

    def tick(self, observed: int | None = None) -> int:
        """Advance for a local event, merging a remote ``observed`` timestamp if given.

        Implements ``L := max(L, observed) + 1``.

        Raises:
            ValueError: if ``observed`` is negative.
        """
        self._value = self.peek(observed)
        return self._value

    @override
    def __repr__(self) -> str:
        return f"LamportClock(value={self._value})"
