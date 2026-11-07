from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from voice_ai_realtime_gateway.domain.clock import LamportClock


def test_clock_starts_at_zero_and_ticks_by_one() -> None:
    clock = LamportClock()
    assert clock.value == 0
    assert clock.tick() == 1
    assert clock.tick() == 2
    assert repr(clock) == "LamportClock(value=2)"


def test_clock_rejects_negative_start_and_observations() -> None:
    with pytest.raises(ValueError, match="start"):
        LamportClock(-1)
    clock = LamportClock(5)
    with pytest.raises(ValueError, match="observed"):
        clock.tick(-1)
    assert clock.value == 5, "rejected observation leaves the clock untouched"


def test_peek_does_not_advance() -> None:
    clock = LamportClock(3)
    assert clock.peek() == 4
    assert clock.peek(10) == 11
    assert clock.value == 3


@given(st.lists(st.one_of(st.none(), st.integers(min_value=0, max_value=10**9)), max_size=100))
def test_ticks_are_strictly_monotonic_and_dominate_observations(
    observations: list[int | None],
) -> None:
    clock = LamportClock()
    previous = clock.value
    for observed in observations:
        stamp = clock.tick(observed)
        assert stamp > previous
        if observed is not None:
            assert stamp > observed, "happened-before: receive is after send"
        previous = stamp
